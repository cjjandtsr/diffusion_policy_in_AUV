#!/usr/bin/env python3
"""Full fixed-bank lightweight endpoint evaluation for E5 BC versus DPPO.

This runner intentionally reuses the formal clean-rebuild construction and
checkpoint restore path.  It does not use the legacy ``eval_ddpm_paired.py``
loader, and it lives outside the checkpoint-hashed package.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Mapping

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.formal_endpoint_eval_core import (  # noqa: E402
    PairedEpisodeRecord,
    PairedEvaluationBank,
    write_endpoint_artifacts,
)


BENCHMARK_SPLITS = {
    "fixed32": ("rl_test", "layerA"),
    "routea-val-test": ("rl_val", "rl_test"),
}
FORMAL_SPLITS = BENCHMARK_SPLITS["fixed32"]
EXPECTED_WORLDS = {"rl_val": 20, "rl_test": 12, "layerA": 20}
EXPECTED_EPISODES = {"rl_val": 80, "rl_test": 48, "layerA": 80}
FORMAL_ROLES = frozenset(
    ("anticipatory", "dense_recovery", "support_expansion", "yaw_lateral_recovery")
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CanonicalEpisodeConfig:
    payload: Mapping[str, object]
    canonical_json: str
    config_hash: str
    seed: int


def canonical_episode_config(config: Mapping[str, object]) -> CanonicalEpisodeConfig:
    required = {
        "episode_id",
        "world_name",
        "goal_pos",
        "start_pos",
        "start_yaw",
        "z_range",
        "disturb_enable",
        "role",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("episode config missing fields: {}".format(missing))
    payload = dict(config)
    role = str(payload["role"])
    if role not in FORMAL_ROLES:
        raise ValueError("unknown formal evaluation role: {}".format(role))
    for name in ("start_pos", "goal_pos"):
        value = np.asarray(payload[name], dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("{} must be a finite xyz triple".format(name))
    z_range = np.asarray(payload["z_range"], dtype=np.float64)
    if z_range.shape != (2,) or not np.isfinite(z_range).all() or z_range[0] >= z_range[1]:
        raise ValueError("z_range must be a finite increasing pair")
    if not np.isfinite(float(payload["start_yaw"])):
        raise ValueError("start_yaw must be finite")
    canonical_json = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    config_hash = _sha256_bytes(canonical_json.encode("utf-8"))
    # The source bank has no recorded RNG seed.  Bind one deterministically to
    # the complete episode config so both policies receive identical PC noise.
    seed = int(config_hash[:8], 16)
    return CanonicalEpisodeConfig(payload, canonical_json, config_hash, seed)


def reset_fixed_episode(environment, config, *, seed: int, reset_callable=None):
    """Make the recorded start/goal effective through the env's own reset path."""

    fixed = canonical_episode_config(config)
    start = np.asarray(fixed.payload["start_pos"], dtype=np.float32)
    goal = np.asarray(fixed.payload["goal_pos"], dtype=np.float32)
    yaw = float(fixed.payload["start_yaw"])
    z_min, z_max = (float(value) for value in fixed.payload["z_range"])
    environment._done_reason = "none"
    environment.mid_state_init_ratio = 0.0
    environment.disturb_enable = bool(fixed.payload["disturb_enable"])
    environment.z_min = z_min
    environment.z_max = z_max
    environment.z_target = (z_min + z_max) / 2.0
    resetter = reset_callable or environment.reset
    observation, info = resetter(
        seed=int(seed),
        options={
            "fixed_start_goal": {
                "start_pos": start.tolist(),
                "start_yaw": yaw,
                "goal_pos": goal.tolist(),
            }
        },
    )
    if environment._done_reason != "none":
        raise RuntimeError("environment reset retained a stale terminal reason")
    if not np.allclose(environment._drone_pos, start, rtol=0.0, atol=1e-6):
        raise RuntimeError("fixed start_pos was not applied")
    if not np.allclose(environment.goal_w, goal, rtol=0.0, atol=1e-6):
        raise RuntimeError("fixed goal_pos was not applied")
    if not np.isclose(environment._drone_yaw, yaw, rtol=0.0, atol=1e-7):
        raise RuntimeError("fixed start_yaw was not applied")
    return observation, info


def canonical_done_reason(raw_reason, z_world, terminated, truncated):
    raw = str(raw_reason or "none")
    if terminated:
        if raw == "invalid_state":
            return "invalid_state"
        if raw == "reached":
            return "reached"
        if raw == "out_of_bounds":
            return "out_of_bounds"
        if raw == "altitude_exceed":
            return "altitude_exceed"
        # KinematicUAVEnv may overwrite crash_ground with the generic
        # "collision" terminal label after calling its collision predicate.
        if raw == "crash_ground" or float(z_world) < 0.2:
            return "crash_ground"
        if raw == "collision" or raw.startswith(("collide(", "penetrate_flyover(")):
            return "collision"
        raise RuntimeError("unclassified terminated outcome: {!r}".format(raw))
    if truncated:
        return "time_limit"
    raise RuntimeError("episode ended without terminated/truncated")


class EpisodeAccumulator:
    def __init__(self, initial_position):
        initial = np.asarray(initial_position, dtype=np.float64)
        if initial.shape != (3,) or not np.isfinite(initial).all():
            raise ValueError("initial_position must be a finite xyz triple")
        self._positions = [initial]
        self._dz = []
        self._episode_return = 0.0
        self._steps = 0
        self._onset_first_step = None
        self._future_clearance = 0.0
        self._d_soft_h = []
        self._trap_recovery_count = 0
        self._loop_count = 0
        self._side_flip_count_max = 0
        self._invalid_steps = 0
        self._reward_schema = "legacy_unknown"
        self._v17_metrics = dict(onset_history_steps=0, onset_contrast_steps=0, onset_invalid_steps=0,
                                onset_history_paid_steps=0, onset_contrast_paid_steps=0,
                                above_demo_dwell_seconds=0., above_demo_unknown_seconds=0.,
                                above_demo_stall_seconds=0., positive_climb_m=0., executed_length_m=0.)

    def add_step(
        self,
        *,
        reward,
        position,
        executed_action,
        reward_terms,
        reward_metrics,
        terminal,
        executed_path=None,
    ):
        self._reward_schema = reward_metrics.get("reward_schema_version", self._reward_schema)
        if reward_metrics.get("invalid_state", False):
            if not terminal or not math.isfinite(float(reward)):
                raise ValueError("invalid state must be a finite diagnostic terminal")
            self._steps += 1
            self._invalid_steps += 1
            self._episode_return += float(reward)
            return  # Do not inject NaN or fake zeros into trajectory statistics.
        position = np.asarray(position, dtype=np.float64)
        action = np.asarray(executed_action, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("step position must be a finite xyz triple")
        if action.ndim != 2 or action.shape[1] < 3 or not np.isfinite(action).all():
            raise ValueError("executed_action must be a finite (steps, action_dim>=3) array")
        self._steps += 1
        self._episode_return += float(reward)
        if self._reward_schema == "v17":
            path = np.asarray(executed_path, dtype=np.float64)
            if path.shape != (5, 3) or not np.isfinite(path).all() or not np.array_equal(path[-1], position):
                raise ValueError("v17 evaluation requires the actual complete H4 executed path")
            if not np.array_equal(path[0], self._positions[-1]):
                raise ValueError("v17 evaluation executed path continuity mismatch")
            self._positions.extend(path[1:])
            mode = reward_metrics.get("onset_mode", "invalid")
            self._v17_metrics["onset_" + mode + "_steps"] += 1
            if mode in ("history", "contrast") and reward_terms.get("onset", 0.) > 0:
                self._v17_metrics["onset_" + mode + "_paid_steps"] += 1
            for key in ("above_demo_dwell_seconds", "above_demo_unknown_seconds", "above_demo_stall_seconds", "positive_climb_m", "executed_length_m"):
                self._v17_metrics[key] += float(reward_metrics.get(key, 0.))
            for key in ("demo_height_max", "goal_above_demo_height", "support_eligible_above_demo", "support_paid_above_demo"):
                if key in reward_metrics:
                    self._v17_metrics[key] = reward_metrics[key]
            for key in ("max_ground_relative_height", "max_goal_relative_height"):
                if key in reward_metrics:
                    self._v17_metrics[key] = max(self._v17_metrics.get(key, -math.inf), float(reward_metrics[key]))
            self._v17_metrics["above_demo_height"] = bool(self._v17_metrics.get("above_demo_height", False) or reward_metrics.get("above_demo_height", False))
        else:
            self._positions.append(position)
        self._dz.extend(float(value) for value in action[:, 2])
        self._future_clearance += float(reward_terms.get("future_clearance", 0.0))
        if bool(reward_metrics.get("onset_active", reward_terms.get("onset", 0.) > 0)) and self._onset_first_step is None:
            self._onset_first_step = self._steps
        # Terminal metrics use 0.0 as a sentinel rather than a measured soft-min.
        if (not terminal or self._reward_schema == "v17") and "d_soft_h" in reward_metrics:
            value = float(reward_metrics["d_soft_h"])
            if np.isfinite(value):
                self._d_soft_h.append(value)
        self._trap_recovery_count += int(bool(reward_metrics.get("trap_recovered", False)))
        self._loop_count += int(bool(reward_metrics.get("loop_detected", False)))
        self._side_flip_count_max = max(
            self._side_flip_count_max, int(reward_metrics.get("side_flip_count", 0))
        )

    def finish(self, *, reward_history):
        if self._steps <= 0 or (not self._dz and not self._invalid_steps):
            raise RuntimeError("cannot finish an empty episode")
        positions = np.stack(self._positions, axis=0)
        dz = np.asarray(self._dz, dtype=np.float64)
        support = reward_history.get("eligible_cells", reward_history.get("new_support_cells_this_episode", set()))
        v17 = dict(self._v17_metrics) if self._reward_schema == "v17" else None
        if v17 is not None:
            known = v17["above_demo_dwell_seconds"] - v17["above_demo_unknown_seconds"]
            v17.update(above_demo_stall_rate=v17["above_demo_stall_seconds"] / known if known > 0 else None,
                       above_demo_coverage=known / v17["above_demo_dwell_seconds"] if v17["above_demo_dwell_seconds"] > 0 else None,
                       observed_duration_seconds=(self._steps - self._invalid_steps) * 1.2,
                       invalid_missing_steps=self._invalid_steps,
                       onset_evidence="reward_activation_not_independent_event_timing")
        return {
            "episode_return": float(self._episode_return),
            "onset_triggered": self._onset_first_step is not None,
            "onset_first_step": self._onset_first_step,
            "future_clearance_reward_sum": float(self._future_clearance),
            "d_soft_h_min": float(min(self._d_soft_h)) if self._d_soft_h else 0.0,
            "d_soft_h_steps": len(self._d_soft_h),
            "support_cells": len(support),
            "terminal_xyz": tuple(float(value) for value in positions[-1]),
            "min_xyz": tuple(float(value) for value in positions.min(axis=0)),
            "max_xyz": tuple(float(value) for value in positions.max(axis=0)),
            "action_dz_mean": float(dz.mean()) if len(dz) else None,
            "action_dz_min": float(dz.min()) if len(dz) else None,
            "action_dz_max": float(dz.max()) if len(dz) else None,
            "trap_recovery_count": self._trap_recovery_count,
            "loop_count": self._loop_count,
            "side_flip_count_max": self._side_flip_count_max,
            "steps": self._steps,
            "invalid_steps": self._invalid_steps,
            "reward_schema_version": self._reward_schema,
            "trajectory_statistics_scope": "partial_before_invalid" if self._invalid_steps else ("complete_executed_polyline" if v17 is not None else "complete_decision_endpoints"),
            "v17_metrics": v17,
        }


def _load_fixed_bank(config_dir: Path, benchmark="fixed32"):
    episodes = []
    seen = set()
    for split in BENCHMARK_SPLITS[benchmark]:
        path = config_dir / "episode_configs_{}.json".format(split)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or len(payload) != EXPECTED_WORLDS[split]:
            raise ValueError("{} bank must contain {} worlds".format(split, EXPECTED_WORLDS[split]))
        split_count = 0
        for world_id in sorted(payload):
            if "/" in world_id or "\\" in world_id:
                raise ValueError("world ID must not contain a path separator")
            if split == "layerA":
                valid_name = re.fullmatch(r"env_\d{3}_obs\d+", world_id) is not None
            else:
                valid_name = "__{}__".format(split) in world_id
            if not valid_name:
                raise ValueError("world does not belong to {}: {}".format(split, world_id))
            if not isinstance(payload[world_id], list) or len(payload[world_id]) != 4:
                raise ValueError("fixed benchmark requires 4 episodes per world: {}".format(world_id))
            for raw_config in payload[world_id]:
                fixed = canonical_episode_config(raw_config)
                if fixed.payload["world_name"] != world_id:
                    raise ValueError("episode world_name does not match its bank key")
                config_id = str(fixed.payload["episode_id"])
                if config_id in seen:
                    raise ValueError("duplicate episode_id: {}".format(config_id))
                seen.add(config_id)
                episodes.append((split, world_id, fixed))
                split_count += 1
        if split_count != EXPECTED_EPISODES[split]:
            raise ValueError(
                "{} fixed bank must contain {} episodes, got {}".format(
                    split, EXPECTED_EPISODES[split], split_count
                )
            )
    return tuple(episodes)


def _validate_bank_sources(fixed_bank, manifest_entries):
    """Route A uses admission; historical layerA uses its fixed config bank."""

    entries = tuple(manifest_entries)
    training_worlds = {
        str(entry["world_uid"]) for entry in entries if entry.get("split") == "rl_train"
    }
    bank_worlds = {world_id for _, world_id, _ in fixed_bank}
    if bank_worlds & training_worlds:
        raise ValueError("evaluation bank overlaps admitted training worlds")
    for split in sorted({split for split, _, _ in fixed_bank} - {"layerA"}):
        admitted = {
            str(entry["world_uid"]) for entry in entries if entry.get("split") == split
        }
        selected = {world_id for item_split, world_id, _ in fixed_bank if item_split == split}
        if selected != admitted:
            raise ValueError("{} bank must exactly match its admitted worlds".format(split))
        admitted_roles = {
            (str(entry["world_uid"]), str(entry.get("role") or entry.get("role_active")))
            for entry in entries if entry.get("split") == split
        }
        for item_split, world_id, fixed in fixed_bank:
            if item_split == split and (world_id, str(fixed.payload["role"])) not in admitted_roles:
                raise ValueError("{} world role differs from admission: {}".format(split, world_id))


def _resolve_obstacle(root, split, world_id):
    # Legacy scan_world_pool uses env_*.obstacles.json; Route A worldgen uses
    # <uid>.world.obstacles.json. Never silently choose differing duplicates.
    suffixes = (".obstacles.json", ".world.obstacles.json") if split == "layerA" else (".world.obstacles.json",)
    candidates = [Path(root) / (world_id + suffix) for suffix in suffixes]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError("missing {} obstacle asset for {} under {}".format(split, world_id, root))
    if len({sha256_file(path) for path in existing}) != 1:
        raise ValueError("ambiguous obstacle assets for {}: {}".format(world_id, existing))
    return existing[0]


class _E5Policy:
    def __init__(self, system, *, use_base_policy, identity_hash):
        self.system = system
        self.use_base_policy = bool(use_base_policy)
        self.identity_hash = str(identity_hash)
        self.last_provenance = None

    def predict(self, observation, observation_context, *, action_seed):
        import torch

        torch.manual_seed(int(action_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(action_seed))
        with torch.no_grad():
            encoded = self.system.observation_adapter.encode(
                observation,
                observation_context["pointcloud_meta_history"],
                episode_identity_hash=observation_context["episode_identity_hash"],
                decision_index=observation_context["decision_index"],
            )
            self.system.observation_adapter.verify(encoded)
            self.last_provenance = encoded.provenance
            condition = encoded.condition
            samples = self.system.official_runtime(
                cond=condition,
                deterministic=True,
                # Frozen upstream DDPM appends the initial sample before its
                # return_chain guard.  Request the chain and discard it below;
                # this preserves the exact final trajectory while avoiding the
                # vendor's None.append failure without changing hashed code.
                return_chain=True,
                use_base_policy=self.use_base_policy,
            )
            full = self.system.bc_runtime.policy.normalizer["action"].unnormalize(
                samples.trajectories
            )
        from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ActionBundle

        return ActionBundle.from_full(full)


def _build_adapter(system, config, world_id, obstacle_path, role):
    from omegaconf import OmegaConf
    from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
    from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.kinematic_adapter import (
        KinematicAdapter,
    )

    reward_config = OmegaConf.to_container(config.reward, resolve=True)
    environment = KinematicUAVEnv(
        world_id=4,
        world_pool=[(world_id, str(obstacle_path))],
        synth_pc_factory=lambda path: SynthPointCloud(
            path, device=str(system.bc_runtime.device)
        ),
        rl_action_steps=8,
        n_exec_steps=4,
        max_episode_steps=int(config.environment.max_episode_steps),
        disturb_enable=False,
        mid_state_init_ratio=0.0,
        curriculum_stage="C2",
        episode_role=role,
        reward_version=getattr(system.reward, "version", "v13"),
        reward_v13_cfg=reward_config,
        reward_v17_cfg=reward_config if getattr(system.reward, "version", "v13") == "v17" else None,
        offline_density=getattr(system.reward, "offline_density", None),
        gamma=reward_config.get("gamma_pbrs", .99),
        seed=int(config.run.seed),
        terminate_at_space_limits=config.get("task_protocol") is None,
        space_cost_budget=(None if config.get("task_protocol") is None else
                           float(config.task_protocol.space_cost_budget)),
    )
    state = system.reward.initial_state(reward_config)
    return KinematicAdapter(environment, system.reward, state, reward_config)


def _action_seed(config_hash, decision_step):
    return int(
        _sha256_bytes("{}:{}".format(config_hash, decision_step).encode("ascii"))[:8],
        16,
    )


def _evaluate_one(
    *,
    policy_name,
    policy,
    adapter,
    fixed,
    split,
    world_id,
    obstacle_hash,
    source_manifest_hash,
):
    observation, observation_context = reset_fixed_episode(
        adapter.environment,
        fixed.payload,
        seed=fixed.seed,
        reset_callable=adapter.reset,
    )
    policy.last_provenance = None
    accumulator = EpisodeAccumulator(adapter.environment._drone_pos.copy())
    terminal_info = None
    observation_rejection = None
    last_input_context = observation_context
    while True:
        last_input_context = observation_context
        try:
            action = policy.predict(
                observation,
                observation_context,
                action_seed=_action_seed(fixed.config_hash, accumulator._steps),
            )
        except Exception as exc:
            from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.observation_provenance import (
                ObservationIdentityMismatch,
                PointCloudRejected,
            )
            if not isinstance(exc, (ObservationIdentityMismatch, PointCloudRejected)):
                raise
            from diffusion_policy_3d.env.reward_v17 import invalid_result
            total, terms, metrics, history = invalid_result(
                adapter.environment._reward_history,
                adapter.environment.reward_v17_cfg,
            )
            adapter.environment._reward_history = history
            metrics["invalid_detail"] = "observation_gate:{}".format(str(exc))
            accumulator.add_step(
                reward=total,
                position=adapter.environment._drone_pos,
                executed_action=None,
                reward_terms=terms,
                reward_metrics=metrics,
                terminal=True,
                executed_path=None,
            )
            observation_rejection = {
                "type": type(exc).__name__,
                "reason": str(exc),
            }
            terminal_info = ("invalid_state", True, False)
            break
        observation, reward, terminated, truncated, info = adapter.step(action)
        observation_context = info
        facts = info.get("transition_facts")
        raw_reason = (
            getattr(facts, "raw_done_reason", None)
            or getattr(adapter.environment, "_done_reason", None)
            or info.get("done_reason")
            or "none"
        )
        terminal = bool(terminated or truncated)
        accumulator.add_step(
            reward=reward,
            position=adapter.environment._drone_pos,
            executed_action=action.rollout_executed[0].detach().cpu().numpy(),
            reward_terms=info.get("reward_terms", {}),
            reward_metrics=info.get("reward_metrics", {}),
            terminal=terminal,
            executed_path=info.get("executed_path"),
        )
        if terminal:
            terminal_info = (raw_reason, terminated, truncated)
            break
    raw_reason, terminated, truncated = terminal_info
    reason = canonical_done_reason(
        raw_reason,
        adapter.environment._drone_pos[2],
        terminated,
        truncated,
    )
    if reason == "crash_ground":
        raw_reason = "crash_ground"
    summary = accumulator.finish(reward_history=adapter.reward_state.history)
    goal_xyz = tuple(float(value) for value in adapter.environment.goal_w)
    terminal_xyz = tuple(float(value) for value in summary["terminal_xyz"])
    endpoint_delta = np.asarray(terminal_xyz) - np.asarray(goal_xyz)
    safety_flags = tuple(
        name for name, active in (
            ("collision", reason == "collision"),
            ("crash_ground", reason == "crash_ground"),
            ("out_of_bounds", reason == "out_of_bounds"),
            ("altitude_exceed", reason == "altitude_exceed"),
            ("invalid_state", reason == "invalid_state"),
        ) if active
    )
    metadata_history = last_input_context.get("pointcloud_meta_history", ())
    last_meta = metadata_history[-1] if metadata_history else None
    provenance = policy.last_provenance
    return PairedEpisodeRecord(
        task_id=str(fixed.payload["episode_id"]),
        config_id=str(fixed.payload["episode_id"]),
        config_hash=fixed.config_hash,
        obstacle_hash=obstacle_hash,
        source_manifest_hash=source_manifest_hash,
        policy_identity_hash=policy.identity_hash,
        world_id=world_id,
        split=split,
        role=str(fixed.payload["role"]),
        seed=fixed.seed,
        policy=policy_name,
        done_reason=reason,
        raw_done_reason=str(raw_reason),
        goal_xyz=goal_xyz,
        final_distance_3d=float(np.linalg.norm(endpoint_delta)),
        final_error_xy=float(np.linalg.norm(endpoint_delta[:2])),
        final_error_z=float(abs(endpoint_delta[2])),
        terminal_safety_flags=safety_flags,
        observation_validity_summary={
            "status": "rejected" if observation_rejection else "valid",
            "rejection": observation_rejection,
            "last_raw_frame_hash": last_meta.raw_array_hash if last_meta else None,
            "last_no_ray_hit_fraction": float(last_meta.no_ray_hit_fraction) if last_meta else None,
            "last_policy_input_hash": provenance.policy_input_hash if provenance else None,
            "last_normalized_history_hash": provenance.normalized_history_hash if provenance else None,
        },
        **summary,
    )


def _rl_policy_identity(checkpoint_dir: Path) -> str:
    payload = {
        "header_sha256": sha256_file(checkpoint_dir / "header.json"),
        "integrity_sha256": sha256_file(checkpoint_dir / "SHA256.json"),
    }
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    )


def run(args):
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.build_system import (
        build_system,
    )
    from diffusion_policy_3d.rl.dppo_uav_rebuild.deployment.loader import (
        load_deployment_policy,
    )
    from diffusion_policy_3d.rl.dppo_uav.integration import manifest_eval_entries

    config_path = args.config.resolve()
    checkpoint_dir = args.rl_checkpoint.resolve()
    config_dir = args.eval_config_dir.resolve()
    output_dir = args.output_dir.resolve()
    config = OmegaConf.load(config_path)
    benchmark = args.benchmark
    splits = BENCHMARK_SPLITS[benchmark]
    fixed_bank = _load_fixed_bank(config_dir, benchmark)
    manifest_path = Path(config.admission_manifest)
    if not manifest_path.is_absolute():
        manifest_path = (_REPO_ROOT / manifest_path).resolve()
    source_manifest_hash = sha256_file(manifest_path)

    _validate_bank_sources(fixed_bank, manifest_eval_entries(str(manifest_path)))
    config_hashes = {
        split: sha256_file(config_dir / "episode_configs_{}.json".format(split))
        for split in splits
    }
    roots = {
        "rl_val": args.worlds_val,
        "rl_test": args.worlds_test,
        "layerA": args.worlds_layera,
    }
    obstacles = {
        (split, world_id): _resolve_obstacle(roots[split].resolve(), split, world_id)
        for split, world_id in sorted({(split, world_id) for split, world_id, _ in fixed_bank})
    }
    for name in ("paired_episodes.jsonl", "endpoint_report.json", "evaluation_provenance.json"):
        path = output_dir / name
        if path.exists() or path.with_suffix(path.suffix + ".tmp").exists():
            raise FileExistsError("refusing to overwrite endpoint artifact: {}".format(path))

    bc_system = build_system(config, _REPO_ROOT)
    deployment = load_deployment_policy(config, checkpoint_dir, _REPO_ROOT)
    if int(deployment.checkpoint_header.update_index) != int(args.expected_update):
        raise RuntimeError(
            "expected checkpoint update {}, got {}".format(
                args.expected_update, deployment.checkpoint_header.update_index
            )
        )
    bc_policy = _E5Policy(
        bc_system,
        use_base_policy=True,
        identity_hash=bc_system.bc_runtime.identity.content_hash,
    )
    rl_policy = _E5Policy(
        deployment.system,
        use_base_policy=False,
        identity_hash=_rl_policy_identity(checkpoint_dir),
    )

    records = []
    obstacle_sources = {}
    total = len(fixed_bank)
    for index, (split, world_id, fixed) in enumerate(fixed_bank, start=1):
        obstacle = obstacles[(split, world_id)]
        obstacle_hash = sha256_file(obstacle)
        obstacle_sources[world_id] = {"split": split, "path": str(obstacle), "sha256": obstacle_hash}
        # layerA is not claimed to be admitted by the Route A manifest.
        episode_source_hash = config_hashes[split] if split == "layerA" else source_manifest_hash
        role = str(fixed.payload["role"])
        bc_adapter = _build_adapter(bc_system, config, world_id, obstacle, role)
        rl_adapter = _build_adapter(deployment.system, config, world_id, obstacle, role)
        try:
            bc_record = _evaluate_one(
                policy_name="bc",
                policy=bc_policy,
                adapter=bc_adapter,
                fixed=fixed,
                split=split,
                world_id=world_id,
                obstacle_hash=obstacle_hash,
                source_manifest_hash=episode_source_hash,
            )
            rl_record = _evaluate_one(
                policy_name="rl",
                policy=rl_policy,
                adapter=rl_adapter,
                fixed=fixed,
                split=split,
                world_id=world_id,
                obstacle_hash=obstacle_hash,
                source_manifest_hash=episode_source_hash,
            )
            records.extend((bc_record, rl_record))
        finally:
            bc_adapter.environment.close()
            rl_adapter.environment.close()
        print(
            "[{}/{}] {} BC={} RL={} dR={:+.3f}".format(
                index,
                total,
                fixed.payload["episode_id"],
                bc_record.done_reason,
                rl_record.done_reason,
                rl_record.episode_return - bc_record.episode_return,
            ),
            flush=True,
        )

    bank = PairedEvaluationBank(records)
    provenance = {
        "schema_version": 2,
        "benchmark": benchmark,
        "evidence_scope": "lightweight_kinematic",
        "gazebo_px4_evidence": False,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_header": asdict(deployment.checkpoint_header),
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": source_manifest_hash,
        "source_binding_by_split": {
            split: "fixed_episode_config_file" if split == "layerA" else "routea_admission_manifest"
            for split in splits
        },
        "obstacle_sources": obstacle_sources,
        "world_count": len(obstacles),
        "episodes_per_policy": len(fixed_bank),
        "episode_pairs_per_world": 4,
        "evaluation_source_sha256": {
            "scripts/evaluate_dppo_uav_formal_endpoint.py": sha256_file(Path(__file__)),
            "scripts/formal_endpoint_eval_core.py": sha256_file(_REPO_ROOT / "scripts/formal_endpoint_eval_core.py"),
        },
        "fixed_config_files": {
            split: {
                "path": str(config_dir / "episode_configs_{}.json".format(split)),
                "sha256": config_hashes[split],
                "worlds": EXPECTED_WORLDS[split],
                "episodes": EXPECTED_EPISODES[split],
            }
            for split in splits
        },
        "seed_note": "No RNG seeds are stored in these episode files; retain the prior content-bound pairing rule, not an unverified reconstruction of master seed 42.",
        "derived_episode_seed_rule": "uint32(sha256(canonical_episode_config)[:8])",
        "paired_action_noise_rule": "uint32(sha256(config_hash:decision_step)[:8])",
        "action_protocol": "full16 -> policy[2:10] -> executed[2:6] (4x11)",
        "bc_use_base_policy": True,
        "rl_use_base_policy": False,
        "bank_content_hash": bank.content_hash,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    provenance_path = output_dir / "evaluation_provenance.json"
    provenance_temporary = output_dir / "evaluation_provenance.json.tmp"
    if provenance_path.exists() or provenance_temporary.exists():
        raise FileExistsError("refusing to overwrite evaluation provenance")
    provenance_temporary.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths = write_endpoint_artifacts(output_dir, bank)
    provenance_temporary.replace(provenance_path)
    print("endpoint report: {}".format(paths["report"]), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rl-checkpoint", required=True, type=Path)
    parser.add_argument("--eval-config-dir", required=True, type=Path)
    parser.add_argument("--benchmark", choices=tuple(BENCHMARK_SPLITS), default="fixed32")
    parser.add_argument("--worlds-val", type=Path)
    parser.add_argument("--worlds-test", required=True, type=Path)
    parser.add_argument("--worlds-layera", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-update", type=int, default=256)
    args = parser.parse_args(argv)
    if args.benchmark == "fixed32" and args.worlds_layera is None:
        parser.error("--benchmark fixed32 requires --worlds-layera")
    if args.benchmark == "routea-val-test" and args.worlds_val is None:
        parser.error("--benchmark routea-val-test requires --worlds-val")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
