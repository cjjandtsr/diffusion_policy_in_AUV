#!/usr/bin/env python3
"""One preregistered fresh first-bundle x downstream-policy 2x2 diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.routea_causal_diagnostics.contracts import (  # noqa: E402
    DiagnosticFailure, capture_rng_state, load_strict_json, restore_rng_state,
    write_strict_json,
)
from scripts.routea_causal_diagnostics.first_bundle_fresh_pair import (  # noqa: E402
    ARMS, INFORMATIVE, classify_task, count_classes, decompose_first_actions,
    factorial_effects, safe_arrival, summarize_action_deltas,
    summarize_factorial_effects, trajectory_metrics,
)
from scripts.routea_causal_diagnostics.reporting import (  # noqa: E402
    file_sha256, verify_sha256_manifest, write_jsonl, write_sha256_manifest,
)

ARM_SPECS = {
    "bb": ("u0", "u0"), "rr": ("seed3-u32", "seed3-u32"),
    "br": ("u0", "seed3-u32"), "rb": ("seed3-u32", "u0"),
}
FROZEN_SELECTION_SHA256 = "96fa29ed157a98558cf0752889343cce037a1f09404626136f2fc11aa39b2425"
SOURCE_FILES = (
    "scripts/run_routea_first_bundle_fresh_pair.py",
    "scripts/routea_causal_diagnostics/first_bundle_fresh_pair.py",
    "scripts/routea_causal_diagnostics/action_feedback.py",
    "scripts/evaluate_dppo_uav_formal_endpoint.py",
    "scripts/probe_routea_goal_sensitivity.py",
    "diffusion_policy_3d/env/traj_utils.py",
    "diffusion_policy_3d/env/kinematic_uav_env.py",
    "diffusion_policy_3d/env/reward_v17_support/environment.py",
    "diffusion_policy_3d/env/reward_v17.py",
    "diffusion_policy_3d/model/common/synth_pointcloud.py",
    "diffusion_policy_3d/rl/dppo_uav_rebuild/environments/observation_provenance.py",
    "diffusion_policy_3d/rl/dppo_uav_rebuild/environments/kinematic_adapter.py",
    "diffusion_policy_3d/rl/dppo_uav_rebuild/engine/contracts.py",
    "tests/dppo_uav_rebuild/audit/test_first_bundle_fresh_pair.py",
)
STATE_FIELDS = (
    "_drone_pos", "_drone_yaw", "_drone_vel", "goal_w", "_episode_start_world",
    "_last_path", "episode_step", "max_episode_steps", "_done_reason",
    "world_id", "obstacles_json", "obstacles", "drone_radius", "collision_thresh",
    "z_min", "z_max", "n_exec_steps", "rl_action_steps", "reward_version",
    "reward_v17_cfg", "offline_density", "_spawn_offset", "_episode_role",
    "exec_noise_scale", "disturb_enable", "disturb_prob", "disturb_pos_scale",
    "_pc_history", "_state_history", "_pc_meta_history", "_last_pc_meta",
    "_observation_frame_index", "_episode_rng_seed", "_episode_identity_hash",
    "_reward_history", "_prev_phi", "_prev_dist", "_count_stuck",
    "_last_reward_terms", "_last_reward_metrics", "_episode_fallback_type",
    "_episode_fallback_reasons", "_episode_reward_term_active",
    "_episode_dense_struct_traversal", "_episode_antic_risk_onset",
    "_episode_support_lowdensity_cov", "_last_rejection_histogram",
)


def _emit(**items: Any) -> None:
    print(json.dumps(items, sort_keys=True, ensure_ascii=False, allow_nan=False), flush=True)


def _typed(value: Any) -> Any:
    """Stable, type-aware JSON form; arrays retain exact dtype, shape and bytes."""
    if is_dataclass(value):
        return {"kind": "dataclass", "type": type(value).__name__, "fields": _typed(asdict(value))}
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _typed(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.kind == "O":
            raise ValueError("object array cannot be canonically hashed")
        return {"kind": "ndarray", "dtype": array.dtype.str, "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return _typed(np.asarray(value))
    if isinstance(value, Mapping):
        return {"kind": "mapping", "fields": {
            str(key): _typed(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }}
    if isinstance(value, (tuple, list)):
        return {"kind": type(value).__name__, "items": [_typed(item) for item in value]}
    if value is None or isinstance(value, (str, bool, int)):
        return {"kind": type(value).__name__, "value": value}
    if isinstance(value, float):
        return {"kind": "float", "hex": value.hex()}
    if isinstance(value, Path):
        return {"kind": "path", "value": str(value)}
    raise TypeError("unsupported state field {}".format(type(value).__name__))


def _compare_snapshots(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: {"left": left.get(key), "right": right.get(key)}
        for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)
    }


def _snapshot(adapter: Any, observation: Any, context: Any, encoded: Any, facts: Any) -> dict[str, Any]:
    env = adapter.environment
    missing = [name for name in STATE_FIELDS if not hasattr(env, name)]
    if missing:
        raise ValueError("v1.7 state schema missing fields: {}".format(missing))
    state = {name: _typed(getattr(env, name)) for name in STATE_FIELDS}
    state["environment_rng"] = _typed(env._rng.bit_generator.state)
    state["gym_rng"] = _typed(env.np_random.bit_generator.state)
    synth = env.synth_pc
    state["synth_pointcloud"] = {
        name: _typed(getattr(synth, name)) for name in (
            "cam", "noise_cfg", "obstacles_data", "_ray_dirs_cam",
            "n_cylinders", "n_boxes", "cyl_center", "cyl_radius",
            "cyl_z_min", "cyl_z_max", "box_center", "box_half_ext", "box_yaw",
        )
    }
    state["adapter_reward_state"] = _typed(adapter.reward_state.history)
    state["last_transition_facts"] = _typed(facts)
    state["observation"] = _typed(observation)
    state["observation_context"] = _typed(context)
    if encoded is not None:
        state["encoded_policy_input"] = _typed(encoded.condition)
        state["encoded_provenance"] = _typed(encoded.provenance)
    return state


def _source_manifest() -> dict[str, Any]:
    return {"schema": "routea_first_bundle_fresh_source_v1",
            "files": {name: file_sha256(REPO_ROOT / name) for name in SOURCE_FILES}}


def _selected_entries(selection: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if (selection.get("schema") != "routea_first_h4_swap_selection_v1"
            or selection.get("pair_count") != 8 or selection.get("task_count") != 16):
        raise ValueError("frozen selection contract differs")
    pairs = selection["pairs"]
    if (len(pairs) != 8 or sum(pair["role"] == "anticipatory" for pair in pairs) != 6
            or sum(pair["role"] == "dense_recovery" for pair in pairs) != 2):
        raise ValueError("frozen role/pair counts differ")
    entries = [entry for pair in pairs for entry in (pair["break"], pair["rescue"])]
    if len(entries) != 16 or len({str(e["task_id"]) for e in entries}) != 16:
        raise ValueError("expected sixteen unique preregistered tasks")
    return entries


def _seed_schedule(entries: list[Mapping[str, Any]], max_steps: int) -> dict[str, Any]:
    from scripts.probe_routea_goal_sensitivity import _policy_seed, _repeat_identity
    result = {}
    for entry in entries:
        task_id = str(entry["task_id"])
        reset_seed, offset = _repeat_identity(entry["config_hash"], int(entry["repeat"]))
        if reset_seed is not None and reset_seed != int(entry["env_reset_seed"]):
            raise ValueError("repeat reset seed mismatch: {}".format(task_id))
        result[task_id] = {
            "repeat": int(entry["repeat"]), "env_reset_seed": int(entry["env_reset_seed"]),
            "action_seeds": [_policy_seed(entry["config_hash"], decision, offset)
                             for decision in range(max_steps)],
        }
    return result


def _policy_array(bundle: Any) -> np.ndarray:
    value = bundle.policy
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.shape != (1, 8, 11) or not np.isfinite(array).all():
        raise ValueError("policy bundle must be finite (1,8,11)")
    return np.ascontiguousarray(array[0], dtype=np.float32)


def _bundle_from_policy(action: np.ndarray) -> Any:
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ActionBundle, ActionSpec
    spec = ActionSpec.e5()
    full = np.zeros((1, spec.full_horizon, spec.action_dim), dtype=np.float32)
    full[:, spec.action_start:spec.action_start + spec.policy_horizon, :] = action
    return ActionBundle.from_full(full, spec)


def _encode(system: Any, observation: Any, context: Mapping[str, Any]) -> Any:
    encoded = system.observation_adapter.encode(
        observation, context["pointcloud_meta_history"],
        episode_identity_hash=context["episode_identity_hash"],
        decision_index=context["decision_index"],
    )
    system.observation_adapter.verify(encoded)
    return encoded


def _provenance(encoded: Any) -> dict[str, Any]:
    p = encoded.provenance
    return {name: getattr(p, name, None) for name in (
        "raw_frame_hash", "raw_history_hash", "normalizer_identity_hash",
        "normalized_history_hash", "policy_input_hash", "episode_identity_hash",
    )}


def _run_condition(
    entry: Mapping[str, Any], arm: str, systems: Mapping[str, Any],
    schedule: Mapping[str, Any], donor_action: np.ndarray | None,
    on_transition: Any = None,
) -> dict[str, Any]:
    from diffusion_policy_3d.env.traj_utils import action_to_world, YAW
    from scripts.evaluate_dppo_uav_formal_endpoint import (
        _E5Policy, _build_adapter, canonical_done_reason,
        canonical_episode_config, reset_fixed_episode,
    )

    donor, follow = ARM_SPECS[arm]
    config, system = systems[follow]
    donor_system = systems[donor][1]
    fixed = canonical_episode_config(entry["episode_config"])
    if fixed.config_hash != entry["config_hash"]:
        raise ValueError("episode config hash differs: {}".format(entry["task_id"]))
    geometry = Path(entry["geometry_path"])
    if file_sha256(geometry) != entry["geometry_sha256"]:
        raise ValueError("geometry sha256 differs: {}".format(geometry))
    adapter = _build_adapter(system, config, entry["world"], geometry, entry["role"])
    policy = _E5Policy(system, use_base_policy=(follow == "u0"), identity_hash=follow)
    donor_policy = _E5Policy(donor_system, use_base_policy=(donor == "u0"), identity_hash=donor)
    result = {"arm": arm, "transitions": []}
    try:
        observation, context = reset_fixed_episode(
            adapter.environment, fixed.payload,
            seed=int(schedule["env_reset_seed"]), reset_callable=adapter.reset,
        )
        encoded_initial = _encode(donor_system, observation, context)
        result["initial_snapshot"] = _snapshot(adapter, observation, context, encoded_initial, None)
        result["initial_provenance"] = _provenance(encoded_initial)
        if donor_action is None:
            first_bundle = donor_policy.predict(
                observation, context, action_seed=int(schedule["action_seeds"][0])
            )
            if {name: getattr(donor_policy.last_provenance, name, None)
                for name in _provenance(encoded_initial)} != _provenance(encoded_initial):
                raise ValueError("decision-0 actual forward input differs from initial audit")
            first_action = _policy_array(first_bundle)
            result["first_forward_actual"] = True
        else:
            first_action = np.asarray(donor_action, dtype=np.float32)
            if first_action.shape != (8, 11) or not np.isfinite(first_action).all():
                raise ValueError("donor action must be finite (8,11)")
            first_bundle = _bundle_from_policy(first_action)
            result["first_forward_actual"] = False
        result["first_action_8x11"] = first_action.tolist()
        result["first_action_identity"] = _typed(first_action)
        decision = 0
        while True:
            before = np.asarray(adapter.environment._drone_pos, dtype=np.float32).copy()
            before_yaw = float(adapter.environment._drone_yaw)
            before_vel = np.asarray(adapter.environment._drone_vel, dtype=np.float32).copy()
            if decision == 0:
                bundle = first_bundle
                action = first_action
                provenance = _provenance(encoded_initial)
            else:
                bundle = policy.predict(
                    observation, context,
                    action_seed=int(schedule["action_seeds"][decision]),
                )
                action = _policy_array(bundle)
                provenance = {
                    name: getattr(policy.last_provenance, name, None)
                    for name in _provenance(encoded_initial)
                }
            obs_next, reward, terminated, truncated, info = adapter.step(bundle)
            facts = info.get("transition_facts")
            path_raw = info.get("executed_path")
            path = None if path_raw is None else np.asarray(path_raw, dtype=np.float32)
            if path is not None and (path.shape != (5, 3) or not np.isfinite(path).all()):
                raise ValueError("invalid executed H4 path")
            yaws = None
            if path is not None:
                wp = action_to_world(action, before, before_vel, before_yaw)
                yaws = np.asarray(wp[:4, YAW], dtype=np.float32).tolist()
            after = np.asarray(adapter.environment._drone_pos, dtype=np.float32).copy()
            reason = "none"
            if terminated or truncated:
                reason = canonical_done_reason(
                    getattr(facts, "raw_done_reason", None) or getattr(adapter.environment, "_done_reason", None),
                    float(after[2]), bool(terminated), bool(truncated),
                )
            clearance = (info.get("reward_metrics") or {}).get("d_min")
            row = {
                "task_id": entry["task_id"], "repeat": int(entry["repeat"]),
                "decision": decision, "arm": arm, "donor": donor, "follow": follow,
                "action_seed": int(schedule["action_seeds"][decision]),
                "policy_action_8x11": action.tolist(), "action_identity": _typed(action),
                "provenance": provenance, "position_before": before.tolist(),
                "position_after": after.tolist(), "executed_path": None if path is None else path.tolist(),
                "executed_yaws": yaws, "reward": float(reward),
                "physical_failure": bool(getattr(facts, "physical_failure", False)),
                "transition_facts": _typed(facts), "done_reason": reason,
                "terminated": bool(terminated), "truncated": bool(truncated),
                "clearance_m": None if clearance is None or not np.isfinite(clearance) else float(clearance),
            }
            result["transitions"].append(row)
            if on_transition is not None:
                on_transition(row)
            if decision == 0:
                encoded_next = None if terminated or truncated or obs_next is None else _encode(system, obs_next, info)
                result["decision1_snapshot"] = _snapshot(adapter, obs_next, info, encoded_next, facts)
                result["decision1_not_applicable"] = bool(terminated or truncated)
                result["first_executed_path"] = None if path is None else path.tolist()
            decision += 1
            if terminated or truncated or obs_next is None:
                result["done_reason"] = reason
                break
            if decision >= len(schedule["action_seeds"]):
                raise ValueError("episode exceeded frozen seed schedule")
            observation, context = obs_next, info
        result["decision_count"] = decision
        return result
    finally:
        adapter.environment.close()


def _assess_task(entry: Mapping[str, Any], arms: Mapping[str, Any]) -> dict[str, Any]:
    mismatches: dict[str, Any] = {}
    reference = arms["bb"]["initial_snapshot"]
    for arm in ("rr", "br", "rb"):
        diff = _compare_snapshots(reference, arms[arm]["initial_snapshot"])
        if diff:
            mismatches["initial_" + arm] = diff
    for same_donor, crossed in (("bb", "br"), ("rr", "rb")):
        if arms[same_donor]["first_action_identity"] != arms[crossed]["first_action_identity"]:
            mismatches[same_donor + "_action"] = "first 8x11 bundle differs"
        if arms[same_donor]["first_executed_path"] != arms[crossed]["first_executed_path"]:
            mismatches[same_donor + "_executed_path"] = "first 5x3 path differs"
        if arms[same_donor]["decision1_not_applicable"] != arms[crossed]["decision1_not_applicable"]:
            mismatches[same_donor + "_terminal_parity"] = "first step termination differs"
        diff = _compare_snapshots(
            arms[same_donor]["decision1_snapshot"], arms[crossed]["decision1_snapshot"]
        )
        if diff:
            mismatches[same_donor + "_decision1_state"] = diff
    endpoints = {
        arm: safe_arrival(arms[arm]["transitions"], entry["episode_config"]["goal_pos"])
        for arm in ARMS
    }
    outcomes = {arm: endpoints[arm]["success"] for arm in ARMS}
    label = classify_task(outcomes, pair_valid=not bool(mismatches))
    action_difference = decompose_first_actions(
        arms["bb"]["first_action_8x11"], arms["rr"]["first_action_8x11"],
        entry["episode_config"]["start_pos"], entry["episode_config"]["goal_pos"],
    )
    metrics = {}
    for arm in ARMS:
        try:
            metrics[arm] = trajectory_metrics(
                arms[arm]["transitions"], entry["episode_config"]["goal_pos"],
                float(entry["episode_config"]["start_yaw"]),
            )
        except (ValueError, TypeError, KeyError) as exc:
            metrics[arm] = {"unavailable_reason": str(exc)}
    metric_names = set.intersection(*(set(value) for value in metrics.values()))
    effects = {}
    for name in sorted(metric_names):
        values = {arm: metrics[arm][name] for arm in ARMS}
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
               for v in values.values()):
            effects[name] = factorial_effects(values)
    if all(value is not None for value in outcomes.values()) and not mismatches:
        effects["safe_arrival_binary"] = factorial_effects(
            {arm: float(outcomes[arm]) for arm in ARMS}
        )
    if all(value is True for value in outcomes.values()) and not mismatches:
        effects["decision_to_safe_arrival_substeps"] = factorial_effects({
            arm: 4. * endpoints[arm]["entry"]["decision"]
                 + endpoints[arm]["entry"]["segment"]
                 + endpoints[arm]["entry"]["fraction"]
            for arm in ARMS
        })
    if label == "INVALID_PAIR":
        fresh = "invalid"
    elif outcomes["bb"] is None or outcomes["rr"] is None:
        fresh = "ambiguous"
    elif outcomes["bb"] and not outcomes["rr"]:
        fresh = "fresh_break"
    elif outcomes["rr"] and not outcomes["bb"]:
        fresh = "fresh_rescue"
    else:
        fresh = "uninformative"
    return {
        "task_id": entry["task_id"], "repeat": int(entry["repeat"]),
        "role": entry["role"], "historical_selection_label": entry["outcome_class"],
        "fresh_control_label": fresh, "pattern": label,
        "pair_mismatches": mismatches, "endpoints": endpoints,
        "action_decomposition": action_difference,
        "continuous_metrics": metrics, "factorial_effects": effects,
        "conditions": {arm: {
            "first_forward_actual": arms[arm]["first_forward_actual"],
            "decision_count": arms[arm]["decision_count"],
            "done_reason": arms[arm]["done_reason"],
            "first_action_identity": arms[arm]["first_action_identity"],
            "initial_provenance": arms[arm]["initial_provenance"],
            "initial_snapshot": arms[arm]["initial_snapshot"],
            "decision1_snapshot": arms[arm]["decision1_snapshot"],
            "decision1_not_applicable": arms[arm]["decision1_not_applicable"],
        } for arm in ARMS},
    }


def _wilson(k: int, n: int) -> dict[str, float] | None:
    if n == 0:
        return None
    z = 1.96
    p = k / n
    denominator = 1. + z * z / n
    center = (p + z * z / (2. * n)) / denominator
    half = z * np.sqrt(p * (1. - p) / n + z * z / (4. * n * n)) / denominator
    return {"lower": float(center - half), "upper": float(center + half)}


def _summarize(tasks: list[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [task["pattern"] for task in tasks]
    counts = count_classes(labels)
    n = counts["informative_task_count"]
    routed = []
    for name in INFORMATIVE:
        supporters = [task for task in tasks if task["pattern"] == name]
        fresh_break = sum(task["fresh_control_label"] == "fresh_break" for task in supporters)
        fresh_rescue = sum(task["fresh_control_label"] == "fresh_rescue" for task in supporters)
        if n >= 8 and len(supporters) >= int(np.ceil(.75 * n)) and fresh_break >= 2 and fresh_rescue >= 2:
            routed.append(name)
    return {
        "schema": "routea_first_bundle_fresh_summary_v1",
        "evidence_scope": "one_repeat_enriched_lightweight_kinematic_diagnostic",
        "training_success_claim_allowed": False,
        "task_count": len(tasks), "planned_task_count": 16, "planned_condition_count": 64,
        "counts": counts,
        "wilson_95_by_pattern": {name: _wilson(counts[name], n) for name in INFORMATIVE},
        "priority_pattern": routed[0] if len(routed) == 1 else "HETEROGENEOUS_OR_INCONCLUSIVE",
        "priority_is_heuristic": True,
        "diagnosis_closed_after_this_run": True,
        "fresh_control_counts": {
            label: sum(task.get("fresh_control_label") == label for task in tasks)
            for label in ("fresh_break", "fresh_rescue", "uninformative", "ambiguous", "invalid")
        },
        "pattern_by_role_and_fresh_label": {
            role: {
                fresh: {pattern: sum(
                    task.get("role") == role and task.get("fresh_control_label") == fresh
                    and task["pattern"] == pattern for task in tasks
                ) for pattern in INFORMATIVE}
                for fresh in ("fresh_break", "fresh_rescue", "uninformative")
            } for role in ("anticipatory", "dense_recovery")
        },
    }


def _run(args: argparse.Namespace) -> int:
    import torch
    from omegaconf import OmegaConf
    from scripts.routea_causal_diagnostics.runtime import (
        install_read_only_tripwires, load_read_only_system, validate_zero_effect_lora_state,
    )

    root = Path(args.output_dir).resolve()
    if root.exists():
        raise DiagnosticFailure("OUTPUT_DIR_EXISTS", str(root), exit_code=2)
    root.mkdir(parents=True)
    error: Exception | None = None
    try:
        selection_dir = Path(args.selection_dir)
        verify_sha256_manifest(selection_dir)
        if file_sha256(selection_dir / "selection.json") != FROZEN_SELECTION_SHA256:
            raise ValueError("frozen selection SHA256 differs")
        selection = load_strict_json(selection_dir / "selection.json")
        entries = _selected_entries(selection)
        write_strict_json(root / "selection_identity.json", {
            "selection_sha256": file_sha256(selection_dir / "selection.json"),
            "selection_manifest_sha256": file_sha256(selection_dir / "manifest_sha256.txt"),
            "task_ids": [entry["task_id"] for entry in entries],
            "historical_selection_only": True,
        })
        write_strict_json(root / "diagnostic_source_manifest.json", _source_manifest())
        bc_config, bc_system = load_read_only_system(args.u0_config, args.u0_checkpoint, REPO_ROOT)
        rl_config, rl_system = load_read_only_system(args.rl_config, args.rl_checkpoint, REPO_ROOT)
        validate_zero_effect_lora_state(bc_system.lora_bundle.adapter_state_dict())
        if (OmegaConf.to_container(bc_config.environment, resolve=True)
                != OmegaConf.to_container(rl_config.environment, resolve=True)
                or OmegaConf.to_container(bc_config.reward, resolve=True)
                != OmegaConf.to_container(rl_config.reward, resolve=True)):
            raise ValueError("BC/RL environment or reward configuration differs")
        max_steps = int(bc_config.environment.max_episode_steps)
        schedule = _seed_schedule(entries, max_steps)
        write_strict_json(root / "seed_schedule.json", schedule)
        systems = {"u0": (bc_config, bc_system), "seed3-u32": (rl_config, rl_system)}
        immutable = [selection_dir / "selection.json", selection_dir / "manifest_sha256.txt",
                     Path(args.u0_config), Path(args.rl_config),
                     Path(args.u0_checkpoint) / "header.json", Path(args.u0_checkpoint) / "SHA256.json",
                     Path(args.rl_checkpoint) / "header.json", Path(args.rl_checkpoint) / "SHA256.json"]
        immutable += [Path(entry["geometry_path"]) for entry in entries]
        outer_rng = capture_rng_state()
        guards = {
            lane: install_read_only_tripwires(system, immutable_files=immutable)
            for lane, (_, system) in systems.items()
        }
        write_strict_json(root / "model_identity.json", {
            lane: {
                "config_sha256": file_sha256(config_path),
                "checkpoint_header_sha256": file_sha256(Path(checkpoint_path) / "header.json"),
                "checkpoint_integrity_sha256": file_sha256(Path(checkpoint_path) / "SHA256.json"),
                "use_base_policy": lane == "u0",
            }
            for lane, config_path, checkpoint_path in (
                ("u0", args.u0_config, args.u0_checkpoint),
                ("seed3-u32", args.rl_config, args.rl_checkpoint),
            )
        })
        tasks = []
        optimize_state = getattr(torch._C, "_get_graph_executor_optimize", None)
        jit_before = None if optimize_state is None else bool(optimize_state())
        with (root / "arm_transitions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            def emit_transition(row: Mapping[str, Any]) -> None:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()

            try:
                with torch.jit.optimized_execution(False):
                    jit_during = None if optimize_state is None else bool(optimize_state())
                    for entry in entries:
                        arms: dict[str, Any] = {}
                        try:
                            task_schedule = schedule[entry["task_id"]]
                            for arm in ("bb", "rr", "br", "rb"):
                                donor = None
                                if arm == "br":
                                    donor = np.asarray(arms["bb"]["first_action_8x11"], dtype=np.float32)
                                elif arm == "rb":
                                    donor = np.asarray(arms["rr"]["first_action_8x11"], dtype=np.float32)
                                arms[arm] = _run_condition(
                                    entry, arm, systems, task_schedule, donor, emit_transition,
                                )
                            task = _assess_task(entry, arms)
                        except Exception as exc:
                            task = {
                                "task_id": entry["task_id"], "repeat": int(entry["repeat"]),
                                "role": entry["role"],
                                "historical_selection_label": entry["outcome_class"],
                                "fresh_control_label": "invalid", "pattern": "INVALID_PAIR",
                                "error": {"type": type(exc).__name__, "detail": str(exc)},
                                "completed_conditions": list(arms),
                            }
                        tasks.append(task)
                        _emit(status="TASK_COMPLETE", task_id=entry["task_id"], pattern=task["pattern"])
            finally:
                restore_rng_state(outer_rng)
        jit_after = None if optimize_state is None else bool(optimize_state())
        write_strict_json(root / "jit_condition.json", {
            "before": jit_before, "during": jit_during, "after": jit_after,
            "torch_jit_optimized_execution_disabled_for_conditions": True,
        })
        tripwire = {
            "status": "READ_ONLY_VERIFIED",
            "lanes": {lane: guard.verify(require_rng_unchanged=True) for lane, guard in guards.items()},
        }
        write_strict_json(root / "tripwire_verification.json", tripwire)
        write_jsonl(root / "task_results.jsonl", tasks)
        write_strict_json(root / "action_delta_strata.json", summarize_action_deltas(tasks))
        write_strict_json(root / "factorial_effect_task_summary.json", summarize_factorial_effects(tasks))
        summary = _summarize(tasks)
        write_strict_json(root / "summary.json", summary)
        _emit(status="RUN_COMPLETE", tasks=16, counts=summary["counts"], priority=summary["priority_pattern"])
    except Exception as exc:
        error = exc
        write_strict_json(root / "run_error.json", {
            "status": "RUN_FAILED", "type": type(exc).__name__, "detail": str(exc),
        })
    finally:
        write_sha256_manifest(root)
    if error is not None:
        if isinstance(error, DiagnosticFailure):
            raise error
        raise DiagnosticFailure("FRESH_PAIR_RUN_FAILED", str(error), exit_code=2) from error
    return 0


def _validate(args: argparse.Namespace) -> int:
    root = Path(args.output_dir)
    files = verify_sha256_manifest(root)
    required = {
        "selection_identity.json", "seed_schedule.json", "model_identity.json",
        "jit_condition.json", "arm_transitions.jsonl", "task_results.jsonl",
        "summary.json", "tripwire_verification.json", "diagnostic_source_manifest.json",
        "action_delta_strata.json", "factorial_effect_task_summary.json",
    }
    if not required <= set(files):
        raise DiagnosticFailure("OUTPUT_CONTRACT_MISMATCH", "required output missing", exit_code=2)
    if load_strict_json(root / "diagnostic_source_manifest.json") != _source_manifest():
        raise DiagnosticFailure("OUTPUT_CONTRACT_MISMATCH", "source files changed", exit_code=2)
    summary = load_strict_json(root / "summary.json")
    if summary["task_count"] != 16 or summary["counts"]["total"] != 16:
        raise DiagnosticFailure("OUTPUT_CONTRACT_MISMATCH", "task denominator differs", exit_code=2)
    if summary["counts"]["pair_valid"] + summary["counts"]["invalid"] != 16:
        raise DiagnosticFailure("OUTPUT_CONTRACT_MISMATCH", "pair denominator differs", exit_code=2)
    tasks = []
    with (root / "task_results.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            tasks.append(json.loads(line))
    if len(tasks) != 16 or len({task["task_id"] for task in tasks}) != 16:
        raise DiagnosticFailure("OUTPUT_CONTRACT_MISMATCH", "task identities differ", exit_code=2)
    if count_classes([task["pattern"] for task in tasks]) != summary["counts"]:
        raise DiagnosticFailure("OUTPUT_CONTRACT_MISMATCH", "classification denominators differ", exit_code=2)
    if summarize_action_deltas(tasks) != load_strict_json(root / "action_delta_strata.json"):
        raise DiagnosticFailure("OUTPUT_CONTRACT_MISMATCH", "action strata differ", exit_code=2)
    if summarize_factorial_effects(tasks) != load_strict_json(root / "factorial_effect_task_summary.json"):
        raise DiagnosticFailure("OUTPUT_CONTRACT_MISMATCH", "factorial summary differs", exit_code=2)
    schedules = load_strict_json(root / "seed_schedule.json")
    rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with (root / "arm_transitions.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["task_id"], row["arm"])
            rows.setdefault(key, []).append(row)
    for task in tasks:
        if task["pattern"] == "INVALID_PAIR" and "conditions" not in task:
            continue
        conditions = task["conditions"]
        if set(conditions) != set(ARMS):
            raise DiagnosticFailure("OUTPUT_CONTRACT_MISMATCH", "four conditions missing", exit_code=2)
        for arm in ARMS:
            seq = rows.get((task["task_id"], arm), [])
            if (len(seq) != conditions[arm]["decision_count"]
                    or [row["decision"] for row in seq] != list(range(len(seq)))
                    or any(row["action_seed"] != schedules[task["task_id"]]["action_seeds"][row["decision"]]
                           for row in seq)):
                raise DiagnosticFailure("OUTPUT_CONTRACT_MISMATCH", "decision seed/row sequence differs", exit_code=2)
        if not conditions["bb"]["first_forward_actual"] or not conditions["rr"]["first_forward_actual"]:
            raise DiagnosticFailure("OUTPUT_CONTRACT_MISMATCH", "fresh controls did not forward", exit_code=2)
        if conditions["br"]["first_forward_actual"] or conditions["rb"]["first_forward_actual"]:
            raise DiagnosticFailure("OUTPUT_CONTRACT_MISMATCH", "cross condition unexpectedly forwarded at decision 0", exit_code=2)
        for donor, crossed in (("bb", "br"), ("rr", "rb")):
            if conditions[donor]["first_action_identity"] != conditions[crossed]["first_action_identity"]:
                raise DiagnosticFailure("OUTPUT_CONTRACT_MISMATCH", "donor first bundle differs", exit_code=2)
        if task["pattern"] != "INVALID_PAIR":
            if task["pair_mismatches"]:
                raise DiagnosticFailure("OUTPUT_CONTRACT_MISMATCH", "valid task has pair mismatch", exit_code=2)
            if classify_task({arm: task["endpoints"][arm]["success"] for arm in ARMS}) != task["pattern"]:
                raise DiagnosticFailure("OUTPUT_CONTRACT_MISMATCH", "outcome pattern differs", exit_code=2)
    jit = load_strict_json(root / "jit_condition.json")
    if jit["during"] is not False or (jit["before"] is not None and jit["before"] != jit["after"]):
        raise DiagnosticFailure("OUTPUT_CONTRACT_MISMATCH", "JIT context isolation differs", exit_code=2)
    if load_strict_json(root / "tripwire_verification.json")["status"] != "READ_ONLY_VERIFIED":
        raise DiagnosticFailure("OUTPUT_CONTRACT_MISMATCH", "read-only tripwire failed", exit_code=2)
    _emit(status="VALIDATE_COMPLETE", tasks=16, verified_files=len(files), counts=summary["counts"])
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--selection-dir", type=Path, required=True)
    run.add_argument("--u0-config", type=Path, required=True)
    run.add_argument("--u0-checkpoint", type=Path, required=True)
    run.add_argument("--rl-config", type=Path, required=True)
    run.add_argument("--rl-checkpoint", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    validate = sub.add_parser("validate-output")
    validate.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        return {"run": _run, "validate-output": _validate}[args.command](args)
    except DiagnosticFailure as exc:
        _emit(status=exc.status, detail=exc.detail, exit_code=exc.exit_code)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
