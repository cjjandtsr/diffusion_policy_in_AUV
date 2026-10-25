#!/usr/bin/env python3
"""Read-only Reward v1.7 DDPO credit-flow probe."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import warnings

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dppo_credit_support.credit_bank import load_dev_bank
from scripts.dppo_credit_support.credit_probe import (
    compute_policy_credit,
    make_batch_schedule,
    probe_minibatch,
    replay_sampled_k_schedules,
    reverse_kernel_kl_record,
)


def _self_check():
    class TinyRuntime:
        norm_adv = False
        gamma_denoising = 0.8
        ft_denoising_steps = 10
        clip_ploss_coef = 0.1
        clip_ploss_coef_base = 0.001
        clip_ploss_coef_rate = 3.0
        clip_advantage_lower_quantile = 0.0
        clip_advantage_upper_quantile = 1.0

        def __init__(self, weight):
            self.weight = weight

        def get_logprobs_subsample(
            self, condition, chains_prev, chains_next, denoising_inds
        ):
            coefficients = condition["state"][:, :1].reshape(-1, 1, 1)
            return self.weight * coefficients.expand(-1, 16, 11)

    weight = torch.nn.Parameter(torch.tensor(0.25))
    before = weight.detach().clone()
    batch = argparse.Namespace(
        condition={"state": torch.tensor([[1.0], [-1.0]])},
        chains_prev=torch.zeros(2, 16, 11),
        chains_next=torch.zeros(2, 16, 11),
        denoising_inds=torch.tensor([9, 9]),
        advantages=torch.tensor([1.0, 2.0]),
        oldlogprobs=torch.zeros(2, 16, 11),
    )
    result = probe_minibatch(
        TinyRuntime(weight),
        batch,
        actor_parameters=(weight,),
        reward_start=2,
        reward_horizon=4,
        outcome_labels=("reached", "timeout"),
        role_labels=("anticipatory", "dense_recovery"),
    )
    return {
        "status": "SELF_CHECK_PASS",
        "component_count": result["credit"]["component_count"],
        "parameter_delta_max_abs": float((weight.detach() - before).abs().max()),
        "gradient_reconstruction_max_abs": max(
            result["gradient_by_outcome"]["gradient_reconstruction_max_abs"],
            result["gradient_by_role"]["gradient_reconstruction_max_abs"],
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_metrics(info):
    result = {}
    for name, value in info.get("reward_metrics", {}).items():
        if isinstance(value, (bool, np.bool_)):
            result[str(name)] = bool(value)
        elif isinstance(value, (int, float, np.number)) and math.isfinite(float(value)):
            result[str(name)] = float(value)
    return result


def _no_ray_warning_count(captured) -> int:
    return sum("synth: no ray hit" in str(item.message) for item in captured)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    return copy.deepcopy(value)


def _tree_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left.cpu(), right.cpu())
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    return left == right


def _d1b_snapshot(system, buffer):
    return {
        "actor": _clone_tree(system.lora_bundle.adapter_state_dict()),
        "actor_grad": _clone_tree(tuple(parameter.grad for parameter in system.trainer.actor_parameters)),
        "critic": _clone_tree(system.critic.state_dict()),
        "critic_grad": _clone_tree(tuple(parameter.grad for parameter in system.trainer.critic_parameters)),
        "optimizers": _clone_tree((system.actor_optimizer.state_dict(), system.critic_optimizer.state_dict())),
        "counters": _clone_tree(asdict(system.trainer.counters)),
        "buffer": _clone_tree(buffer.state_dict()),
        "python_rng": random.getstate(),
        "numpy_rng": _clone_tree(np.random.get_state()),
        "torch_rng": torch.get_rng_state().clone(),
        "cuda_rng": tuple(value.cpu().clone() for value in torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else (),
        "rollout_sampler": _clone_tree(system.rollout_sampler.state_dict()) if hasattr(system.rollout_sampler, "state_dict") else None,
    }


def _d1b_tripwire(before, system, buffer):
    after = _d1b_snapshot(system, buffer)
    categories = {name: _tree_equal(before[name], after[name]) for name in before}
    categories["all_unchanged"] = all(categories.values())
    return categories


def _warning_summary(rows):
    return {
        "reset_warning_total": sum(
            row["reset_no_ray_warning_count_at_episode_start"] for row in rows
        ),
        "step_warning_total": sum(row["step_no_ray_warning_count"] for row in rows),
        "episode_starts_with_reset_warning": sum(
            row["decision_index"] == 0
            and row["reset_no_ray_warning_count_at_episode_start"] > 0
            for row in rows
        ),
        "transitions_with_step_warning": sum(
            row["step_no_ray_warning_count"] > 0 for row in rows
        ),
    }


def _summary(rows, key):
    groups = defaultdict(list)
    for row in rows:
        label = (
            "|".join(str(row[name]) for name in key)
            if isinstance(key, tuple)
            else str(row[key])
        )
        groups[label].append(row)
    result = {}
    for label, values in sorted(groups.items()):
        count = len(values)
        components = sum(int(row["component_count"]) for row in values)
        result[label] = {
            "transitions": count,
            "reward_mean": sum(row["reward"] for row in values) / count,
            "raw_advantage_mean": sum(row["raw_advantage"] for row in values) / count,
            "normalized_advantage_mean": sum(
                row["normalized_advantage"] for row in values
            )
            / count,
            "policy_loss_mean": sum(row["policy_loss"] for row in values) / count,
            "goal_distance_3d_mean": sum(
                row["goal_distance_3d_after"] for row in values
            )
            / count,
            "geometric_approx_kl_mean": sum(
                row["geometric_approx_kl"] for row in values
            )
            / count,
            "clamped_joint_approx_kl_mean": sum(
                row["clamped_joint_approx_kl"] for row in values
            )
            / count,
            "geometric_clip_fraction": sum(
                row["geometric_clip_hit"] for row in values
            )
            / count,
            "clamped_joint_clip_fraction": sum(
                row["clamped_joint_clip_hit"] for row in values
            )
            / count,
            "new_below_clamp_rate": sum(
                row["new_below_clamp_count"] for row in values
            )
            / components,
            "old_below_clamp_rate": sum(
                row["old_below_clamp_count"] for row in values
            )
            / components,
            "new_above_clamp_rate": sum(
                row["new_above_clamp_count"] for row in values
            )
            / components,
            "old_above_clamp_rate": sum(
                row["old_above_clamp_count"] for row in values
            )
            / components,
        }
    return result


def _write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_npy(path: Path, array) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        np.save(stream, np.asarray(array, dtype=np.float32), allow_pickle=False)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)


def _run_live(args):
    # Fail before importing the executable training package.  This prevents a
    # BC/RL comparison from silently mixing the checkpoint with a later source
    # tree, including read-only audit modules added under the hashed package.
    from scripts.audit_checkpoint_source_identity import build_inventory

    _, actual_source_set_hash = build_inventory(REPO_ROOT)
    if actual_source_set_hash != args.expected_source_set_hash:
        raise RuntimeError(
            "source-set mismatch before model import: expected {}, got {}".format(
                args.expected_source_set_hash, actual_source_set_hash
            )
        )

    # Heavy project imports are intentionally delayed so --self-check and
    # fail-closed CLI validation work in a CPU-only minimal environment.
    from omegaconf import OmegaConf

    from diffusion_policy_3d.rl.dppo_uav_rebuild.audit.training_audit import (
        ParameterSnapshot,
    )
    from diffusion_policy_3d.rl.dppo_uav_rebuild.deployment.loader import (
        load_deployment_policy,
    )
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.build_system import (
        build_system,
    )
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.rollout_buffer import (
        OnPolicyRolloutBuffer,
    )
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.factory import (
        RouteAEnvironmentFactory,
    )
    from diffusion_policy_3d.env.reward_v17_support.config import resolve_config as resolve_reward_v17_config
    repo_root = REPO_ROOT
    config_path = args.config.resolve()
    checkpoint_path = (
        args.rl_checkpoint.resolve() if args.rl_checkpoint is not None else None
    )
    bank_path = args.dev_bank.resolve()
    output_dir = args.output_dir.resolve()
    config = OmegaConf.load(config_path)
    # Calibration deliberately measures the observation distribution before a
    # data-derived threshold exists.  build_system consumes the threshold, so
    # install the physical upper bound before constructing any runtime object.
    # The original config file remains immutable and its SHA-256 remains the
    # provenance identity recorded by this probe.
    if args.pointcloud_calibration_mode:
        configured_threshold = config.environment.max_no_ray_hit_fraction
        if configured_threshold is None:
            config.environment.max_no_ray_hit_fraction = 1.0
        config.environment.reject_all_no_ray_hit = False
    seed = int(args.probe_seed)
    # Reproduce the formal fresh-build initialization before selecting BC or
    # restoring RL. restore_checkpoint may restore historical RNG state, so
    # every dev-bank episode is independently reseeded below.
    _seed_all(int(config.run.seed))
    if args.policy_label == "rl":
        deployment = load_deployment_policy(config, checkpoint_path, repo_root)
        checkpoint_update = int(deployment.checkpoint_header.update_index)
        if checkpoint_update != int(args.expected_update):
            raise RuntimeError(
                "expected checkpoint update {}, got {}".format(
                    args.expected_update, checkpoint_update
                )
            )
        system = deployment.system
        initialization = "restored_rl_checkpoint"
    else:
        system = build_system(config, repo_root)
        checkpoint_update = 0
        initialization = "fresh_bc_lora_and_fresh_critic"
    if args.pointcloud_calibration_mode:
        system.observation_adapter.max_no_ray_hit_fraction = 1.0
        system.observation_adapter.reject_all_no_ray_hit = False
    runtime = system.official_runtime
    device = torch.device(system.bc_runtime.device)
    if (
        float(runtime.clip_advantage_lower_quantile),
        float(runtime.clip_advantage_upper_quantile),
    ) != (0.0, 1.0):
        raise RuntimeError("live probe requires the verified 0/100 advantage contract")

    def resolve(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (repo_root / path).resolve()

    factory = RouteAEnvironmentFactory(
        manifest_path=resolve(config.admission_manifest),
        worlds_train_dir=resolve(config.worlds_train),
        reward=system.reward,
        device=str(system.bc_runtime.device),
        base_seed=int(config.run.seed),
        max_episode_steps=int(config.environment.max_episode_steps),
        disturb_enable=bool(config.environment.disturb_enable),
        disturb_prob=float(config.environment.disturb_prob),
        disturb_pos_scale=float(config.environment.disturb_pos_scale),
        exec_noise_scale=float(config.environment.exec_noise_scale),
        mid_state_init_ratio=float(config.environment.mid_state_init_ratio),
        mid_state_pos_scale=float(config.environment.mid_state_pos_scale),
        max_reset_attempts=int(config.environment.max_reset_attempts),
        reward_config=OmegaConf.to_container(config.reward, resolve=True),
    )
    resolved_reward_config = resolve_reward_v17_config(
        OmegaConf.to_container(config.reward, resolve=True)
    )
    manifest_path = resolve(config.admission_manifest)
    expected_bank_tasks = []
    for task in factory.tasks:
        obstacle_path = Path(task.obstacle_json).resolve()
        expected_bank_tasks.append(
            {
                "task_id": task.task_id,
                "world_id": task.world_id,
                "role": task.role,
                "task_seed": task.seed,
                "obstacle_sha256": _sha256(obstacle_path),
            }
        )
    bank = load_dev_bank(
        bank_path,
        expected_manifest_sha256=_sha256(manifest_path),
        expected_tasks=expected_bank_tasks,
    )
    task_by_id = {task.task_id: task for task in factory.tasks}
    actor_parameters = tuple(system.trainer.actor_parameters)
    parameter_snapshot = ParameterSnapshot.capture(actor=actor_parameters)
    gradient_snapshot = tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in actor_parameters
    )
    counter_snapshot = asdict(system.trainer.counters)

    forbidden_calls = []

    def forbid_training_call(*unused_args, **unused_kwargs):
        forbidden_calls.append("training_call")
        raise RuntimeError("credit probe forbids optimizer/trainer update calls")

    system.actor_optimizer.step = forbid_training_call
    system.critic_optimizer.step = forbid_training_call
    system.trainer.update = forbid_training_call

    output_dir.mkdir(parents=True, exist_ok=False)
    transition_rows = []
    gradient_rows = []
    kl_diversity_rows = []
    d1b_report = None
    bank_entry_cursor = 0
    episode_serial = 0
    for block_index in range(args.blocks):
        buffer = OnPolicyRolloutBuffer(
            args.block_transitions,
            gamma=float(config.dppo.gamma_environment),
            gae_lambda=float(config.dppo.gae_lambda),
        )
        block_rows = []
        adapter = None
        observation = None
        observation_context = None
        task = None
        bank_entry = None
        active_indices = []
        decision_index = 0
        reset_seed = None
        start_position = None
        goal_position = None
        initial_yaw = None
        reset_no_ray_warning_count = None
        observation_rng_seed = None
        while len(buffer) < args.block_transitions:
            if adapter is None:
                if bank_entry_cursor >= len(bank["entries"]):
                    raise RuntimeError(
                        "dev bank exhausted after {} entries; build a larger bank".format(
                            bank_entry_cursor
                        )
                    )
                bank_entry = bank["entries"][bank_entry_cursor]
                bank_entry_cursor += 1
                task = task_by_id[bank_entry["task_id"]]
                adapter = factory.make(task)
                reset_seed = int(bank_entry["reset_seed"])
                with warnings.catch_warnings(record=True) as captured_reset:
                    warnings.simplefilter("always")
                    observation, observation_context = adapter.reset(seed=reset_seed)
                reset_no_ray_warning_count = _no_ray_warning_count(captured_reset)
                # Common random numbers across the separate BC/RL invocations.
                # Reset is completed first so observation generation cannot
                # consume or offset the stochastic policy stream.
                _seed_all(int(bank_entry["policy_seed"]))
                episode_serial += 1
                decision_index = 0
                active_indices = []
                environment = adapter.environment
                start_position = np.asarray(environment._drone_pos).tolist()
                goal_position = np.asarray(environment.goal_w).tolist()
                initial_yaw = float(environment._drone_yaw)
                observation_rng_seed = int(environment._episode_rng_seed)

            encoded_observation = system.observation_adapter.encode(
                observation,
                observation_context["pointcloud_meta_history"],
                episode_identity_hash=observation_context["episode_identity_hash"],
                decision_index=observation_context["decision_index"],
            )
            system.observation_adapter.verify(encoded_observation)
            condition = encoded_observation.condition
            with torch.no_grad():
                value = float(system.critic(condition).reshape(-1)[0].item())
                rollout = system.rollout_sampler.sample(
                    condition, deterministic=False
                )
            with warnings.catch_warnings(record=True) as captured_step:
                warnings.simplefilter("always")
                next_observation, reward, terminated, truncated, info = adapter.step(
                    rollout.action
                )
            step_no_ray_warning_count = _no_ray_warning_count(captured_step)
            if info.get("reward_metrics", {}).get("training_usable") is False:
                raise FloatingPointError(
                    "invalid Reward v1.7 transition in credit probe"
                )
            collection_boundary = len(buffer) + 1 == args.block_transitions
            effective_truncated = bool(truncated or (collection_boundary and not terminated))
            next_encoded_observation = system.observation_adapter.encode(
                next_observation,
                info["pointcloud_meta_history"],
                episode_identity_hash=info["episode_identity_hash"],
                decision_index=info["decision_index"],
            )
            system.observation_adapter.verify(next_encoded_observation)
            next_condition = next_encoded_observation.condition
            with torch.no_grad():
                next_value = (
                    0.0
                    if terminated
                    else float(system.critic(next_condition).reshape(-1)[0].item())
                )
            buffer.add(
                condition={name: tensor[0] for name, tensor in condition.items()},
                chain=rollout.normalized_chain[0],
                old_logprobs=rollout.old_logprobs[0],
                reward=reward,
                value=value,
                next_value=next_value,
                terminated=terminated,
                truncated=effective_truncated,
                transition_id="d1b:b{}:t{}".format(block_index, len(buffer)),
                role=task.role,
                outcome=str(info.get("done_reason", "none")),
                provenance_hash=encoded_observation.provenance.policy_input_hash,
            )
            environment = adapter.environment
            reward_metrics = _json_metrics(info)
            goal_delta = np.asarray(environment._drone_pos) - np.asarray(
                environment.goal_w
            )
            goal_distance_3d = float(np.linalg.norm(goal_delta))
            goal_distance_xy = float(np.linalg.norm(goal_delta[:2]))
            goal_abs_z_error = float(abs(goal_delta[2]))
            risk_threshold = float(resolved_reward_config["risk_active_threshold"])
            risk_active = float(reward_metrics.get("risk_gate", 0.0)) >= risk_threshold
            dense_opportunity = bool(
                task.role == "dense_recovery"
                and (
                    risk_active
                    or bool(reward_metrics.get("trap_active", False))
                    or bool(reward_metrics.get("counterfactual_collision", False))
                )
            )
            near_goal = bool(
                goal_distance_xy <= 2.0 * float(environment.goal_tol_xy)
                and goal_abs_z_error <= 2.0 * float(environment.goal_tol_z)
            )
            support_opportunity = bool(
                reward_metrics.get("support_eligible", False)
                or reward_metrics.get("above_demo_height", False)
                or reward_metrics.get("goal_above_demo_height", False)
            )
            row = {
                "schema": "dppo_credit_transition_v2",
                "block_index": block_index,
                "transition_in_block": len(buffer) - 1,
                "episode_serial": episode_serial,
                "bank_entry_index": int(bank_entry["entry_index"]),
                "bank_id": bank["bank_id"],
                "policy_label": args.policy_label,
                "policy_seed": int(bank_entry["policy_seed"]),
                "decision_index": decision_index,
                "task_id": task.task_id,
                "world_id": task.world_id,
                "role": task.role,
                "reset_seed": int(reset_seed),
                "start_position": start_position,
                "goal_position": goal_position,
                "initial_yaw": initial_yaw,
                "observation_rng_seed": observation_rng_seed,
                "observation_provenance_hash": encoded_observation.provenance.policy_input_hash,
                "pointcloud_no_ray_hit_fraction": float(
                    observation_context["pointcloud_meta_history"][-1].no_ray_hit_fraction
                ),
                "pointcloud_valid_hit_rays": int(
                    observation_context["pointcloud_meta_history"][-1].valid_hit_rays
                ),
                "pointcloud_total_rays": int(
                    observation_context["pointcloud_meta_history"][-1].total_rays
                ),
                "pointcloud_used_far_plane_fallback": bool(
                    observation_context["pointcloud_meta_history"][-1].used_far_plane_fallback
                ),
                "reset_no_ray_warning_count_at_episode_start": (
                    reset_no_ray_warning_count if decision_index == 0 else 0
                ),
                "step_no_ray_warning_count": step_no_ray_warning_count,
                "position_after": np.asarray(environment._drone_pos).tolist(),
                "yaw_after": float(environment._drone_yaw),
                "goal_distance_3d_after": goal_distance_3d,
                "goal_distance_xy_after": goal_distance_xy,
                "goal_abs_z_error_after": goal_abs_z_error,
                "near_goal_2x_tolerance": near_goal,
                "support_opportunity": support_opportunity,
                "dense_opportunity": dense_opportunity,
                "above_demo_height": bool(
                    reward_metrics.get("above_demo_height", False)
                ),
                "reward": float(reward),
                "value": value,
                "next_value": next_value,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "collection_boundary": bool(collection_boundary),
                "done_reason_at_transition": str(info.get("done_reason", "none")),
                "reward_terms": {
                    str(name): float(value_)
                    for name, value_ in info.get("reward_terms", {}).items()
                },
                "reward_metrics": reward_metrics,
                "episode_fallback_type": getattr(
                    environment, "_episode_fallback_type", None
                ),
                "episode_fallback_reasons": list(
                    getattr(environment, "_episode_fallback_reasons", ())
                ),
            }
            block_rows.append(row)
            active_indices.append(len(block_rows) - 1)
            decision_index += 1
            observation = next_observation
            observation_context = info
            if terminated or truncated or collection_boundary:
                outcome = (
                    str(info.get("done_reason", "none"))
                    if terminated or truncated
                    else "collection_boundary"
                )
                for active_index in active_indices:
                    block_rows[active_index]["outcome"] = outcome
                adapter = None
                observation = None
                observation_context = None
                task = None
                bank_entry = None

        buffer.finalize()
        if args.run_d1b:
            d1b_before = _d1b_snapshot(system, buffer)
            d1b_report = replay_sampled_k_schedules(
                runtime, buffer, actor_parameters, device=device,
                schedules=100, seed=args.probe_seed,
            )
            reference_vectors = d1b_report.pop("_reference_vectors")
            schedule_vectors = d1b_report.pop("_schedule_vectors")
            array_dir = output_dir / "d1b_arrays"
            array_dir.mkdir()
            array_manifest = {}
            for role in sorted(reference_vectors):
                reference_path = array_dir / (role + "__full_joint.npy")
                schedules_path = array_dir / (role + "__sampled_k_100.npy")
                _write_npy(reference_path, reference_vectors[role].cpu().numpy())
                _write_npy(
                    schedules_path,
                    np.stack([row[role].cpu().numpy() for row in schedule_vectors]),
                )
                array_manifest[role] = {
                    "full_joint": {"path": str(reference_path.relative_to(output_dir)), "sha256": _sha256(reference_path)},
                    "sampled_k_100": {"path": str(schedules_path.relative_to(output_dir)), "sha256": _sha256(schedules_path)},
                }
            d1b_report["gradient_array_manifest"] = array_manifest
            d1b_report["identity"] = {
                "policy_label": args.policy_label,
                "checkpoint_update": checkpoint_update,
                "config_sha256": _sha256(config_path),
                "base_bc_checkpoint_sha256": _sha256(resolve(config.checkpoint)),
                "code_source_set_hash": actual_source_set_hash,
                "dev_bank_id": bank["bank_id"],
                "dev_bank_content_sha256": bank["content_sha256"],
                "task_count": len(factory.tasks),
            }
            d1b_report["tripwire"] = _d1b_tripwire(d1b_before, system, buffer)
            if not d1b_report["tripwire"]["all_unchanged"]:
                raise RuntimeError("D1b read-only tripwire detected state mutation")
            d1b_report["content_sha256"] = hashlib.sha256(
                json.dumps(
                    d1b_report,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
        targets = buffer.audit_tensors()
        for index, row in enumerate(block_rows):
            row["raw_advantage"] = float(targets["advantages"][index])
            row["return"] = float(targets["returns"][index])
            row["td_delta"] = float(
                row["reward"]
                + float(config.dppo.gamma_environment) * row["next_value"]
                - row["value"]
            )

        schedule_seed = seed + (block_index + 1) * 1009
        real_generator = torch.Generator().manual_seed(schedule_seed)
        shadow_generator = torch.Generator().manual_seed(schedule_seed)
        schedules = make_batch_schedule(
            args.block_transitions,
            batch_size=128,
            ft_denoising_steps=int(runtime.ft_denoising_steps),
            generator=shadow_generator,
        )
        batches = buffer.batches(128, generator=real_generator, device=device)
        for minibatch_index, (batch, schedule) in enumerate(zip(batches, schedules)):
            indices, denoising_inds = schedule
            if not torch.equal(batch.denoising_inds.cpu(), denoising_inds):
                raise RuntimeError("shadow denoising schedule diverged from rollout buffer")
            labels_outcome = tuple(block_rows[int(i)]["outcome"] for i in indices)
            labels_role = tuple(block_rows[int(i)]["role"] for i in indices)
            gradient_enabled = minibatch_index < args.gradient_batches_per_block
            if gradient_enabled:
                probed = probe_minibatch(
                    runtime,
                    batch,
                    actor_parameters=actor_parameters,
                    reward_start=system.trainer.config.reward_start,
                    reward_horizon=system.trainer.config.reward_horizon,
                    outcome_labels=labels_outcome,
                    role_labels=labels_role,
                )
                credit = probed["credit"]
                with torch.no_grad():
                    official_policy_loss = float(
                        runtime.loss(
                            batch.condition,
                            batch.chains_prev,
                            batch.chains_next,
                            batch.denoising_inds,
                            batch.returns,
                            batch.oldvalues,
                            batch.advantages.clone(),
                            batch.oldlogprobs,
                            use_bc_loss=False,
                            reward_horizon=system.trainer.config.reward_horizon,
                            reward_start=system.trainer.config.reward_start,
                        )[0]
                    )
                probe_policy_loss = float(
                    credit["per_sample_policy_loss"].mean().detach()
                )
                parity_abs = abs(official_policy_loss - probe_policy_loss)
                if parity_abs > 1e-6:
                    raise RuntimeError(
                        "credit math differs from official loss by {}".format(parity_abs)
                    )
                gradient_rows.append(
                    {
                        "schema": "dppo_credit_gradient_v1",
                        "block_index": block_index,
                        "minibatch_index": minibatch_index,
                        "official_policy_loss": official_policy_loss,
                        "probe_policy_loss": probe_policy_loss,
                        "policy_loss_parity_abs": parity_abs,
                        "by_outcome": probed["gradient_by_outcome"],
                        "by_role": probed["gradient_by_role"],
                    }
                )
            else:
                with torch.no_grad():
                    new_logprobs = runtime.get_logprobs_subsample(
                        batch.condition,
                        batch.chains_prev,
                        batch.chains_next,
                        batch.denoising_inds,
                    )

                    credit = compute_policy_credit(
                        new_logprobs,
                        batch.oldlogprobs,
                        advantages=batch.advantages,
                        denoising_inds=batch.denoising_inds,
                        reward_start=system.trainer.config.reward_start,
                        reward_horizon=system.trainer.config.reward_horizon,
                        norm_adv=bool(runtime.norm_adv),
                        gamma_denoising=float(runtime.gamma_denoising),
                        ft_denoising_steps=int(runtime.ft_denoising_steps),
                        clip_ploss_coef=float(runtime.clip_ploss_coef),
                        clip_ploss_coef_base=float(runtime.clip_ploss_coef_base),
                        clip_ploss_coef_rate=float(runtime.clip_ploss_coef_rate),
                    )

            if args.run_d1b:
                kl_record = reverse_kernel_kl_record(runtime, batch)
                kl_diversity_rows.append({
                    "schema": "dppo_credit_kl_diversity_v2",
                    "block_index": block_index,
                    "minibatch_index": minibatch_index,
                    **kl_record,
                })

            for local_index, source_index in enumerate(indices.tolist()):
                row = block_rows[source_index]
                clip_coefficient = float(credit["clip_coefficient"][local_index])
                geometric_ratio = float(credit["geometric_ratio"][local_index])
                clamped_mean_logratio = float(
                    credit["clamped_mean_logratio"][local_index]
                )
                clamped_joint_logratio = float(
                    credit["clamped_joint_logratio"][local_index]
                )
                clamped_joint_ratio = float(credit["joint_ratio"][local_index])
                row.update(
                    {
                        "minibatch_index": minibatch_index,
                        "denoising_index": int(batch.denoising_inds[local_index]),
                        "component_count": int(credit["component_count"]),
                        "raw_advantage_in_batch": float(batch.advantages[local_index]),
                        "normalized_advantage": float(
                            credit["normalized_advantage"][local_index]
                        ),
                        "discounted_advantage": float(
                            credit["discounted_advantage"][local_index]
                        ),
                        "clip_coefficient": clip_coefficient,
                        "preclamp_joint_logratio": float(
                            credit["preclamp_joint_logratio"][local_index]
                        ),
                        "clamped_joint_logratio": clamped_joint_logratio,
                        "clamped_mean_logratio": clamped_mean_logratio,
                        "geometric_ratio": geometric_ratio,
                        "clamped_joint_ratio": clamped_joint_ratio,
                        "geometric_approx_kl": (
                            geometric_ratio - 1.0 - clamped_mean_logratio
                        ),
                        "clamped_joint_approx_kl": (
                            clamped_joint_ratio - 1.0 - clamped_joint_logratio
                        ),
                        "geometric_clip_hit": bool(
                            abs(geometric_ratio - 1.0) > clip_coefficient
                        ),
                        "clamped_joint_clip_hit": bool(
                            clamped_joint_logratio < math.log1p(-clip_coefficient)
                            or clamped_joint_logratio
                            > math.log1p(clip_coefficient)
                        ),
                        "new_below_clamp_count": int(
                            credit["new_below_clamp_count"][local_index]
                        ),
                        "old_below_clamp_count": int(
                            credit["old_below_clamp_count"][local_index]
                        ),
                        "new_above_clamp_count": int(
                            credit["new_above_clamp_count"][local_index]
                        ),
                        "old_above_clamp_count": int(
                            credit["old_above_clamp_count"][local_index]
                        ),
                        "policy_loss": float(
                            credit["per_sample_policy_loss"][local_index].detach()
                        ),
                    }
                )
        transition_rows.extend(block_rows)
        print(
            json.dumps(
                {
                    "event": "credit_probe_block_complete",
                    "block": block_index + 1,
                    "blocks": args.blocks,
                    "transitions": len(transition_rows),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    parameter_delta = parameter_snapshot.measure(actor=actor_parameters)["actor"]
    if parameter_delta["max_abs"] != 0.0 or parameter_delta["nonzero_elements"] != 0:
        raise RuntimeError("credit probe changed actor parameters")
    for parameter, before_gradient in zip(actor_parameters, gradient_snapshot):
        if before_gradient is None:
            if parameter.grad is not None:
                raise RuntimeError("credit probe populated parameter .grad")
        elif parameter.grad is None or not torch.equal(parameter.grad, before_gradient):
            raise RuntimeError("credit probe changed an existing parameter .grad")
    if asdict(system.trainer.counters) != counter_snapshot:
        raise RuntimeError("credit probe changed training counters")
    if forbidden_calls:
        raise RuntimeError("credit probe attempted a forbidden training call")

    summary = {
        "schema": "dppo_credit_probe_summary_v2",
        "status": "PROBE_COMPLETE_NO_PARAMETER_UPDATE",
        "policy_label": args.policy_label,
        "initialization": initialization,
        "dev_bank_id": bank["bank_id"],
        "dev_bank_content_sha256": bank["content_sha256"],
        "dev_bank_entries_started": bank_entry_cursor,
        "transitions": len(transition_rows),
        "blocks": args.blocks,
        "block_transitions": args.block_transitions,
        "minibatch_size": 128,
        "gradient_minibatches": len(gradient_rows),
        "parameter_delta": parameter_delta,
        "training_counters_unchanged": True,
        "parameter_gradients_unchanged": True,
        "training_call_tripwire_triggered": False,
        "no_ray_warnings": _warning_summary(transition_rows),
        "by_outcome": _summary(transition_rows, "outcome"),
        "by_role": _summary(transition_rows, "role"),
        "by_outcome_and_role": _summary(
            transition_rows, ("outcome", "role")
        ),
        "by_near_goal_2x_tolerance": _summary(
            transition_rows, "near_goal_2x_tolerance"
        ),
        "by_support_opportunity": _summary(
            transition_rows, "support_opportunity"
        ),
        "by_dense_opportunity": _summary(
            transition_rows, "dense_opportunity"
        ),
    }
    provenance = {
        "schema": "dppo_credit_probe_provenance_v2",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "base_bc_checkpoint": str(resolve(config.checkpoint)),
        "base_bc_checkpoint_sha256": _sha256(resolve(config.checkpoint)),
        "policy_label": args.policy_label,
        "initialization": initialization,
        "rl_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_update": checkpoint_update,
        "dev_bank": str(bank_path),
        "dev_bank_file_sha256": _sha256(bank_path),
        "dev_bank_content_sha256": bank["content_sha256"],
        "output_dir": str(output_dir),
        "probe_seed": seed,
        "training_enabled": False,
        "optimizer_step_called": False,
        "training_call_tripwire_armed": True,
        "pointcloud_calibration_mode": bool(args.pointcloud_calibration_mode),
        "expected_source_set_hash": args.expected_source_set_hash,
        "actual_source_set_hash": actual_source_set_hash,
        "script_sha256": _sha256(Path(__file__).resolve()),
        "credit_module_sha256": _sha256(
            repo_root
            / "scripts"
            / "dppo_credit_support"
            / "credit_probe.py"
        ),
        "credit_bank_module_sha256": _sha256(
            repo_root
            / "scripts"
            / "dppo_credit_support"
            / "credit_bank.py"
        ),
    }
    _write_jsonl(output_dir / "transitions.jsonl", transition_rows)
    _write_jsonl(output_dir / "gradient_minibatches.jsonl", gradient_rows)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "provenance.json", provenance)
    if args.run_d1b:
        _write_json(output_dir / "d1b_summary.json", d1b_report)
        _write_jsonl(output_dir / "gradient_records.jsonl", [
            {"schema": "d1b_gradient_role_gate_v2", "role": role, **record}
            for role, record in d1b_report["s0"]["by_role"].items()
        ])
        _write_jsonl(output_dir / "kl_diversity_records.jsonl", kl_diversity_rows)
        _write_json(output_dir / "parameter_tripwire.json", d1b_report["tripwire"])
        _write_json(output_dir / "gate_decision.json", {
            "schema": "routea_v17_d1b_gate_decision_v2",
            "decision": d1b_report["s0"]["decision"],
            "selected_schedule": "S0" if d1b_report["s0"]["decision"] == "PASS" else "S2",
            "failure_codes": d1b_report["s0"]["failure_codes"],
        })
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--policy-label", choices=("bc", "rl"))
    parser.add_argument("--dev-bank", type=Path)
    parser.add_argument("--rl-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-source-set-hash")
    parser.add_argument("--expected-update", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--block-transitions", type=int, default=1024)
    parser.add_argument("--gradient-batches-per-block", type=int, default=1)
    parser.add_argument("--probe-seed", type=int, default=20260909)
    parser.add_argument("--run-d1b", action="store_true")
    parser.add_argument("--pointcloud-calibration-mode", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        print(json.dumps(_self_check(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.blocks <= 0:
        parser.error("--blocks must be positive")
    if args.run_d1b and args.blocks != 1:
        parser.error("--run-d1b requires --blocks 1 for the registered 1024-transition bank")
    if args.pointcloud_calibration_mode and (args.policy_label not in (None, "bc") or args.run_d1b):
        parser.error("--pointcloud-calibration-mode is BC-only and cannot run D1b")
    if args.block_transitions <= 0 or args.block_transitions % 128:
        parser.error("--block-transitions must be positive and divisible by 128")
    batches = args.block_transitions // 128
    if not 0 <= args.gradient_batches_per_block <= batches:
        parser.error("--gradient-batches-per-block must be within the block")
    if args.probe_seed < 0:
        parser.error("--probe-seed must be non-negative")
    missing = [
        name
        for name, value in (
            ("--config", args.config),
            ("--policy-label", args.policy_label),
            ("--dev-bank", args.dev_bank),
            ("--output-dir", args.output_dir),
            ("--expected-source-set-hash", args.expected_source_set_hash),
        )
        if value is None
    ]
    if missing:
        parser.error("live probe requires {}".format(", ".join(missing)))
    try:
        if len(args.expected_source_set_hash) != 64:
            raise ValueError
        int(args.expected_source_set_hash, 16)
    except ValueError:
        parser.error("--expected-source-set-hash must be 64 lowercase hex characters")
    if args.expected_source_set_hash.lower() != args.expected_source_set_hash:
        parser.error("--expected-source-set-hash must be 64 lowercase hex characters")
    if args.policy_label == "rl" and args.rl_checkpoint is None:
        parser.error("--rl-checkpoint is required for --policy-label rl")
    if args.policy_label == "bc" and args.rl_checkpoint is not None:
        parser.error("--rl-checkpoint is forbidden for --policy-label bc")
    if args.output_dir.exists():
        parser.error("refusing to overwrite existing --output-dir")
    return _run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
