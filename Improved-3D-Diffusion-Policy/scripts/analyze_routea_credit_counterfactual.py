#!/usr/bin/env python3
"""Read-only multi-seed D2-C-LAC diagnostic.

The CLI captures one immutable 1,024-transition buffer per lane, computes the
five preregistered credit variants with a frozen critic/trajectory, and then
aggregates u0 plus seed1/2/3-u32 without authorizing training.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import warnings

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.routea_causal_diagnostics.credit_counterfactual import (
    REFERENCE_NAMES,
    VARIANT_NAMES,
    build_credit_variants,
    build_reference_weights,
    compute_credit_variant_gradient,
)


SEEDS = ("seed1", "seed2", "seed3")
LANES = tuple(
    lane for seed in SEEDS for lane in (seed + "-u0", seed + "-u32")
)
TRAINING_SOURCE_SET_HASH = "5aed4fb675591750ac97ad4165ac11b8d3fedf5a15dfa2bf0bc84266c373f637"


class D2CError(RuntimeError):
    def __init__(self, status, detail, exit_code=10):
        super().__init__(detail)
        self.status = str(status)
        self.detail = str(detail)
        self.exit_code = int(exit_code)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_content_sha256(path):
    path = Path(path).resolve()
    if path.is_file():
        return _sha256(path)
    if not path.is_dir():
        raise D2CError("INVALID_INPUT", "missing path: {}".format(path), 2)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise D2CError("INVALID_INPUT", "empty directory: {}".format(path), 2)
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _validate_runtime_identity(
    capture_identity,
    *,
    lane,
    base_checkpoint_sha256,
    checkpoint_sha256,
):
    if capture_identity.get("base_checkpoint_sha256") != base_checkpoint_sha256:
        raise D2CError(
            "CHECKPOINT_CONTRACT_MISMATCH",
            "base BC checkpoint changed after capture",
            11,
        )
    expected_rl = capture_identity.get("checkpoint_sha256")
    if str(lane).endswith("-u0"):
        if expected_rl is not None or checkpoint_sha256 is not None:
            raise D2CError(
                "CHECKPOINT_CONTRACT_MISMATCH",
                "u0 lane must not bind an RL checkpoint",
                11,
            )
    elif expected_rl != checkpoint_sha256:
        raise D2CError(
            "CHECKPOINT_CONTRACT_MISMATCH",
            "RL checkpoint changed after capture",
            11,
        )


def _tensor_sha256(tensor):
    array = np.ascontiguousarray(torch.as_tensor(tensor).detach().cpu().numpy())
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_json(path, payload):
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    with Path(path).open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _write_jsonl(path, rows):
    with Path(path).open("xb") as stream:
        for row in rows:
            stream.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise D2CError(
                    "INVALID_INPUT", "invalid JSONL {}:{}: {}".format(path, line_number, exc), 2
                )
    return rows


def _write_manifest(directory, names):
    lines = ["{}  {}".format(_sha256(Path(directory) / name), name) for name in sorted(names)]
    path = Path(directory) / "manifest_sha256.txt"
    with path.open("xb") as stream:
        stream.write(("\n".join(lines) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _verify_manifest(directory):
    directory = Path(directory).resolve()
    manifest = directory / "manifest_sha256.txt"
    if not manifest.is_file():
        raise D2CError("INPUT_MANIFEST_MISMATCH", "missing {}".format(manifest))
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64:
            raise D2CError("INPUT_MANIFEST_MISMATCH", "malformed manifest line")
        target = (directory / relative).resolve()
        try:
            target.relative_to(directory)
        except ValueError:
            raise D2CError("INPUT_MANIFEST_MISMATCH", "manifest path escapes directory")
        if not target.is_file() or _sha256(target) != digest:
            raise D2CError("INPUT_MANIFEST_MISMATCH", "hash mismatch: {}".format(relative))
        count += 1
    if count == 0:
        raise D2CError("INPUT_MANIFEST_MISMATCH", "empty manifest")
    return count


def _seed_all(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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


def _system_snapshot(system):
    return {
        "actor": _clone_tree(system.lora_bundle.adapter_state_dict()),
        "actor_grad": _clone_tree(tuple(parameter.grad for parameter in system.trainer.actor_parameters)),
        "critic": _clone_tree(system.critic.state_dict()),
        "critic_grad": _clone_tree(tuple(parameter.grad for parameter in system.trainer.critic_parameters)),
        "optimizers": _clone_tree((system.actor_optimizer.state_dict(), system.critic_optimizer.state_dict())),
        "counters": _clone_tree(asdict(system.trainer.counters)),
    }


def _tripwire(before, system):
    after = _system_snapshot(system)
    checks = {name: _tree_equal(before[name], after[name]) for name in before}
    checks["all_unchanged"] = all(checks.values())
    return checks


def _self_check():
    from scripts.audit_checkpoint_source_identity import build_inventory

    _, source_hash = build_inventory(REPO_ROOT)
    rows = [
        {
            "transition_id": "self:t0",
            "reward": 1.0,
            "value": 0.0,
            "next_value": 0.0,
            "terminated": True,
            "truncated": False,
            "collection_boundary": False,
            "terminal_facts": {"done_reason": "reached"},
            "role": "self",
            "episode_identity_hash": "self-episode",
            "decision_index": 0,
            "goal_distance_before_3d": 1.0,
            "goal_distance_after_3d": 0.0,
            "entered_goal_sphere_safe": True,
            "pair_class": "stable_success",
            "outcome": "reached",
        }
    ]
    variants = build_credit_variants(
        rows,
        gamma=0.99,
        gae_lambda=0.95,
        official_advantages=torch.tensor([1.0], dtype=torch.float32),
    )
    references = build_reference_weights(rows)
    if tuple(variants) != VARIANT_NAMES or tuple(references) != REFERENCE_NAMES:
        raise RuntimeError("D2-C self-check variant registry mismatch")
    if source_hash != TRAINING_SOURCE_SET_HASH:
        raise RuntimeError(
            "training source-set hash changed: expected {}, got {}".format(
                TRAINING_SOURCE_SET_HASH, source_hash
            )
        )
    return {
        "status": "SELF_CHECK_PASS",
        "training_source_set_hash": source_hash,
        "lanes": list(LANES),
        "variants": list(VARIANT_NAMES),
        "references": list(REFERENCE_NAMES),
        "training_performed": False,
    }


def _run_finalize(args):
    runs_root = args.runs_root.resolve()
    actual_lane_dirs = {path.name for path in runs_root.iterdir() if path.is_dir()}
    if actual_lane_dirs != set(LANES):
        raise D2CError(
            "D2C_LANE_SET_MISMATCH",
            "expected lanes {}, got {}".format(sorted(LANES), sorted(actual_lane_dirs)),
        )
    summaries = {}
    for lane in LANES:
        lane_dir = runs_root / lane
        if not lane_dir.is_dir():
            raise D2CError("D2C_LANE_SET_MISMATCH", "missing lane directory: {}".format(lane))
        _verify_manifest(lane_dir)
        summary = _read_json(lane_dir / "lane_summary.json")
        if summary.get("schema") != "routea_d2c_lane_summary_v1" or summary.get("lane") != lane:
            raise D2CError("D2C_LANE_IDENTITY_MISMATCH", "invalid lane summary: {}".format(lane))
        if int(summary.get("transition_count", -1)) != 1024 or int(summary.get("pair_count", -1)) != 10240:
            raise D2CError("D2C_COUNT_MISMATCH", "invalid transition/pair count: {}".format(lane))
        if summary.get("tripwire_all_unchanged") is not True:
            raise D2CError("READ_ONLY_TRIPWIRE_FAILED", "tripwire failed: {}".format(lane), 12)
        summaries[lane] = summary
    bank_hashes = {}
    for seed in SEEDS:
        u0_hash = str(summaries[seed + "-u0"].get("bank_content_sha256"))
        u32_hash = str(summaries[seed + "-u32"].get("bank_content_sha256"))
        if u0_hash != u32_hash:
            raise D2CError(
                "D2C_BANK_IDENTITY_MISMATCH",
                "{} u0/u32 do not share one dev bank".format(seed),
            )
        bank_hashes[seed] = u0_hash
    base_hashes = {str(summary.get("base_checkpoint_sha256")) for summary in summaries.values()}
    if len(base_hashes) != 1 or "None" in base_hashes:
        raise D2CError(
            "D2C_BASE_CHECKPOINT_MISMATCH",
            "all six lanes must share one BC checkpoint identity",
        )
    for seed in SEEDS:
        if summaries[seed + "-u0"].get("u0_zero_effect") is not True:
            raise D2CError(
                "CHECKPOINT_CONTRACT_MISMATCH",
                "{} u0 LoRA is not exact zero-effect".format(seed),
                11,
            )
    if args.output_dir.exists():
        raise D2CError("OUTPUT_EXISTS", "refusing to overwrite --output-dir", 2)

    seed_statuses = [summaries[seed + "-u32"]["status"] for seed in SEEDS]
    supported = "LOCAL_ACTOR_CREDIT_ROTATION_CREDIT"
    if all(status == supported for status in seed_statuses):
        status = "CROSS_SEED_LOCAL_ACTOR_CREDIT_ROTATION_CREDIT"
        next_action = "design a small-budget multi-seed training ablation; do not start formal training"
    elif seed_statuses[-1] == supported and all(status != supported for status in seed_statuses[:2]):
        status = "SEED3_SPECIFIC_LOCAL_ACTOR_CREDIT_ROTATION"
        next_action = "localize seed3 update history before any training change"
    elif any(status == supported for status in seed_statuses):
        status = "SEED_SPECIFIC_COMPOSITE"
        next_action = "inspect per-seed critic and state-distribution differences before training"
    else:
        status = "NO_MATERIAL_CREDIT_ROTATION"
        next_action = "stop timeout-credit route and inspect action chunk/history/remaining-time feedback"
    verdict = {
        "schema": "routea_d2c_verdict_v1",
        "status": status,
        "causal_scope": "frozen_critic_fixed_trajectory_local_actor_credit",
        "training_authorized": False,
        "formal_training_authorized": False,
        "reward_change_authorized": False,
        "next_allowed_action": next_action,
        "bank_content_sha256_by_seed": bank_hashes,
        "base_checkpoint_sha256": next(iter(base_hashes)),
        "lane_statuses": {lane: summaries[lane]["status"] for lane in LANES},
        "lane_best_variants": {lane: summaries[lane].get("best_variant") for lane in LANES},
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(args.output_dir / "verdict.json", verdict)
    _write_json(args.output_dir / "lane_summaries.json", summaries)
    _write_manifest(args.output_dir, ("verdict.json", "lane_summaries.json"))
    print(json.dumps({"status": "FINALIZE_D2C_COMPLETE", "verdict": status}, sort_keys=True))
    return 0


def _run_capture(args):
    from omegaconf import OmegaConf

    from diffusion_policy_3d.rl.dppo_uav_rebuild.deployment.loader import load_deployment_policy
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.rollout_buffer import OnPolicyRolloutBuffer
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.build_system import build_system
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.factory import RouteAEnvironmentFactory
    from scripts.analyze_routea_reward_credit_divergence import (
        _executed_action_4x11,
        _first_sphere_entry,
    )
    from scripts.audit_checkpoint_source_identity import build_inventory
    from scripts.dppo_credit_support.credit_bank import load_dev_bank

    _, actual_source_hash = build_inventory(REPO_ROOT)
    if actual_source_hash != args.expected_source_set_hash:
        raise D2CError(
            "SOURCE_SET_MISMATCH",
            "expected {}, got {}".format(args.expected_source_set_hash, actual_source_hash),
            11,
        )
    config = OmegaConf.load(args.config.resolve())
    _seed_all(int(config.run.seed))
    if args.lane.endswith("-u0"):
        system = build_system(config, REPO_ROOT)
        checkpoint_update = 0
        checkpoint_hash = None
    else:
        deployment = load_deployment_policy(config, args.checkpoint.resolve(), REPO_ROOT)
        system = deployment.system
        checkpoint_update = int(deployment.checkpoint_header.update_index)
        if checkpoint_update != 32:
            raise D2CError(
                "CHECKPOINT_CONTRACT_MISMATCH",
                "{} requires update 32, got {}".format(args.lane, checkpoint_update),
                11,
            )
        checkpoint_hash = _path_content_sha256(args.checkpoint.resolve())
    runtime = system.official_runtime
    if (
        float(runtime.clip_advantage_lower_quantile),
        float(runtime.clip_advantage_upper_quantile),
    ) != (0.0, 1.0):
        raise D2CError("RUNTIME_CONTRACT_MISMATCH", "D2-C requires 0/100 advantage clipping", 11)
    device = torch.device(system.bc_runtime.device)

    def resolve(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()

    base_checkpoint_hash = _path_content_sha256(resolve(config.checkpoint))
    adapter_state = system.lora_bundle.adapter_state_dict()
    lora_b_tensors = [
        tensor for name, tensor in adapter_state.items() if str(name).endswith("lora_b")
    ]
    u0_zero_effect = None
    if args.lane.endswith("-u0"):
        u0_zero_effect = bool(
            lora_b_tensors
            and all(bool(torch.count_nonzero(tensor).item() == 0) for tensor in lora_b_tensors)
        )
        if not u0_zero_effect:
            raise D2CError(
                "CHECKPOINT_CONTRACT_MISMATCH",
                "fresh u0 LoRA does not have exact zero effect",
                11,
            )

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
    manifest_path = resolve(config.admission_manifest)
    expected_tasks = [
        {
            "task_id": task.task_id,
            "world_id": task.world_id,
            "role": task.role,
            "task_seed": task.seed,
            "obstacle_sha256": _sha256(Path(task.obstacle_json).resolve()),
        }
        for task in factory.tasks
    ]
    bank = load_dev_bank(
        args.dev_bank.resolve(),
        expected_manifest_sha256=_sha256(manifest_path),
        expected_tasks=expected_tasks,
    )
    task_by_id = {task.task_id: task for task in factory.tasks}
    before = _system_snapshot(system)
    forbidden_calls = []

    def forbid_training_call(*unused_args, **unused_kwargs):
        forbidden_calls.append("training_call")
        raise RuntimeError("D2-C forbids optimizer/trainer update calls")

    system.actor_optimizer.step = forbid_training_call
    system.critic_optimizer.step = forbid_training_call
    system.trainer.update = forbid_training_call

    capacity = 1024
    buffer = OnPolicyRolloutBuffer(
        capacity,
        gamma=float(config.dppo.gamma_environment),
        gae_lambda=float(config.dppo.gae_lambda),
        outer_update_id=checkpoint_update,
    )
    rows = []
    bank_cursor = 0
    adapter = None
    observation = None
    context = None
    task = None
    bank_entry = None
    decision_index = 0
    active_indices = []
    while len(buffer) < capacity:
        if adapter is None:
            if bank_cursor >= len(bank["entries"]):
                raise D2CError("DEV_BANK_EXHAUSTED", "dev bank exhausted before 1024 transitions", 2)
            bank_entry = bank["entries"][bank_cursor]
            bank_cursor += 1
            task = task_by_id[bank_entry["task_id"]]
            adapter = factory.make(task)
            observation, context = adapter.reset(seed=int(bank_entry["reset_seed"]))
            _seed_all(int(bank_entry["policy_seed"]))
            decision_index = 0
            active_indices = []

        encoded = system.observation_adapter.encode(
            observation,
            context["pointcloud_meta_history"],
            episode_identity_hash=context["episode_identity_hash"],
            decision_index=context["decision_index"],
        )
        system.observation_adapter.verify(encoded)
        condition = encoded.condition
        before_position = np.asarray(adapter.environment._drone_pos, dtype=np.float64).copy()
        goal = np.asarray(adapter.environment.goal_w, dtype=np.float64).copy()
        with torch.no_grad():
            value = float(system.critic(condition).reshape(-1)[0].item())
            rollout = system.rollout_sampler.sample(condition, deterministic=False)
        next_observation, reward, terminated, truncated, info = adapter.step(rollout.action)
        collection_boundary = bool(len(buffer) + 1 == capacity and not (terminated or truncated))
        next_encoded = system.observation_adapter.encode(
            next_observation,
            info["pointcloud_meta_history"],
            episode_identity_hash=info["episode_identity_hash"],
            decision_index=info["decision_index"],
        )
        system.observation_adapter.verify(next_encoded)
        with torch.no_grad():
            next_value = 0.0 if terminated else float(
                system.critic(next_encoded.condition).reshape(-1)[0].item()
            )
        transition_id = "d2c:{}:b{}:d{}".format(
            args.lane, int(bank_entry["entry_index"]), decision_index
        )
        buffer.add(
            condition={name: tensor[0] for name, tensor in condition.items()},
            chain=rollout.normalized_chain[0],
            old_logprobs=rollout.old_logprobs[0],
            reward=float(reward),
            value=value,
            next_value=next_value,
            terminated=bool(terminated),
            truncated=bool(truncated or collection_boundary),
            transition_id=transition_id,
            role=task.role,
            outcome="pending",
            provenance_hash=encoded.provenance.policy_input_hash,
        )
        after_position = np.asarray(adapter.environment._drone_pos, dtype=np.float64).copy()
        executed_path = np.asarray(info.get("executed_path"), dtype=np.float64)
        entered, _, _ = _first_sphere_entry(executed_path, goal, radius=0.5)
        facts = info.get("transition_facts")
        physical_failure = bool(getattr(facts, "physical_failure", False))
        terminal_facts = (
            asdict(facts)
            if hasattr(facts, "__dataclass_fields__")
            else dict(facts) if isinstance(facts, dict) else {}
        )
        done_reason = str(info.get("done_reason", "none"))
        terminal_facts["done_reason"] = done_reason
        rows.append(
            {
                "schema": "routea_d2c_transition_fact_v1",
                "lane": args.lane,
                "transition_id": transition_id,
                "bank_entry_index": int(bank_entry["entry_index"]),
                "bank_id": str(bank["bank_id"]),
                "task_id": task.task_id,
                "world_id": task.world_id,
                "role": task.role,
                "episode_identity_hash": str(context["episode_identity_hash"]),
                "decision_index": decision_index,
                "reward": float(reward),
                "value": value,
                "next_value": next_value,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "collection_boundary": collection_boundary,
                "terminal_facts": terminal_facts,
                "outcome": "pending",
                "position_before_xyz": before_position.tolist(),
                "position_after_xyz": after_position.tolist(),
                "goal_xyz": goal.tolist(),
                "goal_distance_before_3d": float(np.linalg.norm(goal - before_position)),
                "goal_distance_after_3d": float(np.linalg.norm(goal - after_position)),
                "executed_action_4x11": _executed_action_4x11(rollout.action),
                "entered_goal_sphere_safe": bool(entered and not physical_failure),
                "pair_class": "pending",
                "policy_input_hash": encoded.provenance.policy_input_hash,
            }
        )
        active_indices.append(len(rows) - 1)
        observation = next_observation
        context = info
        decision_index += 1
        if terminated or truncated or collection_boundary:
            outcome = done_reason if (terminated or truncated) else "collection_boundary"
            for index in active_indices:
                rows[index]["outcome"] = outcome
                buffer._items[index]["outcome"] = outcome
            adapter = observation = context = task = bank_entry = None

    buffer.finalize()
    tripwire = _tripwire(before, system)
    tripwire["forbidden_call_count"] = len(forbidden_calls)
    if not tripwire["all_unchanged"] or forbidden_calls:
        raise D2CError("READ_ONLY_TRIPWIRE_FAILED", "capture changed training state", 12)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    buffer_path = args.output_dir / "buffer_state.pt"
    with buffer_path.open("xb") as stream:
        torch.save(buffer.state_dict(), stream)
        stream.flush()
        os.fsync(stream.fileno())
    _write_jsonl(args.output_dir / "transition_facts.jsonl", rows)
    identity = {
        "schema": "routea_d2c_capture_identity_v1",
        "lane": args.lane,
        "checkpoint_update": checkpoint_update,
        "checkpoint_sha256": checkpoint_hash,
        "base_checkpoint_sha256": base_checkpoint_hash,
        "u0_zero_effect": u0_zero_effect,
        "config_sha256": _sha256(args.config.resolve()),
        "dev_bank_sha256": _sha256(args.dev_bank.resolve()),
        "bank_content_sha256": str(bank["content_sha256"]),
        "bank_id": str(bank["bank_id"]),
        "transition_count": len(rows),
        "buffer_state_sha256": _sha256(buffer_path),
        "training_source_set_hash": actual_source_hash,
    }
    _write_json(args.output_dir / "identity.json", identity)
    _write_json(args.output_dir / "tripwire_verification.json", tripwire)
    names = ("buffer_state.pt", "transition_facts.jsonl", "identity.json", "tripwire_verification.json")
    _write_manifest(args.output_dir, names)
    print(json.dumps({"status": "CAPTURE_D2C_COMPLETE", "lane": args.lane, "transitions": len(rows)}, sort_keys=True))
    return 0


def _run_credit(args):
    import csv

    from omegaconf import OmegaConf

    from diffusion_policy_3d.rl.dppo_uav_rebuild.deployment.loader import load_deployment_policy
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.rollout_buffer import OnPolicyRolloutBuffer
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.build_system import build_system
    from scripts.audit_checkpoint_source_identity import build_inventory
    from scripts.dppo_credit_support.credit_probe import compute_full_joint_gradient_reference

    _, actual_source_hash = build_inventory(REPO_ROOT)
    if actual_source_hash != args.expected_source_set_hash:
        raise D2CError(
            "SOURCE_SET_MISMATCH",
            "expected {}, got {}".format(args.expected_source_set_hash, actual_source_hash),
            11,
        )
    captures = {}
    facts_by_lane = {}
    for lane in LANES:
        directory = args.capture_root.resolve() / lane
        _verify_manifest(directory)
        identity = _read_json(directory / "identity.json")
        if identity.get("lane") != lane or int(identity.get("transition_count", -1)) != 1024:
            raise D2CError("D2C_CAPTURE_IDENTITY_MISMATCH", "invalid capture: {}".format(lane))
        if identity.get("training_source_set_hash") != actual_source_hash:
            raise D2CError("SOURCE_SET_MISMATCH", "capture source hash differs: {}".format(lane), 11)
        facts = _read_jsonl(directory / "transition_facts.jsonl")
        if len(facts) != 1024:
            raise D2CError("D2C_COUNT_MISMATCH", "capture must contain 1024 facts: {}".format(lane))
        captures[lane] = (directory, identity)
        facts_by_lane[lane] = facts
    for seed in SEEDS:
        if (
            captures[seed + "-u0"][1]["bank_content_sha256"]
            != captures[seed + "-u32"][1]["bank_content_sha256"]
        ):
            raise D2CError(
                "D2C_BANK_IDENTITY_MISMATCH",
                "{} capture pair uses different dev banks".format(seed),
            )

    def outcomes(rows):
        result = {}
        for row in rows:
            key = int(row["bank_entry_index"])
            value = str(row["outcome"])
            if key in result and result[key] != value:
                raise D2CError("D2C_EPISODE_OUTCOME_MISMATCH", "inconsistent episode outcome")
            result[key] = value
        return result

    seed_name = args.lane.split("-", 1)[0]
    baseline_lane = seed_name + "-u0"
    bc_outcomes = outcomes(facts_by_lane[baseline_lane])
    lane_outcomes = outcomes(facts_by_lane[args.lane])
    enriched = []
    for row in facts_by_lane[args.lane]:
        item = dict(row)
        key = int(item["bank_entry_index"])
        bc = bc_outcomes.get(key, "collection_boundary")
        rl = lane_outcomes.get(key, "collection_boundary")
        if "collection_boundary" in (bc, rl):
            pair_class = "indeterminate"
        elif bc == "reached" and rl != "reached":
            pair_class = "break"
        elif bc != "reached" and rl == "reached":
            pair_class = "rescue"
        elif bc == "reached" and rl == "reached":
            pair_class = "stable_success"
        else:
            pair_class = "stable_failure"
        item["pair_class"] = pair_class
        enriched.append(item)

    config = OmegaConf.load(args.config.resolve())
    _seed_all(int(config.run.seed))
    if args.lane.endswith("-u0"):
        system = build_system(config, REPO_ROOT)
        checkpoint_update = 0
    else:
        deployment = load_deployment_policy(config, args.checkpoint.resolve(), REPO_ROOT)
        system = deployment.system
        checkpoint_update = int(deployment.checkpoint_header.update_index)
        if checkpoint_update != 32:
            raise D2CError("CHECKPOINT_CONTRACT_MISMATCH", "D2-C requires u32", 11)
    runtime = system.official_runtime
    device = torch.device(system.bc_runtime.device)
    before = _system_snapshot(system)
    forbidden_calls = []

    def forbid_training_call(*unused_args, **unused_kwargs):
        forbidden_calls.append("training_call")
        raise RuntimeError("D2-C forbids optimizer/trainer update calls")

    system.actor_optimizer.step = forbid_training_call
    system.critic_optimizer.step = forbid_training_call
    system.trainer.update = forbid_training_call

    capture_dir, capture_identity = captures[args.lane]
    base_checkpoint_path = Path(config.checkpoint)
    if not base_checkpoint_path.is_absolute():
        base_checkpoint_path = REPO_ROOT / base_checkpoint_path
    current_checkpoint_hash = (
        None
        if args.lane.endswith("-u0")
        else _path_content_sha256(args.checkpoint.resolve())
    )
    _validate_runtime_identity(
        capture_identity,
        lane=args.lane,
        base_checkpoint_sha256=_path_content_sha256(base_checkpoint_path.resolve()),
        checkpoint_sha256=current_checkpoint_hash,
    )
    with (capture_dir / "buffer_state.pt").open("rb") as stream:
        state = torch.load(stream, map_location="cpu")
    buffer = OnPolicyRolloutBuffer(
        int(state["capacity"]),
        gamma=float(state["gamma"]),
        gae_lambda=float(state["gae_lambda"]),
        outer_update_id=state.get("outer_update_id"),
    )
    buffer.load_state_dict(state)
    audit = buffer.audit_tensors()
    variants = build_credit_variants(
        enriched,
        gamma=float(state["gamma"]),
        gae_lambda=float(state["gae_lambda"]),
        official_advantages=audit["advantages"],
    )
    references = build_reference_weights(enriched)
    actor_parameters = tuple(system.trainer.actor_parameters)
    parameter_entries = []
    offset = 0
    for name, parameter in system.lora_bundle.named_adapter_parameters():
        count = int(parameter.numel())
        parameter_entries.append(
            {
                "name": str(name),
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "flat_start": offset,
                "flat_end": offset + count,
            }
        )
        offset += count
    vector_names = list(VARIANT_NAMES) + list(REFERENCE_NAMES)
    roles = tuple(str(row["role"]) for row in enriched)
    tasks = tuple(str(row["task_id"]) for row in enriched)
    unique_roles = sorted(set(roles))
    unique_tasks = sorted(set(tasks))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    total_map = np.lib.format.open_memmap(
        args.output_dir / "gradient_vectors.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(vector_names), offset),
    )
    role_map = np.lib.format.open_memmap(
        args.output_dir / "role_gradient_vectors.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(vector_names), len(unique_roles), offset),
    )
    task_map = np.lib.format.open_memmap(
        args.output_dir / "task_level_gradient_vectors.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(vector_names), len(unique_tasks), offset),
    )
    gradient_meta = {}
    for vector_index, name in enumerate(vector_names):
        if name in variants:
            raw = variants[name]["raw_advantages"]
            normalization = "per_role" if name == "C4-role-c20" else "global"
        else:
            raw = references[name]["weights"]
            normalization = "none"
        if normalization == "none":
            if references[name]["nonzero_count"] == 0:
                total_map[vector_index] = 0.0
                role_map[vector_index] = 0.0
                task_map[vector_index] = 0.0
                gradient_meta[name] = {"zero_reference": True}
                continue
        result = compute_credit_variant_gradient(
            runtime,
            buffer,
            actor_parameters,
            raw_advantages=raw,
            roles=roles,
            group_labels=tasks,
            normalization=normalization,
            device=device,
            microbatch_pairs=int(args.microbatch_pairs),
        )
        total_map[vector_index] = result["total_descent"].numpy()
        for role_index, role in enumerate(unique_roles):
            role_map[vector_index, role_index] = result["role_descent"].get(
                role, torch.zeros(offset)
            ).numpy()
        for task_index, task in enumerate(unique_tasks):
            task_map[vector_index, task_index] = result["group_descent"].get(
                task, torch.zeros(offset)
            ).numpy()
        gradient_meta[name] = {
            "zero_reference": False,
            "normalization": normalization,
            "pair_count": result["pair_count"],
            "role_pair_counts": result["role_pair_counts"],
            "reconstruction_max_abs": result["reconstruction_max_abs"],
        }
        del result
    total_map.flush()
    role_map.flush()
    task_map.flush()

    # C0 must agree with the pre-existing official full-joint path.
    official_by_role = compute_full_joint_gradient_reference(
        runtime, buffer, actor_parameters, device=device
    )
    official_gradient = torch.stack(tuple(official_by_role.values())).sum(dim=0).numpy()
    c0_descent = np.asarray(total_map[0], dtype=np.float64)
    official_descent = -np.asarray(official_gradient, dtype=np.float64)
    denominator = max(float(np.linalg.norm(official_descent)), 1e-12)
    official_parity_rel_l2 = float(np.linalg.norm(c0_descent - official_descent) / denominator)
    if official_parity_rel_l2 > 1e-5:
        raise D2CError(
            "OFFICIAL_GRADIENT_PARITY_FAILED",
            "C0 relative L2 error {}".format(official_parity_rel_l2),
            10,
        )

    def cosine(left, right):
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        return None if denom < 1e-12 else float(np.dot(left, right) / denom)

    c0 = np.asarray(total_map[0], dtype=np.float64)
    c0_norm = float(np.linalg.norm(c0))
    reference_indices = {name: vector_names.index(name) for name in REFERENCE_NAMES}
    c0_refs = {
        name: cosine(c0, total_map[index]) for name, index in reference_indices.items()
    }
    comparisons = []
    variant_records = {}
    for name in VARIANT_NAMES:
        index = vector_names.index(name)
        vector = np.asarray(total_map[index], dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        refs = {ref: cosine(vector, total_map[ref_index]) for ref, ref_index in reference_indices.items()}
        improvements = {
            ref: (
                None if refs[ref] is None or c0_refs[ref] is None else refs[ref] - c0_refs[ref]
            )
            for ref in REFERENCE_NAMES
        }
        record = {
            "variant": name,
            "norm": norm,
            "norm_ratio_to_c0": None if c0_norm < 1e-12 else norm / c0_norm,
            "cosine_with_c0": cosine(vector, c0),
            "reference_cosines": refs,
            "reference_improvements": improvements,
        }
        variant_records[name] = record
        for ref in REFERENCE_NAMES:
            comparisons.append(
                {
                    "variant": name,
                    "reference": ref,
                    "variant_norm": norm,
                    "norm_ratio_to_c0": record["norm_ratio_to_c0"],
                    "cosine_with_c0": record["cosine_with_c0"],
                    "reference_cosine": refs[ref],
                    "reference_improvement_vs_c0": improvements[ref],
                }
            )
    candidates = []
    for name in VARIANT_NAMES[1:]:
        record = variant_records[name]
        aligned = [
            record["reference_improvements"][ref]
            for ref in ("R-safe-arrival", "R-rescue")
            if record["reference_improvements"][ref] is not None
        ]
        if (
            record["cosine_with_c0"] is not None
            and record["cosine_with_c0"] < 0.50
            and record["norm_ratio_to_c0"] is not None
            and record["norm_ratio_to_c0"] >= 0.20
            and aligned
            and max(aligned) >= 0.20
        ):
            candidates.append(name)
    best_variant = (
        max(
            candidates,
            key=lambda name: max(
                value
                for ref, value in variant_records[name]["reference_improvements"].items()
                if ref in ("R-safe-arrival", "R-rescue") and value is not None
            ),
        )
        if candidates
        else None
    )
    status = (
        "U0_IDENTITY_BASELINE"
        if args.lane.endswith("-u0")
        else "LOCAL_ACTOR_CREDIT_ROTATION_CREDIT"
        if best_variant is not None
        else "NO_MATERIAL_CREDIT_ROTATION"
    )

    advantage_path = args.output_dir / "advantage_variants.csv"
    with advantage_path.open("x", encoding="utf-8", newline="") as stream:
        fieldnames = ["transition_id", "task_id", "role", "outcome", "pair_class"]
        for name in VARIANT_NAMES:
            fieldnames.extend((name + "__raw", name + "__normalized"))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(enriched):
            output = {key: row[key] for key in fieldnames[:5]}
            for name in VARIANT_NAMES:
                output[name + "__raw"] = float(variants[name]["raw_advantages"][index])
                output[name + "__normalized"] = float(
                    variants[name]["normalized_advantages"][index]
                )
            writer.writerow(output)
    comparison_path = args.output_dir / "gradient_comparisons.csv"
    with comparison_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    _write_jsonl(args.output_dir / "transition_facts.jsonl", enriched)
    tripwire = _tripwire(before, system)
    tripwire.update(
        {
            "forbidden_call_count": len(forbidden_calls),
            "capture_buffer_hash_unchanged": _sha256(capture_dir / "buffer_state.pt")
            == capture_identity["buffer_state_sha256"],
        }
    )
    tripwire["all_unchanged"] = bool(
        tripwire["all_unchanged"]
        and tripwire["capture_buffer_hash_unchanged"]
        and not forbidden_calls
    )
    if not tripwire["all_unchanged"]:
        raise D2CError("READ_ONLY_TRIPWIRE_FAILED", "run changed frozen state", 12)
    gradient_manifest = {
        "schema": "routea_d2c_gradient_manifest_v1",
        "vector_names": vector_names,
        "role_names": unique_roles,
        "task_names": unique_tasks,
        "flat_parameter_count": offset,
        "parameters": parameter_entries,
        "vectors": gradient_meta,
        "official_c0_gradient_parity_rel_l2": official_parity_rel_l2,
        "gradient_vectors_sha256": _sha256(args.output_dir / "gradient_vectors.npy"),
        "role_gradient_vectors_sha256": _sha256(args.output_dir / "role_gradient_vectors.npy"),
        "task_level_gradient_vectors_sha256": _sha256(
            args.output_dir / "task_level_gradient_vectors.npy"
        ),
    }
    _write_json(args.output_dir / "gradient_manifest.json", gradient_manifest)
    _write_json(args.output_dir / "tripwire_verification.json", tripwire)
    summary = {
        "schema": "routea_d2c_lane_summary_v1",
        "lane": args.lane,
        "bank_content_sha256": capture_identity["bank_content_sha256"],
        "transition_count": len(enriched),
        "pair_count": len(enriched) * 10,
        "status": status,
        "best_variant": best_variant,
        "variants": variant_records,
        "c0_parity_max_abs": variants["C0-current"]["c0_parity_max_abs"],
        "official_c0_gradient_parity_rel_l2": official_parity_rel_l2,
        "tripwire_all_unchanged": tripwire["all_unchanged"],
        "base_checkpoint_sha256": capture_identity["base_checkpoint_sha256"],
        "u0_zero_effect": capture_identity.get("u0_zero_effect"),
        "training_performed": False,
    }
    _write_json(args.output_dir / "lane_summary.json", summary)
    names = (
        "advantage_variants.csv",
        "gradient_comparisons.csv",
        "gradient_manifest.json",
        "gradient_vectors.npy",
        "lane_summary.json",
        "role_gradient_vectors.npy",
        "task_level_gradient_vectors.npy",
        "transition_facts.jsonl",
        "tripwire_verification.json",
    )
    _write_manifest(args.output_dir, names)
    print(json.dumps({"status": "RUN_D2C_COMPLETE", "lane": args.lane, "verdict": status}, sort_keys=True))
    return 0


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    capture = subparsers.add_parser("capture-d2c")
    capture.add_argument("--lane", choices=LANES, required=True)
    capture.add_argument("--config", type=Path, required=True)
    capture.add_argument("--checkpoint", type=Path)
    capture.add_argument("--dev-bank", type=Path, required=True)
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.add_argument("--expected-source-set-hash", required=True)
    capture.add_argument("--probe-seed", type=int, default=20260925)

    run = subparsers.add_parser("run-d2c")
    run.add_argument("--lane", choices=LANES, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--capture-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--expected-source-set-hash", required=True)
    run.add_argument("--microbatch-pairs", type=int, choices=(128,), default=128)

    finalize = subparsers.add_parser("finalize-d2c")
    finalize.add_argument("--runs-root", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.self_check:
        if args.command is not None:
            parser.error("--self-check cannot be combined with a command")
        print(json.dumps(_self_check(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.command is None:
        parser.error("a command or --self-check is required")
    if args.command in ("capture-d2c", "run-d2c"):
        if args.lane.endswith("-u0") and args.checkpoint is not None:
            parser.error("--checkpoint is forbidden for lane u0")
        if args.lane.endswith("-u32") and args.checkpoint is None:
            parser.error("--checkpoint is required for RL lanes")
        if args.output_dir.exists():
            parser.error("refusing to overwrite existing --output-dir")
        if len(args.expected_source_set_hash) != 64 or any(
            character not in "0123456789abcdef" for character in args.expected_source_set_hash
        ):
            parser.error("--expected-source-set-hash must be 64 lowercase hex characters")
    try:
        return {
            "capture-d2c": _run_capture,
            "run-d2c": _run_credit,
            "finalize-d2c": _run_finalize,
        }[args.command](args)
    except D2CError as exc:
        print(
            json.dumps(
                {"status": exc.status, "detail": exc.detail, "exit_code": exc.exit_code},
                sort_keys=True,
            )
        )
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
