#!/usr/bin/env python3
"""First executed-H4 causal swap in the lightweight Route A environment.

The first action is the accepted frozen D0.5-B H8 bundle (only H4 executes).
All four arms then use policy forward on their own live observations. This is
diagnostic intervention, never a deployment action rewrite or training path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.routea_causal_diagnostics.contracts import (  # noqa: E402
    DiagnosticFailure,
    canonical_array_sha256,
    capture_rng_state,
    load_strict_json,
    restore_rng_state,
    write_strict_json,
)
from scripts.routea_causal_diagnostics.first_h4_swap import (  # noqa: E402
    ARM_SPECS,
    classify_swap,
    first_actions,
    select_matched_tasks,
)
from scripts.routea_causal_diagnostics.reporting import (  # noqa: E402
    AtomicOutputDirectory,
    file_sha256,
    publish_stage_manifest,
    verify_sha256_manifest,
    write_jsonl,
)


EXIT_REPLAY = 10
EXIT_CHECKPOINT = 11


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _refuse(status: str, detail: str, code: int = 2) -> DiagnosticFailure:
    return DiagnosticFailure(status, detail, exit_code=code)


def _read_first_rows(path: Path) -> Dict[tuple, list]:
    rows: Dict[tuple, list] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if int(row["decision_index"]) == 0:
                key = (str(row["task_id"]), int(row["repeat"]))
                rows.setdefault(key, []).append(row)
    return rows


def _read_episode_rows(path: Path, selected_keys: set) -> Dict[tuple, list]:
    rows: Dict[tuple, list] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (str(row["task_id"]), int(row["repeat"])) not in selected_keys:
                continue
            key = (str(row["task_id"]), int(row["repeat"]), str(row["lane"]))
            rows.setdefault(key, []).append(row)
    for key, episode in rows.items():
        episode.sort(key=lambda row: int(row["decision_index"]))
        if [int(row["decision_index"]) for row in episode] != list(range(len(episode))):
            raise _refuse("STAGE_INPUT_IDENTITY_MISMATCH", "non-consecutive D0.5-B transitions {}".format(key))
    return rows


def _source_manifest() -> Dict[str, Any]:
    files = (
        "scripts/run_routea_first_h4_swap.py",
        "scripts/routea_causal_diagnostics/first_h4_swap.py",
    )
    return {
        "schema": "routea_first_h4_swap_source_v1",
        "files": {name: file_sha256(REPO_ROOT / name) for name in files},
    }


def _validated_parent(capture_dir: Path, final_dir: Path) -> Dict[str, Any]:
    capture_files = verify_sha256_manifest(capture_dir)
    final_files = verify_sha256_manifest(final_dir)
    if not {"selection.json", "paired_trace_transitions.jsonl"} <= set(capture_files):
        raise _refuse("STAGE_INPUT_HASH_MISMATCH", "D0.5-B capture manifest incomplete")
    if not {"selection.json", "d05_verdict.json"} <= set(final_files):
        raise _refuse("STAGE_INPUT_HASH_MISMATCH", "D0.5-B final manifest incomplete")
    selection = load_strict_json(capture_dir / "selection.json")
    final_selection = load_strict_json(final_dir / "selection.json")
    verdict = load_strict_json(final_dir / "d05_verdict.json")
    if file_sha256(capture_dir / "selection.json") != file_sha256(final_dir / "selection.json"):
        raise _refuse("STAGE_INPUT_IDENTITY_MISMATCH", "capture/final selection differs")
    lineage = selection.get("run_lineage_id")
    if (
        lineage != final_selection.get("run_lineage_id")
        or lineage != verdict.get("run_lineage_id")
        or verdict.get("status") != "INCONCLUSIVE"
    ):
        raise _refuse("STAGE_INPUT_IDENTITY_MISMATCH", "D0.5-B lineage/verdict differs")
    if (
        selection.get("target_repeat_count") != 160
        or selection.get("target_task_count") != 49
        or selection.get("break_repeat_count") != 143
        or selection.get("rescue_repeat_count") != 17
    ):
        raise _refuse("STAGE_INPUT_IDENTITY_MISMATCH", "D0.5-B population changed")
    return {
        "selection": selection,
        "parent_identity": {
            "run_lineage_id": lineage,
            "capture_manifest_sha256": file_sha256(capture_dir / "manifest_sha256.txt"),
            "final_manifest_sha256": file_sha256(final_dir / "manifest_sha256.txt"),
            "selection_sha256": file_sha256(capture_dir / "selection.json"),
            "transitions_sha256": file_sha256(capture_dir / "paired_trace_transitions.jsonl"),
            "verdict_sha256": file_sha256(final_dir / "d05_verdict.json"),
        },
    }


def _select(args: argparse.Namespace) -> int:
    capture_dir, final_dir = Path(args.capture_dir), Path(args.final_dir)
    parent = _validated_parent(capture_dir, final_dir)
    first = _read_first_rows(capture_dir / "paired_trace_transitions.jsonl")
    eligible, excluded = [], []
    for entry in parent["selection"]["targets"]:
        key = (str(entry["task_id"]), int(entry["repeat"]))
        try:
            first_actions(first.get(key, []), *key)
        except ValueError as exc:
            excluded.append({"task_id": key[0], "repeat": key[1], "reason": str(exc)})
        else:
            eligible.append(entry)
    expected_exclusion = {
        ("routeA_rl_v1__anticipatory__rl_test__front_offset_obstacle__000003__cfg00", 2)
    }
    if len(eligible) != 159 or {(row["task_id"], row["repeat"]) for row in excluded} != expected_exclusion:
        raise _refuse("FIRST_H4_CONTRACT_MISMATCH", "paired initial-observation exclusion set changed")
    pairs = select_matched_tasks(eligible)
    role_counts = {role: sum(pair["role"] == role for pair in pairs) for role in ("anticipatory", "dense_recovery")}
    if len(pairs) != 8 or role_counts != {"anticipatory": 6, "dense_recovery": 2}:
        raise _refuse("SELECTION_CONTRACT_MISMATCH", "expected 6 anticipatory and 2 dense rescue/break pairs")
    tasks = [entry for pair in pairs for entry in (pair["break"], pair["rescue"])]
    if len({entry["task_id"] for entry in tasks}) != 16:
        raise _refuse("SELECTION_CONTRACT_MISMATCH", "selected task IDs are not independent")
    for entry in tasks:
        key = (str(entry["task_id"]), int(entry["repeat"]))
        try:
            first_actions(first.get(key, []), *key)
        except ValueError as exc:
            raise _refuse("FIRST_H4_CONTRACT_MISMATCH", "{}: {}".format(key, exc)) from exc
    output = {
        "schema": "routea_first_h4_swap_selection_v1",
        "evidence_scope": "lightweight_kinematic_diagnostic_intervention",
        "actor_rerun_after_first_h4": True,
        "training_authorized": False,
        "reward_change_authorized": False,
        "pair_count": len(pairs),
        "task_count": len(tasks),
        "arm_count": len(tasks) * len(ARM_SPECS),
        "eligible_repeat_count": len(eligible),
        "excluded_repeat_identities": excluded,
        "pairs": pairs,
        "parent_identity": parent["parent_identity"],
    }
    with AtomicOutputDirectory(args.output_dir) as staging:
        write_strict_json(staging / "selection.json", output)
        publish_stage_manifest(staging, _source_manifest())
    _emit({"status": "SELECT_FIRST_H4_COMPLETE", "pairs": 8, "tasks": 16, "arms": 64})
    return 0


def execute_four_arms(entry: Mapping[str, Any], run_arm) -> Dict[str, Any]:
    """Run controls first; refuse cross-arm interpretation if either fails parity."""

    result = {}
    for arm in ("bc_bc", "rl_rl"):
        donor, follow = ARM_SPECS[arm]
        result[arm] = run_arm(entry, arm, donor, follow)
        expected = entry["bc_done_reason"] if arm == "bc_bc" else entry["seed3_done_reason"]
        expected_summary = entry["expected_bc_summary"] if arm == "bc_bc" else entry["expected_seed3_summary"]
        terminal = np.asarray(result[arm]["terminal_xyz"], dtype=np.float64)
        expected_terminal = np.asarray(expected_summary["terminal_xyz"], dtype=np.float64)
        if (
            result[arm]["done_reason"] != expected
            or result[arm]["done_reason"] != expected_summary["done_reason"]
            or int(result[arm]["decision_count"]) != int(expected_summary["steps"])
            or terminal.shape != (3,)
            or expected_terminal.shape != (3,)
            or not np.allclose(terminal, expected_terminal, rtol=0.0, atol=1e-6)
        ):
            raise ValueError(
                "baseline endpoint mismatch {} r{} {} expected_reason={} actual_reason={} expected_steps={} actual_steps={} expected_xyz={} actual_xyz={}".format(
                    entry["task_id"], entry["repeat"], arm, expected, result[arm]["done_reason"],
                    expected_summary["steps"], result[arm]["decision_count"],
                    expected_terminal.tolist(), terminal.tolist(),
                )
            )
    for arm in ("bc_rl", "rl_bc"):
        donor, follow = ARM_SPECS[arm]
        result[arm] = run_arm(entry, arm, donor, follow)
    result["swap_pattern"] = classify_swap(
        *(result[arm]["done_reason"] == "reached" for arm in ("bc_bc", "rl_rl", "bc_rl", "rl_bc"))
    )
    return result


def _action_bundle(action: np.ndarray):
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ActionBundle, ActionSpec

    spec = ActionSpec.e5()
    full = np.zeros((1, spec.full_horizon, spec.action_dim), dtype=np.float32)
    full[:, spec.action_start : spec.action_start + spec.policy_horizon, :] = action
    return ActionBundle.from_full(full, spec)


def _executed_numpy(action) -> np.ndarray:
    value = action.rollout_executed[0]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (4, 11) or not np.isfinite(array).all():
        raise _refuse("EXECUTED_H4_CONTRACT_MISMATCH", "invalid action rollout H4")
    return array


def _verify_baseline_path(path: np.ndarray, expected_rows: Sequence[Mapping[str, Any]], decision: int) -> None:
    if decision >= len(expected_rows):
        raise ValueError("baseline path has more decisions than frozen parent")
    expected = np.asarray(expected_rows[decision]["executed_h4_path_5x3"], dtype=np.float32)
    if expected.shape != (5, 3) or not np.array_equal(path.view(np.uint32), expected.view(np.uint32)):
        raise ValueError("baseline path differs at decision {}".format(decision))


def _assert_first_input(task_id, arm, expected, provenance):
    if getattr(provenance, "policy_input_hash", None) == expected["policy_input_hash"]:
        return
    expected_hashes = {
        name: expected[name]
        for name in ("raw_frame_hash", "normalized_history_hash", "policy_input_hash", "episode_identity_hash")
    }
    actual_hashes = {
        name: getattr(provenance, name)
        for name in ("raw_frame_hash", "raw_history_hash", "normalizer_identity_hash",
                     "normalized_history_hash", "policy_input_hash", "episode_identity_hash")
    }
    if expected_hashes["raw_frame_hash"] != actual_hashes["raw_frame_hash"]:
        layer = "RAW_FRAME"
    elif expected_hashes["normalized_history_hash"] != actual_hashes["normalized_history_hash"]:
        layer = "NORMALIZED_HISTORY_OR_UPSTREAM"
    else:
        layer = "POLICY_INPUT"
    detail = json.dumps({
        "task_id": task_id, "arm": arm, "action_seed": int(expected["action_seed"]),
        "first_mismatch_layer": layer, "expected": expected_hashes, "actual": actual_hashes,
    }, sort_keys=True, separators=(",", ":"))
    raise _refuse("SNAPSHOT_REPLAY_MISMATCH", detail, EXIT_REPLAY)


def _run_one_arm(entry, actions, arm, donor, follow, systems, baseline_rows=None):
    import torch
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.observation_provenance import (
        ObservationIdentityMismatch, PointCloudRejected,
    )
    from scripts.evaluate_dppo_uav_formal_endpoint import (
        _E5Policy, _build_adapter, canonical_done_reason,
        canonical_episode_config, reset_fixed_episode,
    )
    from scripts.probe_routea_goal_sensitivity import _policy_seed, _repeat_identity

    config, system = systems[follow]
    donor_system = systems[donor][1]
    fixed = canonical_episode_config(entry["episode_config"])
    if fixed.config_hash != entry["config_hash"]:
        raise _refuse("EPISODE_CONFIG_IDENTITY_MISMATCH", str(entry["task_id"]))
    reset_seed, offset = _repeat_identity(fixed.config_hash, int(entry["repeat"]))
    if reset_seed is not None and reset_seed != int(entry["env_reset_seed"]):
        raise _refuse("EPISODE_CONFIG_IDENTITY_MISMATCH", "repeat/reset seed differs")
    geometry = Path(entry["geometry_path"])
    if file_sha256(geometry) != entry["geometry_sha256"]:
        raise _refuse("STAGE_INPUT_HASH_MISMATCH", "geometry differs: {}".format(geometry))
    adapter = _build_adapter(system, config, entry["world"], geometry, entry["role"])
    policy = _E5Policy(system, use_base_policy=(follow == "u0"), identity_hash=follow)
    try:
        observation, context = reset_fixed_episode(
            adapter.environment,
            fixed.payload,
            seed=int(entry["env_reset_seed"]),
            reset_callable=adapter.reset,
        )
        first = actions[donor]
        expected_seed = _policy_seed(fixed.config_hash, 0, offset)
        if int(first["row"]["action_seed"]) != expected_seed:
            raise _refuse("FIRST_H4_CONTRACT_MISMATCH", "first action seed differs")
        try:
            with torch.no_grad():
                encoded = donor_system.observation_adapter.encode(
                    observation,
                    context["pointcloud_meta_history"],
                    episode_identity_hash=context["episode_identity_hash"],
                    decision_index=context["decision_index"],
                )
                donor_system.observation_adapter.verify(encoded)
        except (ObservationIdentityMismatch, PointCloudRejected) as exc:
            raise _refuse("SNAPSHOT_REPLAY_MISMATCH", "initial observation rejected {} {}: {}".format(entry["task_id"], arm, exc), EXIT_REPLAY) from exc
        _assert_first_input(entry["task_id"], arm, first["row"], encoded.provenance)
        transitions = []
        decision = 0
        while True:
            before = np.asarray(adapter.environment._drone_pos, dtype=np.float32).copy()
            goal = np.asarray(adapter.environment.goal_w, dtype=np.float64)
            if decision == 0:
                action = _action_bundle(first["action"])
                actor_lane = "frozen_" + donor
                provenance = encoded.provenance
            else:
                try:
                    action = policy.predict(
                        observation, context,
                        action_seed=_policy_seed(fixed.config_hash, decision, offset),
                    )
                except (ObservationIdentityMismatch, PointCloudRejected) as exc:
                    transitions.append({
                        "task_id": entry["task_id"], "repeat": int(entry["repeat"]),
                        "outcome_class": entry["outcome_class"], "role": entry["role"],
                        "arm": arm, "decision_index": decision, "actor_lane": follow,
                        "action_seed": _policy_seed(fixed.config_hash, decision, offset),
                        "action_sha256": None, "position_before_xyz": before.tolist(),
                        "position_after_xyz": before.tolist(), "goal_xyz": goal.tolist(),
                        "goal_distance_before_3d": float(np.linalg.norm(goal - before)),
                        "goal_distance_after_3d": float(np.linalg.norm(goal - before)),
                        "static_clearance_m": None, "executed_h4_path_5x3": None,
                        "reward": None, "terminated": True, "truncated": False,
                        "policy_input_hash": None, "raw_frame_hash": None,
                        "normalized_history_hash": None,
                        "observation_rejection": {"type": type(exc).__name__, "detail": str(exc)},
                    })
                    after = before
                    reason = "invalid_state"
                    decision += 1
                    break
                actor_lane = follow
                provenance = policy.last_provenance
            observation, reward, terminated, truncated, info = adapter.step(action)
            context = info
            path = np.asarray(info.get("executed_path"), dtype=np.float32)
            if path.shape != (5, 3) or not np.isfinite(path).all():
                raise _refuse("EXECUTED_H4_CONTRACT_MISMATCH", "missing executed H4 path")
            if decision == 0 and not np.array_equal(path.view(np.uint32), first["path"].view(np.uint32)):
                raise _refuse("SNAPSHOT_REPLAY_MISMATCH", "first executed H4 differs {} {}".format(entry["task_id"], arm), EXIT_REPLAY)
            if baseline_rows is not None:
                try:
                    _verify_baseline_path(path, baseline_rows, decision)
                except ValueError as exc:
                    raise _refuse("BASELINE_REPLAY_MISMATCH", "{} r{} {}: {}".format(entry["task_id"], entry["repeat"], arm, exc), EXIT_REPLAY) from exc
            after = np.asarray(adapter.environment._drone_pos, dtype=np.float32).copy()
            clearance = info.get("reward_metrics", {}).get("d_min")
            clearance = None if clearance is None or not np.isfinite(float(clearance)) else float(clearance)
            transitions.append({
                "task_id": entry["task_id"], "repeat": int(entry["repeat"]),
                "outcome_class": entry["outcome_class"], "role": entry["role"],
                "arm": arm, "decision_index": decision, "actor_lane": actor_lane,
                "action_seed": _policy_seed(fixed.config_hash, decision, offset),
                "policy_input_hash": getattr(provenance, "policy_input_hash", None),
                "raw_frame_hash": getattr(provenance, "raw_frame_hash", None),
                "normalized_history_hash": getattr(provenance, "normalized_history_hash", None),
                "action_sha256": canonical_array_sha256(_executed_numpy(action)),
                "position_before_xyz": before.tolist(), "position_after_xyz": after.tolist(),
                "goal_xyz": goal.tolist(),
                "goal_distance_before_3d": float(np.linalg.norm(goal - before)),
                "goal_distance_after_3d": float(np.linalg.norm(goal - after)),
                "static_clearance_m": clearance,
                "executed_h4_path_5x3": path.tolist(), "reward": float(reward),
                "terminated": bool(terminated), "truncated": bool(truncated),
            })
            decision += 1
            if terminated or truncated:
                facts = info.get("transition_facts")
                raw_reason = getattr(facts, "raw_done_reason", None) or getattr(adapter.environment, "_done_reason", None)
                reason = canonical_done_reason(raw_reason, after[2], terminated, truncated)
                break
            if decision >= int(config.environment.max_episode_steps):
                raise _refuse("EPISODE_BOUNDARY_MISMATCH", "episode did not terminate at configured limit", EXIT_REPLAY)
        goal = np.asarray(adapter.environment.goal_w, dtype=np.float64)
        return {
            "done_reason": reason,
            "decision_count": decision,
            "final_distance_3d": float(np.linalg.norm(goal - after)),
            "terminal_xyz": after.tolist(),
            "transitions": transitions,
        }
    finally:
        adapter.environment.close()


def _run(args: argparse.Namespace) -> int:
    selection_dir = Path(args.selection_dir)
    verify_sha256_manifest(selection_dir)
    selected = load_strict_json(selection_dir / "selection.json")
    if load_strict_json(selection_dir / "diagnostic_source_manifest.json") != _source_manifest():
        raise _refuse("STAGE_INPUT_HASH_MISMATCH", "selection source differs from running source")
    if (
        selected["schema"] != "routea_first_h4_swap_selection_v1"
        or selected["pair_count"] != 8
        or selected["task_count"] != 16
        or selected["arm_count"] != 64
    ):
        raise _refuse("SELECTION_CONTRACT_MISMATCH", "invalid selection")
    parent = _validated_parent(Path(args.capture_dir), Path(args.final_dir))
    if selected["parent_identity"] != parent["parent_identity"]:
        raise _refuse("STAGE_INPUT_IDENTITY_MISMATCH", "selection parent changed")
    first = _read_first_rows(Path(args.capture_dir) / "paired_trace_transitions.jsonl")
    selected_keys = {
        (str(entry["task_id"]), int(entry["repeat"]))
        for pair in selected["pairs"] for entry in (pair["break"], pair["rescue"])
    }
    episodes_by_lane = _read_episode_rows(
        Path(args.capture_dir) / "paired_trace_transitions.jsonl", selected_keys
    )
    expected_lanes = {(task, repeat, lane) for task, repeat in selected_keys for lane in ("u0", "seed3-u32")}
    if set(episodes_by_lane) != expected_lanes:
        raise _refuse("STAGE_INPUT_IDENTITY_MISMATCH", "selected baseline transition lanes are incomplete")
    from scripts.routea_causal_diagnostics.runtime import (
        install_read_only_tripwires, load_read_only_system, validate_zero_effect_lora_state,
    )
    u0_config, u0_system = load_read_only_system(args.u0_config, args.u0_checkpoint, REPO_ROOT)
    rl_config, rl_system = load_read_only_system(args.rl_config, args.rl_checkpoint, REPO_ROOT)
    validate_zero_effect_lora_state(u0_system.lora_bundle.adapter_state_dict())
    from omegaconf import OmegaConf
    if (
        OmegaConf.to_container(u0_config.environment, resolve=True) != OmegaConf.to_container(rl_config.environment, resolve=True)
        or OmegaConf.to_container(u0_config.reward, resolve=True) != OmegaConf.to_container(rl_config.reward, resolve=True)
    ):
        raise _refuse("CHECKPOINT_CONTRACT_MISMATCH", "u0/seed3 environment or reward differs", EXIT_CHECKPOINT)
    systems = {"u0": (u0_config, u0_system), "seed3-u32": (rl_config, rl_system)}
    immutable = [
        selection_dir / "selection.json", selection_dir / "manifest_sha256.txt",
        Path(args.capture_dir) / "paired_trace_transitions.jsonl",
        Path(args.final_dir) / "d05_verdict.json",
        Path(args.u0_config), Path(args.rl_config),
        Path(args.u0_checkpoint) / "header.json", Path(args.u0_checkpoint) / "SHA256.json",
        Path(args.rl_checkpoint) / "header.json", Path(args.rl_checkpoint) / "SHA256.json",
    ]
    immutable.extend(sorted({
        Path(entry["geometry_path"])
        for pair in selected["pairs"] for entry in (pair["break"], pair["rescue"])
    }, key=str))
    outer_rng = capture_rng_state()
    guards = {lane: install_read_only_tripwires(system, immutable_files=immutable) for lane, (_, system) in systems.items()}
    episodes = []
    try:
        for pair in selected["pairs"]:
            for entry in (pair["break"], pair["rescue"]):
                key = (str(entry["task_id"]), int(entry["repeat"]))
                actions = first_actions(first[key], *key)

                def run_arm(e, arm, donor, follow):
                    baseline_rows = episodes_by_lane[(key[0], key[1], donor)] if arm in ("bc_bc", "rl_rl") else None
                    return _run_one_arm(e, actions, arm, donor, follow, systems, baseline_rows)

                try:
                    result = execute_four_arms(entry, run_arm)
                except ValueError as exc:
                    raise _refuse("BASELINE_REPLAY_MISMATCH", str(exc), EXIT_REPLAY) from exc
                episodes.append({"entry": entry, "arms": result})
                _emit({"status": "TASK_FIRST_H4_COMPLETE", "task_id": entry["task_id"], "repeat": entry["repeat"], "pattern": result["swap_pattern"]})
    finally:
        restore_rng_state(outer_rng)
    tripwire = {
        "status": "READ_ONLY_VERIFIED",
        "lanes": {lane: guard.verify(require_rng_unchanged=True) for lane, guard in guards.items()},
    }
    model_identity = {
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
    }
    summary = {"schema": "routea_first_h4_swap_summary_v1", "evidence_scope": "lightweight_kinematic_diagnostic_intervention", "training_authorized": False, "reward_change_authorized": False, "parent_identity": selected["parent_identity"], "model_identity": model_identity, "task_count": len(episodes), "arm_count": 4 * len(episodes), "task_results": []}
    transitions = []
    for episode in episodes:
        entry, arms = episode["entry"], episode["arms"]
        summary["task_results"].append({
            "task_id": entry["task_id"], "repeat": entry["repeat"], "role": entry["role"],
            "outcome_class": entry["outcome_class"], "swap_pattern": arms["swap_pattern"],
            "arms": {arm: {key: value for key, value in arms[arm].items() if key != "transitions"} for arm in ARM_SPECS},
        })
        for arm in ARM_SPECS:
            transitions.extend(arms[arm]["transitions"])
    summary["pattern_counts"] = {
        outcome: {pattern: sum(row["outcome_class"] == outcome and row["swap_pattern"] == pattern for row in summary["task_results"]) for pattern in ("FIRST_H4_DOMINANT", "DOWNSTREAM_POLICY_DOMINANT", "MIXED")}
        for outcome in ("break", "rescue")
    }
    summary["pattern_counts_by_role"] = {
        role: {
            outcome: {
                pattern: sum(
                    row["role"] == role and row["outcome_class"] == outcome
                    and row["swap_pattern"] == pattern
                    for row in summary["task_results"]
                )
                for pattern in ("FIRST_H4_DOMINANT", "DOWNSTREAM_POLICY_DOMINANT", "MIXED")
            }
            for outcome in ("break", "rescue")
        }
        for role in ("anticipatory", "dense_recovery")
    }
    with AtomicOutputDirectory(args.output_dir) as staging:
        write_strict_json(staging / "summary.json", summary)
        write_strict_json(staging / "tripwire_verification.json", tripwire)
        write_jsonl(staging / "arm_transitions.jsonl", transitions)
        publish_stage_manifest(staging, _source_manifest())
    _emit({"status": "RUN_FIRST_H4_COMPLETE", "tasks": 16, "arms": 64, "patterns": summary["pattern_counts"]})
    return 0


def _validate(args: argparse.Namespace) -> int:
    verified = verify_sha256_manifest(args.output_dir)
    if not {"summary.json", "tripwire_verification.json", "arm_transitions.jsonl"} <= set(verified):
        raise _refuse("OUTPUT_CONTRACT_MISMATCH", "run manifest incomplete")
    summary = load_strict_json(Path(args.output_dir) / "summary.json")
    tripwire = load_strict_json(Path(args.output_dir) / "tripwire_verification.json")
    if summary["task_count"] != 16 or summary["arm_count"] != 64 or tripwire["status"] != "READ_ONLY_VERIFIED":
        raise _refuse("OUTPUT_CONTRACT_MISMATCH", "task, arm, or tripwire count differs")
    groups: Dict[tuple, list] = {}
    with (Path(args.output_dir) / "arm_transitions.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["task_id"], int(row["repeat"]), row["arm"])
            groups.setdefault(key, []).append(row)
    if len(groups) != 64:
        raise _refuse("OUTPUT_CONTRACT_MISMATCH", "transition arm count differs")
    for task in summary["task_results"]:
        for arm, (donor, follow) in ARM_SPECS.items():
            key = (task["task_id"], int(task["repeat"]), arm)
            rows = groups.get(key, [])
            if (
                len(rows) != int(task["arms"][arm]["decision_count"])
                or [row["decision_index"] for row in rows] != list(range(len(rows)))
                or not rows
                or rows[0]["actor_lane"] != "frozen_" + donor
                or any(row["actor_lane"] != follow for row in rows[1:])
            ):
                raise _refuse("OUTPUT_CONTRACT_MISMATCH", "arm transition sequence differs: {}".format(key))
    _emit({"status": "VALIDATE_FIRST_H4_COMPLETE", "tasks": 16, "arms": 64, "manifest_files": len(verified)})
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--capture-dir", type=Path, required=True)
    select.add_argument("--final-dir", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--selection-dir", type=Path, required=True)
    run.add_argument("--capture-dir", type=Path, required=True)
    run.add_argument("--final-dir", type=Path, required=True)
    run.add_argument("--u0-config", type=Path, required=True)
    run.add_argument("--u0-checkpoint", type=Path, required=True)
    run.add_argument("--rl-config", type=Path, required=True)
    run.add_argument("--rl-checkpoint", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    validate = sub.add_parser("validate-output")
    validate.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = _parser().parse_args(argv)
        return {"select": _select, "run": _run, "validate-output": _validate}[args.command](args)
    except DiagnosticFailure as exc:
        _emit({"status": exc.status, "detail": exc.detail, "exit_code": exc.exit_code})
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
