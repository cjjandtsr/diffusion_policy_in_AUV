#!/usr/bin/env python3
"""Unified no-training Route A v1.7 F0/G0/G1 admission runner."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.config_schema_v2 import validate_v2_config
from scripts.dppo_credit_support.credit_probe import assess_sampled_k_gate


GATE_NAMES = (
    "F0_SOURCE_IDENTITY", "F0_RUNTIME_IDENTITY",
    "G0_GOAL_TERMINAL_GEOMETRY", "G0_OBSERVATION_PROVENANCE",
    "G0_TASK_DISTRIBUTION", "G0_PAIR_SCHEDULE", "G0_LOGICAL_REDUCTION",
    "G0_GAE_TRANSACTION", "G0_CAPACITY_ACTION_CONTRACT", "G0_REWARD_ISOLATION",
    "G1_D1B_CREDIT", "G1_NO_UPDATE_TRIPWIRE",
)


@dataclass(frozen=True)
class GateRunResult:
    status: str
    selected_schedule: str | None
    gates: tuple
    blockers: tuple


_PREFLIGHT_TESTS = {
    "F0_SOURCE_IDENTITY": (
        "tests/dppo_uav_rebuild/entrypoints/test_source_closure_v2.py",
        "tests/dppo_uav_rebuild/entrypoints/test_config_schema_v2.py",
        "tests/dppo_uav_rebuild/entrypoints/test_gate_runner_v2.py",
        "tests/dppo_uav_rebuild/entrypoints/test_pipeline_integration_v2.py",
    ),
    "F0_RUNTIME_IDENTITY": ("tests/dppo_uav_rebuild/official_core/test_direct_import.py", "tests/dppo_uav_rebuild/official_core/test_e5_official_runtime.py"),
    "G0_GOAL_TERMINAL_GEOMETRY": ("tests/dppo_uav_rebuild/environments/test_terminal_contract_v2.py", "tests/dppo_uav_rebuild/environments/test_geometry_oracle_v2.py", "tests/dppo_uav_rebuild/environments/test_reset_commit_v2.py"),
    "G0_OBSERVATION_PROVENANCE": (
        "tests/dppo_uav_rebuild/environments/test_pointcloud_provenance_v2.py",
        "tests/dppo_uav_rebuild/entrypoints/test_pointcloud_production_wiring_v2.py",
        "tests/dppo_uav_rebuild/entrypoints/test_pointcloud_calibration_v2.py",
    ),
    "G0_TASK_DISTRIBUTION": ("tests/dppo_uav_rebuild/environments/test_task_identity_v2.py", "tests/dppo_uav_rebuild/environments/test_role_sampler_v2.py", "tests/dppo_uav_rebuild/entrypoints/test_eval_only_admission_v2.py"),
    "G0_PAIR_SCHEDULE": ("tests/dppo_uav_rebuild/engine/test_pair_schedule_v2.py", "tests/dppo_uav_rebuild/engine/test_logical_batches_v2.py"),
    "G0_LOGICAL_REDUCTION": ("tests/dppo_uav_rebuild/engine/test_official_loss_components_v2.py", "tests/dppo_uav_rebuild/engine/test_logical_accumulation_v2.py", "tests/dppo_uav_rebuild/engine/test_value_clipping_reduction_v2.py"),
    "G0_GAE_TRANSACTION": (
        "tests/dppo_uav_rebuild/engine/test_buffer_and_gae.py",
        "tests/dppo_uav_rebuild/engine/test_update_transaction_v2.py",
        "tests/dppo_uav_rebuild/audit/test_training_audit.py",
    ),
    "G0_CAPACITY_ACTION_CONTRACT": ("tests/dppo_uav_rebuild/actor_ddpm/test_rollout_shapes.py", "tests/dppo_uav_rebuild/test_contracts_and_vendor.py"),
    "G0_REWARD_ISOLATION": ("tests/dppo_uav_rebuild/rewards/test_reward_v17.py", "tests/dppo_uav_rebuild/rewards/test_reward_v17_integration.py", "tests/dppo_uav_rebuild/rewards/test_reward_v17_calibration.py"),
}


def _classify_pytest_result(returncode, output):
    lowered = str(output).lower()
    forbidden = ("skipped", "xfailed", "xpassed")
    return "PASS" if int(returncode) == 0 and not any(word in lowered for word in forbidden) else "BLOCKED"


def _schedule_neutral_config(config):
    payload = OmegaConf.to_container(config, resolve=True)
    payload = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    payload.pop("output_dir", None)
    payload["run"].pop("schedule_label", None)
    payload["dppo"].pop("denoising_schedule", None)
    payload["dppo"].pop("logical_batch_pairs", None)
    payload["dppo"].pop("microbatch_pairs", None)
    return payload


def _assert_fallback_compatible(primary, fallback):
    if _schedule_neutral_config(primary) != _schedule_neutral_config(fallback):
        raise ValueError("fallback config differs outside registered schedule fields")


def execute_preflight(config, *, repo_root, log_dir):
    """Build the real WSL runtime and execute every registered no-training group."""

    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.build_system import build_system
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import build_runtime_identity
    from diffusion_policy_3d.rl.dppo_uav_rebuild.audit.training_audit import ParameterSnapshot
    from scripts.audit_checkpoint_source_identity import build_inventory
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.environment_manifest import build_environment_manifest
    validate_v2_config(config)
    root = Path(repo_root).resolve()
    logs = Path(log_dir)
    logs.mkdir(parents=True, exist_ok=False)
    system = build_system(config, root)
    snapshot = ParameterSnapshot.capture(
        actor=system.trainer.actor_parameters,
        critic=system.trainer.critic_parameters,
    )
    optimizer_state_sizes_before = (
        len(system.actor_optimizer.state), len(system.critic_optimizer.state)
    )
    manifest = Path(config.admission_manifest)
    if not manifest.is_absolute():
        manifest = root / manifest
    identity = build_runtime_identity(config, root, system, manifest.resolve())
    source_records, code_source_set_hash = build_inventory(root)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["DPPO_VENDOR_ROOT"] = str((root / config.vendor_root).resolve())
    gates = {}
    commands = {}
    for name, paths in _PREFLIGHT_TESTS.items():
        command = [sys.executable, "-m", "pytest", "-q", *paths]
        result = subprocess.run(command, cwd=root, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (logs / (name.lower() + ".log")).write_text(result.stdout, encoding="utf-8")
        gates[name] = _classify_pytest_result(result.returncode, result.stdout)
        commands[name] = {"argv": command, "exit_code": result.returncode}
    parameter_delta = snapshot.measure(
        actor=system.trainer.actor_parameters,
        critic=system.trainer.critic_parameters,
    )
    optimizer_state_sizes_after = (
        len(system.actor_optimizer.state), len(system.critic_optimizer.state)
    )
    if any(group["nonzero_elements"] for group in parameter_delta.values()) or optimizer_state_sizes_before != optimizer_state_sizes_after:
        raise RuntimeError("no-training preflight mutated model or optimizer state")
    return {
        "schema": "routea_v17_preflight_summary_v2",
        "gates": gates,
        "commands": commands,
        "runtime_identity_hash": identity.content_hash,
        "source_set_hash": identity.source_set_hash,
        "code_source_set_hash": code_source_set_hash,
        "code_source_file_count": len(source_records),
        "code_source_files": source_records,
        "runtime_identity": asdict(identity),
        "environment_manifest": build_environment_manifest(device=str(config.device), dtype="float32"),
        "training_enabled": False,
        "optimizer_step_count": 0,
        "parameter_delta": parameter_delta,
        "optimizer_state_sizes_before": optimizer_state_sizes_before,
        "optimizer_state_sizes_after": optimizer_state_sizes_after,
    }


def run_gates(config, *, preflight_summary=None, d1b_summary=None):
    contract = validate_v2_config(config)
    records = []
    for name in GATE_NAMES[:10]:
        passed = (
            isinstance(preflight_summary, dict)
            and preflight_summary.get("schema") == "routea_v17_preflight_summary_v2"
            and preflight_summary.get("gates", {}).get(name) == "PASS"
        )
        records.append({
            "name": name, "status": "PASS" if passed else "BLOCKED",
            "basis": "immutable_preflight_evidence" if passed else "PREFLIGHT_EVIDENCE_MISSING_OR_FAILED",
        })
    d1b_valid = isinstance(d1b_summary, dict) and d1b_summary.get("schema") == "routea_v17_d1b_summary_v2"
    s0_pass = d1b_valid and d1b_summary.get("s0", {}).get("decision") == "PASS"
    s2_pass = d1b_valid and d1b_summary.get("s2", {}).get("decision") == "PASS"
    requested = str(config.run.schedule_label)
    if requested == "S0" and s0_pass:
        selected = "S0"
    elif s2_pass:
        selected = "S2"
    else:
        selected = None
    records.append({
        "name": "G1_D1B_CREDIT", "status": "PASS" if selected else "BLOCKED",
        "basis": "immutable_d1b_summary" if d1b_valid else "D1B_EVIDENCE_MISSING",
    })
    records.append({
        "name": "G1_NO_UPDATE_TRIPWIRE",
        "status": "PASS" if d1b_valid and d1b_summary.get("tripwire", {}).get("all_unchanged") is True else "BLOCKED",
        "basis": "immutable_d1b_tripwire" if d1b_valid else "D1B_EVIDENCE_MISSING",
    })
    blockers = tuple(row["name"] for row in records if row["status"] != "PASS")
    return GateRunResult(
        status="PASS" if not blockers else "BLOCKED",
        selected_schedule=selected,
        gates=tuple(records), blockers=blockers,
    )


def load_and_recompute_d1b(directory):
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.io_safety import load_bounded_json, secure_open_under
    raw_root = Path(directory)
    if raw_root.is_symlink():
        raise ValueError("D1b evidence root symlink is forbidden")
    root = raw_root.resolve()
    summary = load_bounded_json(root, "d1b_summary.json", max_bytes=16 * 1024 * 1024)
    recorded_content_hash = summary.get("content_sha256")
    body = {key: value for key, value in summary.items() if key != "content_sha256"}
    actual_content_hash = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    if recorded_content_hash != actual_content_hash:
        raise ValueError("D1b summary content hash mismatch")
    manifest = summary.get("gradient_array_manifest")
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("D1b raw gradient array manifest is missing")
    import numpy as np
    reference = {}
    schedules_by_role = {}
    for role, entries in manifest.items():
        for key in ("full_joint", "sampled_k_100"):
            relative = Path(entries[key]["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("D1b array path traversal")
            with secure_open_under(root, relative, max_bytes=2 * 1024 * 1024 * 1024) as stream:
                digest_state = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest_state.update(block)
                digest = digest_state.hexdigest()
            if digest != entries[key]["sha256"]:
                raise ValueError("D1b gradient array hash mismatch")
        with secure_open_under(root, entries["full_joint"]["path"], max_bytes=2 * 1024 * 1024 * 1024) as stream:
            reference[role] = np.load(stream, allow_pickle=False)
        with secure_open_under(root, entries["sampled_k_100"]["path"], max_bytes=2 * 1024 * 1024 * 1024) as stream:
            schedules_by_role[role] = np.load(stream, allow_pickle=False)
        if schedules_by_role[role].shape[0] != 100:
            raise ValueError("D1b schedule array must have 100 rows")
        if reference[role].ndim != 1 or schedules_by_role[role].ndim != 2:
            raise ValueError("D1b gradients must be a vector and a schedule matrix")
        if schedules_by_role[role].shape[1:] != reference[role].shape:
            raise ValueError("D1b schedule/reference gradient shape mismatch")
        if reference[role].dtype.kind != "f" or schedules_by_role[role].dtype.kind != "f":
            raise ValueError("D1b gradients must use floating dtype")
        if not np.isfinite(reference[role]).all() or not np.isfinite(schedules_by_role[role]).all():
            raise ValueError("D1b gradients contain non-finite values")
    schedules = [
        {role: schedules_by_role[role][index] for role in sorted(reference)}
        for index in range(100)
    ]
    recomputed = assess_sampled_k_gate(reference, schedules)
    if recomputed != summary.get("s0"):
        raise ValueError("D1b summary differs from raw gradient recomputation")
    return summary


def _validate_d1b_identity(summary, *, config_path, config, preflight, repo_root):
    identity = summary.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("D1b identity is missing")
    checkpoint = Path(config.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = Path(repo_root) / checkpoint
    expected = {
        "policy_label": "bc",
        "checkpoint_update": 0,
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "base_bc_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "code_source_set_hash": preflight["code_source_set_hash"],
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError("D1b identity mismatch: {}".format(key))
    if not identity.get("dev_bank_id") or not identity.get("dev_bank_content_sha256"):
        raise ValueError("D1b dev-bank identity is missing")
    if int(identity.get("task_count", 0)) < 1:
        raise ValueError("D1b task count is invalid")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fallback-config", type=Path)
    parser.add_argument("--d1b-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = OmegaConf.load(args.config)
    validate_v2_config(config)
    fallback_config = None
    if str(config.run.schedule_label) == "S0" and not args.fallback_config:
        raise ValueError("S0 admission requires the registered S2 fallback config")
    if args.fallback_config:
        fallback_config = OmegaConf.load(args.fallback_config)
        fallback_contract = validate_v2_config(fallback_config)
        if str(fallback_config.run.schedule_label) != "S2":
            raise ValueError("fallback config must be the registered S2 schedule")
        _assert_fallback_compatible(config, fallback_config)
    preflight = execute_preflight(config, repo_root=Path(__file__).resolve().parents[1], log_dir=args.output.parent / (args.output.name + ".preflight_logs"))
    d1b = load_and_recompute_d1b(args.d1b_dir)
    _validate_d1b_identity(
        d1b,
        config_path=args.config,
        config=config,
        preflight=preflight,
        repo_root=Path(__file__).resolve().parents[1],
    )
    result = run_gates(config, preflight_summary=preflight, d1b_summary=d1b)
    selected_config_path = (
        args.fallback_config
        if result.selected_schedule == "S2" and str(config.run.schedule_label) == "S0"
        else args.config
    )
    payload = {
        "schema": "routea_v17_unified_gate_decision_v2",
        **asdict(result),
        "schedule_decision": result.selected_schedule or "BLOCKED_NO_VALID_SCHEDULE",
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "selected_config": str(selected_config_path) if result.selected_schedule else None,
        "selected_config_sha256": (
            hashlib.sha256(Path(selected_config_path).read_bytes()).hexdigest()
            if result.selected_schedule else None
        ),
        "training_enabled": False,
        "optimizer_step_count": 0,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "gate_decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "preflight_summary.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output / "gates.jsonl").open("x", encoding="utf-8") as stream:
        for gate in result.gates:
            stream.write(json.dumps(gate, sort_keys=True, allow_nan=False) + "\n")
    if d1b is not None:
        (args.output / "d1b_summary.json").write_text(
            json.dumps(d1b, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({
        "status": result.status,
        "selected_schedule": result.selected_schedule,
        "schedule_decision": payload["schedule_decision"],
    }, sort_keys=True))
    return 0 if result.status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
