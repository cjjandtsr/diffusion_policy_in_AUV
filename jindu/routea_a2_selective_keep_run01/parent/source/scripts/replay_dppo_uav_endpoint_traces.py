#!/usr/bin/env python3
"""Replay a frozen diagnostic selection through the unchanged Endpoint runner.

This is policy inference, not training. Run in the original WSL project after
syncing only these new scripts. Every original summary must reproduce.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_dppo_uav_endpoint import load_evidence, read_json, sha256, write_json, select_replay_pairs
from scripts.endpoint_trajectory_trace import TraceAdapter, compare_record
from scripts.evaluate_dppo_uav_formal_endpoint import (
    _build_adapter, _E5Policy, _evaluate_one, _load_fixed_bank, _rl_policy_identity,
    _validate_bank_sources,
)


def validate_selection(selection, bank, source):
    if selection.get("bank_content_hash") != bank.content_hash:
        raise ValueError("selection source bank mismatch")
    for name in ("paired_episodes.jsonl", "endpoint_report.json", "evaluation_provenance.json"):
        if selection.get("source_sha256", {}).get(name) != sha256(Path(source) / name):
            raise ValueError("selection artifact hash mismatch: " + name)
    expected = select_replay_pairs(bank)
    if selection.get("pairs") != expected or selection.get("selection_count") != len(expected):
        raise ValueError("selection changed from the deterministic complete rule")
    return {row["task_id"] for row in expected}


def bound_file(path, expected):
    path = Path(path)
    if sha256(path) != expected:
        raise ValueError("source asset SHA256 mismatch: " + str(path))
    return path


def run(args):
    bank, _, provenance = load_evidence(args.source_dir)
    selected = validate_selection(read_json(args.selection), bank, args.source_dir)
    # Pin original evaluation code. No patch to identity-bearing production files.
    for relative, digest in provenance["evaluation_source_sha256"].items():
        path = (ROOT / relative).resolve()
        if ROOT.resolve() not in path.parents:
            raise ValueError("evaluation source path escapes repository")
        bound_file(path, digest)
    config_path = bound_file(provenance["config_path"], provenance["config_sha256"])
    manifest = bound_file(provenance["source_manifest_path"], provenance["source_manifest_sha256"])
    config_dirs = {bound_file(entry["path"], entry["sha256"]).parent for entry in provenance["fixed_config_files"].values()}
    if len(config_dirs) != 1:
        raise ValueError("fixed episode banks must share their original directory")
    fixed_bank = _load_fixed_bank(config_dirs.pop(), "fixed32")
    from diffusion_policy_3d.rl.dppo_uav.integration import manifest_eval_entries
    _validate_bank_sources(fixed_bank, manifest_eval_entries(str(manifest)))
    lookup = {bc.task_id: (bc, rl) for bc, rl in bank.pairs}
    for split, world_id, fixed in fixed_bank:
        bc, _ = lookup[str(fixed.payload["episode_id"])]
        if (bc.config_hash != fixed.config_hash or bc.seed != fixed.seed
                or (bc.split, bc.world_id) != (split, world_id)):
            raise ValueError("fixed episode content no longer matches original record")
        source = provenance["obstacle_sources"][world_id]
        if source["split"] != split or source["sha256"] != bc.obstacle_hash:
            raise ValueError("obstacle provenance does not match paired record")
        bound_file(source["path"], source["sha256"])
    checkpoint = Path(provenance["checkpoint_dir"])
    if _rl_policy_identity(checkpoint) != bank.pairs[0][1].policy_identity_hash:
        raise ValueError("checkpoint header/integrity identity changed")
    output = Path(args.output_dir).resolve()
    if output == Path(args.source_dir).resolve():
        raise ValueError("diagnostic output must not replace source endpoint")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "replay_request.json", dict(
        status="prepared_not_completed", source_provenance=provenance,
        selection=read_json(args.selection), selected_pair_count=len(selected),
        numeric_comparison=dict(rtol=1e-6, atol=1e-6),
        scope="diagnostic_subset_not_formal_endpoint", latency_measurement_valid=False,
        observer="temporary Python return profiler; original synthesize called once; no RNG draws",
        diagnostic_source_sha256={name: sha256(ROOT / "scripts" / name) for name in
            ("diagnose_dppo_uav_endpoint.py", "endpoint_trajectory_trace.py", "replay_dppo_uav_endpoint_traces.py")}))
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.build_system import build_system
    from diffusion_policy_3d.rl.dppo_uav_rebuild.deployment.loader import load_deployment_policy
    config = OmegaConf.load(config_path)
    bc_system = build_system(config, ROOT)
    deployment = load_deployment_policy(config, checkpoint, ROOT)
    if asdict(deployment.checkpoint_header) != provenance["checkpoint_header"]:
        raise ValueError("runtime checkpoint header differs from original provenance")
    if deployment.checkpoint_header.update_index != 256 or getattr(bc_system.reward, "version", None) != "v17":
        raise ValueError("requires final update256 and v17")
    policies = {
        "bc": _E5Policy(bc_system, use_base_policy=True, identity_hash=bc_system.bc_runtime.identity.content_hash),
        "rl": _E5Policy(deployment.system, use_base_policy=False, identity_hash=_rl_policy_identity(checkpoint)),
    }
    if policies["bc"].identity_hash != bank.pairs[0][0].policy_identity_hash:
        raise ValueError("BC policy identity mismatch")
    checks = []
    stems = set()
    for split, world_id, fixed in fixed_bank:
        task_id = str(fixed.payload["episode_id"])
        if task_id not in selected:
            continue
        stem = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:20]
        if stem in stems:
            raise ValueError("diagnostic output task hash collision")
        stems.add(stem)
        obstacle = provenance["obstacle_sources"][world_id]
        # Copy geometry into the diagnostic pack for self-contained plots.
        geometry = read_json(obstacle["path"])
        task_path = output / (stem + ".task.json")
        write_json(task_path, dict(task_id=task_id, split=split, world_id=world_id,
            fixed_config=fixed.payload, obstacle_source=obstacle, geometry=geometry))
        for policy_name, expected in zip(("bc", "rl"), lookup[task_id]):
            policy = policies[policy_name]
            adapter = _build_adapter(policy.system, config, world_id, Path(obstacle["path"]), expected.role)
            trace_path = output / (stem + "." + policy_name + ".trace.jsonl")
            with trace_path.open("x", encoding="utf-8") as stream:
                traced = TraceAdapter(adapter, stream, dict(task_id=task_id, policy=policy_name,
                    config_hash=fixed.config_hash, policy_identity_hash=policy.identity_hash))
                try:
                    actual = _evaluate_one(policy_name=policy_name, policy=policy, adapter=traced,
                        fixed=fixed, split=split, world_id=world_id, obstacle_hash=expected.obstacle_hash,
                        source_manifest_hash=expected.source_manifest_hash)
                    differences = compare_record(expected, actual)
                    traced.emit("episode_end", summary=asdict(actual), reproduction_differences=differences)
                finally:
                    traced.close()
            checks.append(dict(task_id=task_id, policy=policy_name, trace_file=trace_path.name,
                               trace_sha256=sha256(trace_path), task_sha256=sha256(task_path),
                               matched=not differences, differences=differences,
                               expected=asdict(expected), actual=asdict(actual)))
            if differences:
                write_json(output / "replay_verification.json", dict(status="mismatch", checks=checks,
                           selected_pairs=len(selected), completed_episodes=len(checks)))
                print("Replay mismatch; preserved trace, stopped without changing physics or identity.", flush=True)
                return 2
        print("diagnostic pairs completed: {}/{} {}".format(len(checks) // 2, len(selected), task_id), flush=True)
    if len(checks) != 2 * len(selected):
        raise RuntimeError("selection was not fully replayed")
    write_json(output / "replay_verification.json", dict(status="matched", checks=checks,
               selected_pairs=len(selected), completed_episodes=len(checks)))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
