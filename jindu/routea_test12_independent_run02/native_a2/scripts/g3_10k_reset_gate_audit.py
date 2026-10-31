#!/usr/bin/env python3
"""G3 gate (design §5.5): 10k-reset admission-path audit — CLI entry.

Runs the REAL training admission path — RoleSampler -> task spec ->
UAVVectorEnvAdapter.reset — over n_resets (10,000 in production) and
writes G3_GATE_REPORT.json with the 15 gate verdicts (8 §5.5 gates +
R2-rework admission_blocked / manifest_hash_mismatch /
dense_path_stretch_fail + R7-5 invalid_window / source_hash_missing /
source_hash_mismatch / runtime_hash_mismatch) plus the per-role reset
counts/proportions/deltas. Deterministic for a fixed seed.

Production launch (asserts the role-proportion gate, which is only
statistically meaningful at the 10k scale — ~5.9pp std at n=60):

    python scripts/g3_10k_reset_gate_audit.py \\
        --manifest <admission_manifest.json> \\
        --worlds-dir <worlds/train> \\
        --n-resets 10000 --seed 0 \\
        --out <...>/G3_GATE_REPORT.json \\
        --assert-role-proportion

Exit code: 0 iff every asserted gate PASSes.

The report self-binds its run origin (R2 closure rework #2, 审核裁决
P2): tree_hash / command / exit_code / manifest_sha256 /
world_inputs_hash, plus the honest frame accounting (R7-5):
validated_reset_frames = n_resets; observation_window_rows = n_frames =
resets * obs_steps — every window frame is independently validated
through the window gate. Any provenance failure hard-fails: no report is
written.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.gate_audit import run_reset_gate_audit
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import manifest_record_key
from diffusion_policy_3d.rl.dppo_uav.integration import (
    build_adapter,
    build_routea_train_pool,
    build_task_specs,
    entry_metadata_hash,
    manifest_train_entries,
)

GATE_NAMES = (
    "invalid_pc",
    "unvalidated_fallback",
    "reachability",
    "clearance",
    "role_unknown",
    "role_proportion",
    "eval_leakage",
    "hash_missing",
    # R2 rework (审核裁决 P1-1): real gates — counted but never gated
    # before, or masked entirely.
    "admission_blocked",
    "manifest_hash_mismatch",
    "dense_path_stretch_fail",
    # R7-5 window gates: the full obs_steps window per reset + dual-hash
    # independence (source from the manifest entry, runtime from the
    # env's actual post-reset state).
    "invalid_window",
    "source_hash_missing",
    "source_hash_mismatch",
    "runtime_hash_mismatch",
)

PRODUCTION_RESETS = 10000

# Production defaults (R2.6): the CLI runs with ONLY --n-resets/--seed/
# --output (guide Step 2); --manifest/--worlds-dir fall back to the
# DPPO_MANIFEST / DPPO_WORLDS_TRAIN env-var convention (same as
# tests/dppo_parity/conftest.py), then to the known production paths.
DEFAULT_MANIFEST = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration"
    "/admission_manifest.json"
)
DEFAULT_WORLDS_TRAIN = (
    "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train"
)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tree_hash() -> str:
    """The exact git tree the evidence was produced from (self-binding,
    R2 closure rework #2 — 审核裁决 P2).

    Hard-fails (raise) instead of writing a placeholder: a report whose
    tree hash cannot be resolved must never be sealed as evidence.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"], capture_output=True,
            text=True, cwd=_REPO_ROOT,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        # rework #4 (审核裁决): narrowed — a git that runs but exits
        # nonzero is caught by returncode above; only launch-level
        # failures land here.
        pass
    raise RuntimeError(
        "provenance: cannot resolve git tree hash (not a git repo?); "
        "refusing to write a report without self-binding")


def _sha256_file(path: str) -> str:
    """SHA256 of the manifest file — the report self-binds its input."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _world_inputs_hash(paths: List[str]) -> str:
    """SHA256 over the sorted RESOLVED world obstacle-input set (审核裁决 P2).

    The input set is exactly the files the pool resolved to — never a glob
    of the whole directory (a glob also covers rl_val / rl_test /
    unadmitted worlds and would fingerprint inputs the pool never loads;
    the pre-rework glob hashed 180 files for a 21-world pool). An empty or
    unreadable set hard-fails — never a placeholder.
    """
    paths = sorted(paths)
    if not paths:
        raise RuntimeError(
            "provenance: empty resolved world input set; refusing to write"
            " a report without input-set fingerprint")
    h = hashlib.sha256()
    for p in paths:
        h.update(p.encode("utf-8"))
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def source_hash_records(entries: List[Dict[str, Any]]) -> Dict[str, str]:
    """Composite-key -> SOURCE-entry hash, recomputed independently of the
    spec construction path (R2 rework P1-1).

    Mirrors TaskSpecRegistry's exact derivation (role from the resolved
    entry field, default admission_id f"{world_uid}__{role}__{split}") but
    computes the hash from the manifest ENTRY, not from the spec — any
    divergence in build_task_specs (dropped/renamed hash, role or
    admission_id derivation drift) now surfaces as a manifest_hash_mismatch
    gate FAIL instead of being masked by a self-copy.
    """
    return {
        manifest_record_key(
            e["world_uid"],
            e["role"],
            e.get("admission_id", f"{e['world_uid']}__{e['role']}__{e['split']}"),
        ): entry_metadata_hash(e)
        for e in entries
    }


def gates_ok(metrics: Dict[str, Any], assert_role_proportion: bool) -> bool:
    """True iff every asserted gate PASSes.

    role_proportion is a 10k-scale gate (§5.5): a FAIL blocks only when
    ``assert_role_proportion`` is set. All other gates always block.
    """
    for name in GATE_NAMES:
        if metrics["gates"][name] != "PASS":
            if name == "role_proportion" and not assert_role_proportion:
                continue
            return False
    return True


def run(
    *,
    manifest: Optional[str] = None,
    worlds_dir: Optional[str] = None,
    n_resets: int,
    seed: int,
    out: str,
    device: str = "cpu",
    assert_role_proportion: bool = False,
) -> int:
    """Run the gate audit and write the report; returns the exit code.

    Raises:
        FileNotFoundError: admission manifest missing.
        ValueError: assert_role_proportion below the production scale.
    """
    if manifest is None:
        manifest = os.environ.get("DPPO_MANIFEST", DEFAULT_MANIFEST)
    if worlds_dir is None:
        worlds_dir = os.environ.get("DPPO_WORLDS_TRAIN", DEFAULT_WORLDS_TRAIN)
    if assert_role_proportion and n_resets < PRODUCTION_RESETS:
        raise ValueError(
            f"--assert-role-proportion requires the production scale"
            f" (>= {PRODUCTION_RESETS} resets; n=60 std ~5.9pp on a 0.30 target)"
        )

    entries = manifest_train_entries(manifest)
    if not entries:
        raise ValueError("no admitted rl_train entries in the admission manifest")

    # rework #3 (审核裁决 P2): the worlds input set is the RESOLVED pool —
    # the exact files the adapter loads — plus the honest world counts
    # (resolved vs. manifest entries). A glob of worlds_dir would cover
    # rl_val/rl_test/unadmitted worlds the pool never loads.
    pool = build_routea_train_pool(manifest, worlds_dir)
    if not pool:
        raise RuntimeError(
            f"provenance: resolved pool is empty under {worlds_dir}; "
            "refusing to write a report without an input-set fingerprint")
    resolved_world_count = len(pool)

    specs, _ = build_task_specs(
        [dict(e, manifest_entry_hash=entry_metadata_hash(e)) for e in entries]
    )
    # R2 rework: recomputed from the SOURCE entries (not copied from specs)
    # — feeds both the adapter's rule-6 records and the audit's
    # expected_hashes, so a spec/hash divergence is a gate FAIL, never a
    # masked self-match.
    records = source_hash_records(entries)
    adapter = build_adapter(
        RLSystemContract(),
        worlds_dir=worlds_dir,
        manifest_path=manifest,
        manifest_records=records,
        device=device,
        max_episode_steps=100,
    )
    metrics = run_reset_gate_audit(
        adapter,
        specs,
        n_resets=n_resets,
        seed=seed,
        # R2 rework: recomputed source-entry hashes — a spec whose hash
        # diverges from the manifest is a gate FAIL, not a masked blocked.
        expected_hashes=records,
    )
    metrics["role_proportion_asserted"] = assert_role_proportion

    ok = gates_ok(metrics, assert_role_proportion)
    # R2 closure rework #2 (审核裁决 P2): the report self-binds its run
    # origin — tree hash / command / exit code / manifest sha256 / worlds
    # input-set hash — and spells out the frame accounting (R7-5): each
    # reset validates the FULL obs_steps window (validated_reset_frames =
    # n_resets windows); every window row (n_frames = resets * obs_steps)
    # is independently validated through the window gate.
    report = dict(metrics)
    report["provenance"] = {
        "tree_hash": _tree_hash(),
        "command": " ".join(sys.argv),
        "exit_code": 0 if ok else 1,
        "manifest_sha256": _sha256_file(manifest),
        "world_inputs_hash": _world_inputs_hash([p for _, p in pool]),
        "resolved_world_count": resolved_world_count,
        "manifest_world_count": len(entries),
        "validated_reset_frames": metrics["n_resets"],
        "observation_window_rows": metrics["n_frames"],
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"G3 gate audit: {n_resets} resets -> {out}")
    for gate, verdict in metrics["gates"].items():
        print(f"  {gate:24s} {verdict}")
    print("  role_proportion_asserted:", assert_role_proportion)
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="G3 gate (design §5.5): 10k-reset admission-path audit"
    )
    parser.add_argument("--manifest", default=None,
                        help="RouteA RL v1 admission manifest JSON "
                             "(default: $DPPO_MANIFEST or the production path)")
    parser.add_argument("--worlds-dir", default=None,
                        help="train worlds dir with <uid>.world.obstacles.json "
                             "(default: $DPPO_WORLDS_TRAIN or the production path)")
    parser.add_argument("--n-resets", type=int, default=PRODUCTION_RESETS,
                        help=f"resets to audit (default {PRODUCTION_RESETS})")
    parser.add_argument("--seed", type=int, default=0, help="deterministic seed")
    parser.add_argument("--out", default="G3_GATE_REPORT.json",
                        help="report output path")
    parser.add_argument("--device", default="cpu", help="torch device")
    parser.add_argument("--assert-role-proportion", action="store_true",
                        help="fail the run if the role-proportion gate FAILs "
                             "(10k production scale only)")
    args = parser.parse_args(argv)
    return run(
        manifest=args.manifest,
        worlds_dir=args.worlds_dir,
        n_resets=args.n_resets,
        seed=args.seed,
        out=args.out,
        device=args.device,
        assert_role_proportion=args.assert_role_proportion,
    )


if __name__ == "__main__":
    sys.exit(main())
