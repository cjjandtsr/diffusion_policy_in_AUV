"""build_g8_capability_bank.py — freeze the R7-8 capability-informative
paired bank + REAL offline density on the eval-only splits (guide
Task R7-8, lines 1430-1507).

The bank freezes exactly 100 paired episode configs from rl_val ∪
rl_test (train leakage is hard-failed), each carrying a policy-
INDEPENDENT geometry opportunity (nominal-path / world-topology /
offline-density — never anything derived from an RL checkpoint, Step
5), a two-phase exposure (NOT_YET_MEASURED in v1; the R7-9 canonical
100-episode BC replay rebuilds v2, Step 4/8), and three independent
integrity hashes (membership / content / bank_id, Step 6).

The offline density is REAL (chunked read of the frozen EGO zarr —
user ruling 2026-08-24; R7-5 EXIT=137 lesson: never materialize the
4.8 GB dataset at once), never the old synthetic placeholder.

Run (freeze probes need CUDA):
  python scripts/build_g8_capability_bank.py \
      --manifest /home/cjj/AUV/program/reports/routeA_rl_env_v1/\
calibration/admission_manifest.json \
      --worlds-train /home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/\
worlds/train \
      --worlds-val /home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/\
worlds/val \
      --worlds-test /home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/\
worlds/test \
      --bc-ckpt /home/cjj/AUV/program/reports/routeA_rl_env_v1/\
<BC_CHECKPOINT>.ckpt \
      --dataset /home/cjj/AUV/avoidance_dataset/merged_body_12.zarr \
      --bank-out /home/cjj/AUV/program/evidence/08_paired_bank/\
paired_bank_r7.json \
      --density-out /home/cjj/AUV/program/evidence/08_paired_bank/\
offline_density_r7.json \
      --device cuda:0
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    manifest_record_key,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import (
    r7_bank_to_dict,
    r7_bank_validate,
)
from diffusion_policy_3d.rl.dppo_uav.integration import (
    build_adapter,
    build_routea_eval_pool,
    build_task_specs,
    entry_metadata_hash,
    manifest_eval_entries,
    manifest_train_entries,
)
from diffusion_policy_3d.rl.dppo_uav.r7_bank_builder import (
    build_density_from_zarr,
    check_train_leakage,
    compute_aggregate,
    expand_eval_candidates,
    freeze_and_annotate,
    reward13_density_to_capability,
    select_episodes,
)

DEFAULT_MANIFEST = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration"
    "/admission_manifest.json"
)
DEFAULT_WORLDS_TRAIN = (
    "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train"
)
DEFAULT_WORLDS_VAL = (
    "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/val"
)
DEFAULT_WORLDS_TEST = (
    "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/test"
)

# The builder's source surface the builder_source_hash binds (script +
# the modules whose code actually builds the bank). Fixed order.
_BUILDER_SOURCES = (
    "scripts/build_g8_capability_bank.py",
    "diffusion_policy_3d/rl/dppo_uav/r7_bank_builder.py",
    "diffusion_policy_3d/rl/dppo_uav/capability_metrics.py",
    "diffusion_policy_3d/rl/dppo_uav/evaluator.py",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def builder_source_hash() -> str:
    """sha256 over the concatenated builder source files (fixed order)
    — any builder-code change breaks the binding."""
    h = hashlib.sha256()
    for rel in _BUILDER_SOURCES:
        h.update((rel + "\0").encode("utf-8"))
        h.update((REPO_ROOT / rel).read_bytes())
    return h.hexdigest()


def _load_obstacles_by_uid(
    pool: Sequence[Tuple[str, str]],
) -> Dict[str, List[Dict[str, Any]]]:
    """world_uid -> obstacle records, from the pool's resolved paths."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for uid, json_path in pool:
        with open(json_path, "r", encoding="utf-8") as f:
            out[uid] = json.load(f)["obstacles"]
    return out


def _candidate_counts(
    frozen: Sequence[Any],
) -> Dict[str, int]:
    """Step-1 aggregate candidates: the geometry-opportunity counts
    over ALL pre-selection episodes (anticipatory / full_space /
    dense_recovery), from the frozen blocks — never from role names."""
    counts = {"anticipatory": 0, "full_space": 0, "dense_recovery": 0}
    for entry in frozen:
        op = getattr(entry, "opportunity", None)
        if op is None or not getattr(op, "valid", False):
            continue
        if getattr(op, "risk_shell_crossing", False):
            counts["anticipatory"] += 1
        # dense opportunity (user ruling 2026-08-24): trap ∪ cluttered.
        if (
            getattr(op, "trap_opportunity_count", 0) > 0
            or getattr(op, "topology_class", "") == "cluttered"
        ):
            counts["dense_recovery"] += 1
        if getattr(op, "support_stratum", ""):
            counts["full_space"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--worlds-train", default=DEFAULT_WORLDS_TRAIN)
    parser.add_argument("--worlds-val", default=DEFAULT_WORLDS_VAL)
    parser.add_argument("--worlds-test", default=DEFAULT_WORLDS_TEST)
    parser.add_argument("--bc-ckpt", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bank-out", required=True)
    parser.add_argument("--density-out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (100 freeze probes need cuda)")

    # ── Step 6: real offline density (chunked read, capability schema)
    density_raw = build_density_from_zarr(args.dataset)
    density = reward13_density_to_capability(density_raw)
    density_path = Path(args.density_out)
    density_path.parent.mkdir(parents=True, exist_ok=True)
    # A previous run froze this output read-only (chmod 444); rebuilds
    # unlock the file first, then re-freeze it — the artifact's final
    # state is always read-only, its content hash is re-recorded.
    if density_path.exists():
        density_path.chmod(0o644)
    density_path.write_text(
        json.dumps(density.to_dict(), indent=2), encoding="utf-8")
    density_path.chmod(0o444)
    print(f"[r7-bank] density cells={len(density.counts)} "
          f"tau_D={density.tau_D:.3f} -> {density_path}")

    # ── eval-only entry surface (rl_val ∪ rl_test reachable, Step 3)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    entries = manifest_eval_entries(args.manifest)
    _, specs_eval = build_task_specs(
        [dict(e, manifest_entry_hash=entry_metadata_hash(e))
         for e in entries]
    )
    pool = build_routea_eval_pool(
        args.manifest, args.worlds_train, args.worlds_val, args.worlds_test)
    pool_uids = {uid for uid, _ in pool}
    missing = sorted({s.world_uid for s in specs_eval} - pool_uids)
    if missing:
        raise SystemExit(
            "eval specs without an obstacle file in the pool: "
            f"{missing} — the bank cannot annotate phantom worlds"
        )
    obstacles = _load_obstacles_by_uid(pool)

    # ── dedicated single-slot env for the freeze probes
    contract = RLSystemContract()
    records_eval = {
        manifest_record_key(s.world_uid, s.role, s.admission_id):
            s.manifest_entry_hash
        for s in specs_eval
    }
    env_f = build_adapter(
        contract, worlds_dir=args.worlds_train, pool=pool,
        manifest_records=records_eval, device=args.device,
        max_episode_steps=100, n_envs=1,
    )

    # ── Step 2/5: expand candidates (manifest/geometry only, never any
    # RL checkpoint result), freeze the sampled configs, annotate.
    candidates = expand_eval_candidates(specs_eval)
    print(f"[r7-bank] candidates={len(candidates)} "
          f"({len(pool_uids)} eval worlds)")
    frozen = freeze_and_annotate(env_f, candidates, obstacles, density)

    # ── Step 2: exact 100 quota (roles >= 20 each, no RL conditioning)
    chosen = select_episodes(
        [(e.world_uid, e.role, e.seed) for e in frozen])
    chosen_set = set(chosen)
    bank_entries = [
        e for e in frozen
        if (e.world_uid, e.role, e.seed) in chosen_set
    ]
    if len(bank_entries) != 100:
        raise SystemExit(
            f"quota selection produced {len(bank_entries)} entries, "
            "expected exactly 100"
        )

    # ── Step 3: train leakage is a hard fail
    train_uids = {e["world_uid"] for e in manifest_train_entries(
        args.manifest)}
    check_train_leakage([e.world_uid for e in bank_entries], train_uids)

    # ── Step 1/6: aggregate + frozen artifact (three integrity hashes)
    aggregate = compute_aggregate(bank_entries, _candidate_counts(frozen))
    bank = r7_bank_to_dict(
        bank_entries,
        source_manifest_hash=_sha256_file(manifest_path),
        bc_checkpoint_hash=_sha256_file(Path(args.bc_ckpt)),
        builder_source_hash=builder_source_hash(),
        creation_commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
        aggregate=aggregate,
    )
    r7_bank_validate(bank, train_world_uids=train_uids)

    bank_path = Path(args.bank_out)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    if bank_path.exists():
        bank_path.chmod(0o644)
    bank_path.write_text(json.dumps(bank, indent=2), encoding="utf-8")
    bank_path.chmod(0o444)

    print(f"[r7-bank] bank_id={bank['bank_id']}")
    print(f"[r7-bank] bank_entries={len(bank_entries)} "
          f"bank_size={bank['bank_size']}")
    print(f"[r7-bank] opportunity={bank['aggregate']['opportunity']}")
    print(f"[r7-bank] exposure={bank['aggregate']['exposure']}")
    print(f"[r7-bank] membership_hash={bank['membership_hash']}")
    print(f"[r7-bank] content_hash={bank['content_hash']}")
    print(f"[r7-bank] manifest_sha256={bank['source_hash']}")
    print(f"[r7-bank] bc_ckpt_sha256={bank['bc_checkpoint_hash']}")
    print(f"[r7-bank] builder_source_sha256={bank['builder_source_hash']}")
    print(f"[r7-bank] bank_file_sha256={_sha256_file(bank_path)}")
    print(f"[r7-bank] density_file_sha256={_sha256_file(density_path)}")
    print(f"[r7-bank] bank_out={bank_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
