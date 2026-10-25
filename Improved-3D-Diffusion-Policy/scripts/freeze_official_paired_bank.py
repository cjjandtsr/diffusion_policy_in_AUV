"""freeze_official_paired_bank.py — freeze the G7/G8/G9 paired bank +
offline density on the production eval pool.

The bank is frozen ONCE and reused by every G7-G9 milestone (design
§5.10: "每个里程碑使用同一冻结 paired bank"). Composition (pre-
registered in the G7 preregistration):

- eval pool = production admission manifest rl_train ∪ rl_eval entries
  (R3 EVAL-AUDIT pool semantics — eval-only worlds reachable, never
  trained on);
- specs = the first 2 specs per capability role in manifest order
  (anticipatory / dense_recovery / support_expansion), each frozen with
  2 seeds (seeds 1..12) → 12 entries;
- entry content = the ACTUAL sampled start/goal/yaw + episode_config_hash
  (deterministic given (spec, seed) — the freeze contract);
- bank provenance = manifest file sha256 (source_hash) + creation commit;
  bank_id = canonical sha256 over the full content (any byte change
  breaks the binding).

The offline density is a FROZEN SYNTHETIC 6-cell placeholder — valid for
G7/G8 artifact audit (capability groups per episode are computed from
traces, not from the density), but G9's full-space support-expansion
claims require a REAL offline visit-count field (Guide R4.6: without it
G9 support claims are BLOCKED, never fabricated).

Run:
  python scripts/freeze_official_paired_bank.py \
      --bank-out /home/cjj/AUV/program/reports/routeA_rl_env_v1/\
g7_microtrain/paired_bank.json \
      --density-out /home/cjj/AUV/program/reports/routeA_rl_env_v1/\
g7_microtrain/offline_density.json
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy_3d.rl.dppo_uav.capability_metrics import OfflineDensity
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    manifest_record_key,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import (
    freeze_paired_bank,
    paired_bank_to_dict,
)
from diffusion_policy_3d.rl.dppo_uav.integration import (
    build_adapter,
    build_routea_eval_pool,
    build_task_specs,
    entry_metadata_hash,
    manifest_eval_entries,
)

DEFAULT_MANIFEST = (
    "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration"
    "/admission_manifest.json"
)
DEFAULT_WORLDS = "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train"

CAPABILITY_ROLES = (
    "anticipatory", "dense_recovery", "support_expansion",
)
SPECS_PER_ROLE = 2
SEEDS_PER_SPEC = 2


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _bank_specs(specs: List[Any]) -> List[Any]:
    """Deterministic bank selection: SPECS_PER_ROLE specs per capability
    role, manifest order (all three roles covered — the artifact's
    capability metrics hard-fail on a role with no record)."""
    by_role: Dict[str, List[Any]] = {}
    for s in specs:
        by_role.setdefault(s.role, []).append(s)
    chosen: List[Any] = []
    for role in CAPABILITY_ROLES:
        chosen.extend(by_role.get(role, [])[:SPECS_PER_ROLE])
    missing = [r for r in CAPABILITY_ROLES if r not in by_role]
    if missing:
        raise SystemExit(f"eval pool lacks capability roles: {missing}")
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--worlds", default=DEFAULT_WORLDS)
    parser.add_argument("--bank-out", required=True)
    parser.add_argument("--density-out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (frozen synth path is cuda:0)")

    contract = RLSystemContract()
    entries = manifest_eval_entries(args.manifest)
    specs_eval, _ = build_task_specs(
        [dict(e, manifest_entry_hash=entry_metadata_hash(e))
         for e in entries]
    )
    pool = build_routea_eval_pool(args.manifest, args.worlds)
    records_eval = {
        manifest_record_key(s.world_uid, s.role, s.admission_id):
            s.manifest_entry_hash
        for s in specs_eval
    }
    env_f = build_adapter(
        contract, worlds_dir=args.worlds, pool=pool,
        manifest_records=records_eval, device=args.device,
        max_episode_steps=100, n_envs=1,
    )
    bank_specs = _bank_specs(specs_eval)
    seeds = list(range(1, len(bank_specs) * SEEDS_PER_SPEC + 1))
    # (spec, seed) pairs: seed-major so both seeds of a spec stay adjacent
    bank_specs_expanded = [s for s in bank_specs for _ in range(SEEDS_PER_SPEC)]
    bank = freeze_paired_bank(env_f, bank_specs_expanded, seeds)
    bank_dict = paired_bank_to_dict(
        bank,
        source_hash=_sha256_file(Path(args.manifest)),
        creation_commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
    )
    bank_path = Path(args.bank_out)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    bank_path.write_text(json.dumps(bank_dict, indent=2), encoding="utf-8")

    density = OfflineDensity(counts={
        (0, 0, 0): 20, (1, 0, 0): 5, (0, 1, 0): 5, (0, 0, 1): 2,
        (1, 1, 1): 2, (2, 0, 0): 1,
    })
    density_path = Path(args.density_out)
    density_path.write_text(
        json.dumps(density.to_dict(), indent=2), encoding="utf-8"
    )

    print(f"bank_id={bank_dict['bank_id']}")
    print(f"bank_entries={len(bank)} specs={len(bank_specs)}")
    print(f"bank_file_sha256={_sha256_file(bank_path)}")
    print(f"density_file_sha256={_sha256_file(density_path)}")
    print(f"bank_out={bank_path}")
    print(f"density_out={density_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
