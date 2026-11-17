"""Preflight (run02, 指导 v1.1): decisive identity premise check.

v1.1 premise: (a) restored A1-era eval entry + current tree rebuilds the
A1 checkpoint runtime identity exactly; (b) new dispatch script
scripts/run_routea_val20_endpoint.py is outside the executed-source closure.
Read-only; no training, no checkpoint writes.
"""
import json, sys
from pathlib import Path

ROOT = Path("/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy")
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf
from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.build_system import build_system
from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import (
    build_runtime_identity,
    _source_rows,
)

CASES = [
    ("b2560_u128", "train_e5_dppo_common_b2560_seed202609203_v1.yaml",
     "routea_common_b2560_seed202609203_v1/checkpoint_update_000128"),
    ("b2560_u256", "train_e5_dppo_common_b2560_seed202609203_v1.yaml",
     "routea_common_b2560_seed202609203_v1/checkpoint_update_000256"),
    ("b1280_u0", "train_e5_dppo_common_b1280_seed202609203_v1.yaml",
     "routea_common_b1280_seed202609203_v1/checkpoint_update_000000"),
    ("b1280_u32", "train_e5_dppo_common_b1280_seed202609203_v1.yaml",
     "routea_common_b1280_seed202609203_v1/checkpoint_update_000032"),
]

ok = True
rows = _source_rows(ROOT)
rel = [r["relative_path"] for r in rows]
print("CLOSURE_COUNT", len(rel))
assert "scripts/run_routea_val20_endpoint.py" not in rel, "dispatch script INSIDE closure!"
print("DISPATCH_OUTSIDE_CLOSURE PASS")
print("ENTRY_IN_CLOSURE", "scripts/evaluate_dppo_uav_formal_endpoint.py" in rel)

AXES = (
    "source_set_hash",
    "config_hash",
    "bc_identity_hash",
    "task_manifest_hash",
    "reward_identity_hash",
    "observation_contract_hash",
    "action_contract_hash",
)

for label, cfg_rel, ckpt_rel in CASES:
    cfg_path = ROOT / "diffusion_policy_3d/rl/dppo_uav_rebuild/config" / cfg_rel
    ckpt = ROOT / "data/outputs/dppo_uav_rebuild" / ckpt_rel
    header = json.loads((ckpt / "header.json").read_text(encoding="utf-8"))
    stored = header["runtime_identity"]
    config = OmegaConf.load(cfg_path)
    system = build_system(config, ROOT)
    manifest = Path(config.admission_manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    ident = build_runtime_identity(config, ROOT, system, manifest.resolve())
    match = ident.content_hash == header["runtime_identity_hash"]
    ok &= match
    print("{}: {} (stored {} vs computed {})".format(
        label, "MATCH" if match else "MISMATCH",
        header["runtime_identity_hash"][:16], ident.content_hash[:16]))
    for name in AXES:
        computed = getattr(ident, name)
        expected = stored.get(name, header.get(name))
        mark = "OK " if computed == expected else "DIFF"
        if computed != expected:
            ok = False
            print("    {} {}\n        stored:   {}\n        computed: {}".format(mark, name, expected, computed))

print("PREFLIGHT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
