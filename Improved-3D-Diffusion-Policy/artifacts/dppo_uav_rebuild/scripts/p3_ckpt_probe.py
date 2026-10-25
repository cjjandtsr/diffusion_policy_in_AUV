#!/usr/bin/env python
"""P3 probe: E5 checkpoint identity.

Loads latest.ckpt with the native loader semantics (torch.load, cpu) and
reports structure WITHOUT touching any RL policy_loader.py.
"""
import hashlib
import os
import sys
import traceback

CKPT = "/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy/data/outputs/2026.08.16/08.53.26_e5_lambda0p5/checkpoints/latest.ckpt"

print("=== torch.load(weights_only=False, map_location=cpu) ===")
try:
    import torch
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    print("LOAD_OK")
except Exception:
    traceback.print_exc()
    sys.exit(1)

print("\n=== top-level keys ===")
for k in ckpt.keys():
    v = ckpt[k]
    if isinstance(v, dict):
        print(f"  {k}: dict with {len(v)} keys")
    elif isinstance(v, (list, tuple)):
        print(f"  {k}: {type(v).__name__} len={len(v)}")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")

for sd_name in ["state_dicts", "state_dict", "model", "ema_model"]:
    if sd_name in ckpt:
        sd = ckpt[sd_name]
        if isinstance(sd, dict) and all(not isinstance(x, dict) for x in sd.values()):
            print(f"\n=== {sd_name}: {len(sd)} param keys ===")
        else:
            print(f"\n=== {sd_name}: contains nested dicts, keys: {list(sd.keys())[:20]} ===")

# model / ema_model deep-dive
for name in ["model", "ema_model"]:
    if name in ckpt and isinstance(ckpt[name], dict):
        sd = ckpt[name]
        n_params = sum(1 for k in sd if not k.endswith("num_batches_tracked"))
        print(f"\n=== {name}: {len(sd)} keys, {n_params} parameter/buffer tensors ===")
        keys = sorted(sd.keys())
        print("first 15 keys:", keys[:15])
        print("last 5 keys:", keys[-5:])

# cfg inside ckpt?
if "cfg" in ckpt:
    print("\n=== ckpt['cfg'] present ===")
    cfg = ckpt["cfg"]
    print(type(cfg))
    try:
        import omegaconf
        if isinstance(cfg, omegaconf.DictConfig):
            print(omegaconf.OmegaConf.to_yaml(cfg)[:6000])
        else:
            print(str(cfg)[:6000])
    except Exception as e:
        print(f"cfg dump failed: {e}; raw type={type(cfg)}")
else:
    print("\n=== ckpt has no 'cfg' key ===")
