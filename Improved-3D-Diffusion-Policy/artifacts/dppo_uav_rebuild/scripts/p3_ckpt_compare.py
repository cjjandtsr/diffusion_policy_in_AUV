#!/usr/bin/env python
"""P3 probe part 2: model vs ema_model comparison, normalizer search."""
import sys
import traceback
import torch

CKPT = "/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy/data/outputs/2026.08.16/08.53.26_e5_lambda0p5/checkpoints/latest.ckpt"

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
sds = ckpt["state_dicts"]
print("state_dicts keys:", list(sds.keys()))

model = sds["model"]
ema = sds["ema_model"]

print(f"\nmodel: {len(model)} keys")
print(f"ema_model: {len(ema)} keys")
print(f"model keys == ema keys: {set(model.keys()) == set(ema.keys())}")

if set(model.keys()) == set(ema.keys()):
    diffs = []
    max_abs = 0.0
    sum_abs = 0.0
    n_total = 0
    for k in sorted(model.keys()):
        m, e = model[k], ema[k]
        if m.shape != e.shape:
            diffs.append((k, tuple(m.shape), tuple(e.shape)))
            continue
        if m.numel() == 0:
            continue
        d = (m - e).abs()
        max_abs = max(max_abs, d.max().item())
        sum_abs += d.sum().item()
        n_total += d.numel()
    print(f"shape-mismatched tensors: {len(diffs)}")
    for k, sm, se in diffs[:10]:
        print(f"  MISMATCH {k}: model{sm} vs ema{se}")
    print(f"numel compared: {n_total}")
    print(f"max abs diff (all tensors): {max_abs}")
    print(f"mean abs diff: {sum_abs / n_total}")
    # count tensors that differ at all
    n_diff = 0
    for k in sorted(model.keys()):
        if model[k].shape == ema[k].shape and not torch.equal(model[k], ema[k]):
            n_diff += 1
    print(f"tensors that differ (torch.equal False): {n_diff}")

# optimizer keys
opt = sds["optimizer"]
print(f"\noptimizer: {type(opt)}; keys: {list(opt.keys()) if isinstance(opt, dict) else 'not a dict'}")

# normalizer search
print("\n=== normalizer search ===")
for sd_name, sd in sds.items():
    hits = [k for k in sd.keys() if "normal" in k.lower()]
    if hits:
        print(f"  {sd_name}: {len(hits)} normalizer-ish keys")
        for k in hits[:20]:
            v = sd[k]
            if hasattr(v, "shape"):
                print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                print(f"    {k}: {type(v)}")
    else:
        print(f"  {sd_name}: no keys containing 'normal'")
