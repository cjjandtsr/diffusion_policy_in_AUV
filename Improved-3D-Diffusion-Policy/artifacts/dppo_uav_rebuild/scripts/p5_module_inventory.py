#!/usr/bin/env python
"""P5: E5 module / feature / LoRA-candidate inventory.

Outputs:
- full list of Linear / Conv1d / Conv2d named modules with weight shapes and
  param counts (LoRA candidate allowlist input for ActorIdentity),
- named_parameters summary (total / trainable),
- encoder vs diffusion-net boundaries,
- forward-hook evidence of the 3-frame history layout feeding the global cond,
- film / conditional network input shapes observed during predict_action.
"""
import os
import sys
import time

REPO = "/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from native_e5 import load_ckpt, build_policy, load_weights  # noqa: E402

DEVICE = "cuda:0"
SEED = 12345

print("=== P5: load (native path, normalizer->cuda per P4 fix) ===")
ckpt = load_ckpt()
policy = build_policy(ckpt, device=DEVICE)
load_weights(policy, ckpt, which="model", strict=False)
policy.normalizer.to(DEVICE)
policy.eval()

total_params = sum(p.numel() for p in policy.parameters())
trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
print(f"total params: {total_params:,}")
print(f"trainable params: {trainable_params:,}")

enc = policy.obs_encoder
model = policy.model
print(f"\n=== boundaries ===")
print(f"policy type: {type(policy).__module__}.{type(policy).__name__}")
print(f"obs_encoder: {type(enc).__module__}.{type(enc).__name__} "
      f"params={sum(p.numel() for p in enc.parameters()):,}")
print(f"model(diffusion net): {type(model).__module__}.{type(model).__name__} "
      f"params={sum(p.numel() for p in model.parameters()):,}")
print(f"n_obs_steps={policy.n_obs_steps} obs_feature_dim={policy.obs_feature_dim} "
      f"global_cond_dim={policy.obs_feature_dim * policy.n_obs_steps}")

print("\n=== Linear / Conv1d / Conv2d modules (name, type, weight shape, params) ===")
nlin = nconv1 = nconv2 = 0
for name, m in policy.named_modules():
    t = type(m).__name__
    if t in ("Linear", "Conv1d", "Conv2d"):
        w = m.weight
        p = w.numel() + (m.bias.numel() if m.bias is not None else 0)
        print(f"  {name:<70} {t:<7} {tuple(w.shape)}  {p:,}")
        if t == "Linear":
            nlin += 1
        elif t == "Conv1d":
            nconv1 += 1
        else:
            nconv2 += 1
print(f"counts: Linear={nlin} Conv1d={nconv1} Conv2d={nconv2}")

print("\n=== named_parameters top-level grouping ===")
groups = {}
for name, p in policy.named_parameters():
    top = name.split(".")[0]
    groups[top] = groups.get(top, 0) + p.numel()
for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<30} {v:,}")

print("\n=== forward hooks: 3-frame history -> global cond ===")
obs_hook_shape = {}
def obs_hook(module, args, output):
    inp = args[0]
    if isinstance(inp, dict):
        obs_hook_shape["input"] = {k: tuple(v.shape) for k, v in inp.items()}
    else:
        obs_hook_shape["input"] = tuple(inp.shape)
    obs_hook_shape["output"] = tuple(output.shape)
enc.register_forward_hook(obs_hook)

# hook the Linear inside each cond_encoder (film): input = global_cond or
# per-level film vector; hook model.forward to capture the raw global_cond
film_shapes = []
def mk_linear_hook(nm):
    def hook(module, args, output):
        film_shapes.append((nm, tuple(args[0].shape), tuple(output.shape)))
    return hook

def mk_model_hook(nm):
    def hook(module, args, output):
        # module-level forward hook gets no kwargs in torch 2.4;
        # global_cond shape is observed at the cond_encoder.1 Linear inputs below
        film_shapes.append((nm, None, tuple(output.shape)))
    return hook

for name, m in policy.model.named_modules():
    if name.endswith("cond_encoder.1"):  # the Linear that maps global_cond -> film
        m.register_forward_hook(mk_linear_hook(name))
        print(f"  registered hook on {name} ({type(m).__name__})")
model.register_forward_hook(mk_model_hook("model.forward"))

# minimal deterministic input: window 0 from the frozen zarr (same as P4)
import numpy as np
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer  # noqa: E402
from diffusion_policy_3d.common.sampler import (  # noqa: E402
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.dataset.drone_dataset import uniform_sampling_numpy  # noqa: E402

ZARR = "/home/cjj/AUV/avoidance_dataset/merged_body_12.zarr"
rb = ReplayBuffer.create_from_path(ZARR, mode="r")
val_mask = get_val_mask(n_episodes=rb.n_episodes, val_ratio=0.1, seed=42)
train_mask = downsample_mask(mask=~val_mask, max_n=None, seed=42)
sampler = SequenceSampler(replay_buffer=rb, sequence_length=16, pad_before=2,
                          pad_after=7, episode_mask=train_mask)
buf_start, buf_end, s_start, s_end = sampler.indices[0]
sample = {}
for key in ["state", "action", "point_cloud"]:
    arr = rb[key]
    raw = arr[buf_start:buf_end]
    if (s_start > 0) or (s_end < 16):
        data = np.zeros((16,) + arr.shape[1:], dtype=arr.dtype)
        if s_start > 0:
            data[:s_start] = raw[0]
        if s_end < 16:
            data[s_end:] = raw[-1]
        data[s_start:s_end] = raw
    else:
        data = raw
    sample[key] = data

np.random.seed(SEED)
torch.manual_seed(SEED)
pc = uniform_sampling_numpy(sample["point_cloud"].astype(np.float32), 4096)
obs_dict = {
    "agent_pos": torch.from_numpy(sample["state"].astype(np.float32)).unsqueeze(0).to(DEVICE),
    "point_cloud": torch.from_numpy(pc).unsqueeze(0).to(DEVICE),
}

print("\n=== predict_action with hooks ===")
t0 = time.time()
with torch.no_grad():
    result = policy.predict_action(obs_dict)
print(f"wall time: {time.time() - t0:.4f} s")
print(f"obs_encoder hook: input={obs_hook_shape.get('input')} output={obs_hook_shape.get('output')}")
print("  => To=n_obs_steps frames flattened into batch: B*To rows; "
      "output = B*To x obs_feature_dim")
print(f"  => global_cond = reshape(B, To*Do) = (1, {policy.obs_feature_dim * policy.n_obs_steps})")
for name, a, o in film_shapes:
    print(f"  film/cond {name}: input={a} output={o}")
print(f"action_pred shape: {tuple(result['action_pred'].shape)}")
print(f"action shape: {tuple(result['action'].shape)}")
print("DONE")
