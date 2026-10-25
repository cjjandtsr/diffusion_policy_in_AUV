#!/usr/bin/env python
"""P4 (lazy-memory): native E5 single-sample deterministic smoke.

Replicates DroneDataset window selection exactly (get_val_mask ->
downsample_mask -> create_indices -> sample_sequence padding) but opens
the zarr lazily (ReplayBuffer.create_from_path) to avoid the 5.9GB
in-memory copy that OOM-killed the eager version.
"""
import hashlib
import os
import sys
import time

REPO = "/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from native_e5 import load_ckpt, build_policy, load_weights  # noqa: E402
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer  # noqa: E402
from diffusion_policy_3d.common.sampler import (  # noqa: E402
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer  # noqa: E402

ZARR = "/home/cjj/AUV/avoidance_dataset/merged_body_12.zarr"
DEVICE = "cuda:0"
SEED = 12345


def tensor_sha256(t):
    return hashlib.sha256(t.detach().cpu().float().numpy().tobytes()).hexdigest()


def stat(name, t):
    t = t.detach().cpu()
    print(f"  {name}: shape={tuple(t.shape)} dtype={t.dtype} "
          f"min={t.min().item():.6f} max={t.max().item():.6f} "
          f"mean={t.mean().item():.6f} sha256={tensor_sha256(t)}")


def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def hash_normalizer_state(sd):
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        t = sd[k]
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


print("=== dataset (lazy) ===")
rb = ReplayBuffer.create_from_path(ZARR, mode="r")
print(f"n_episodes={rb.n_episodes} n_steps={rb.n_steps} keys={list(rb.keys())}")

val_mask = get_val_mask(n_episodes=rb.n_episodes, val_ratio=0.1, seed=42)
train_mask = ~val_mask
train_mask = downsample_mask(mask=train_mask, max_n=None, seed=42)
print(f"train episodes={int(train_mask.sum())} val episodes={int(val_mask.sum())}")

sampler = SequenceSampler(
    replay_buffer=rb,
    sequence_length=16,
    pad_before=2, pad_after=7,
    episode_mask=train_mask,
)
print(f"train windows={len(sampler)}")
idx = 0
buf_start, buf_end, s_start, s_end = sampler.indices[idx]
ep_ends = rb.episode_ends
ep_starts = np.concatenate([[0], ep_ends[:-1]])
ep_idx = int(np.searchsorted(ep_ends, buf_start, side="right"))
print(f"window idx={idx}: buffer=[{buf_start},{buf_end}) sample=[{s_start},{s_end}) "
      f"episode={ep_idx} frame_in_episode={buf_start - ep_starts[ep_idx]} "
      f"episode_frames=[{ep_starts[ep_idx]},{ep_ends[ep_idx]})")

# replicate SequenceSampler.sample_sequence for window 0
sample = {}
for key in ["state", "action", "point_cloud"]:
    arr = rb[key]
    raw = arr[buf_start:buf_end]
    data = raw
    if (s_start > 0) or (s_end < 16):
        data = np.zeros((16,) + arr.shape[1:], dtype=arr.dtype)
        if s_start > 0:
            data[:s_start] = raw[0]
        if s_end < 16:
            data[s_end:] = raw[-1]
        data[s_start:s_end] = raw
    sample[key] = data

agent_pos = sample["state"].astype(np.float32)
point_cloud = sample["point_cloud"].astype(np.float32)
action = sample["action"].astype(np.float32)

# replicate DroneDataset._sample_to_data downsampling (deterministic given np seed)
from diffusion_policy_3d.dataset.drone_dataset import uniform_sampling_numpy  # noqa: E402
set_seed()
point_cloud = uniform_sampling_numpy(point_cloud, 4096)

print(f"sample shapes: agent_pos={agent_pos.shape} point_cloud={point_cloud.shape} action={action.shape}")

obs_dict = {
    "agent_pos": torch.from_numpy(agent_pos).unsqueeze(0),
    "point_cloud": torch.from_numpy(point_cloud).unsqueeze(0),
}
gt_action = torch.from_numpy(action).unsqueeze(0)

print("\n=== normalizer recompute from full action column (gaussian) ===")
action_full = rb["action"][:]  # (60151, 11) — 2.6MB, fine
norm = LinearNormalizer()
norm.fit(data={"action": action_full}, last_n_dims=1, mode="gaussian")
rh = hash_normalizer_state(norm.state_dict())
CKPT_NORM_HASH = "1edbc81f78c66a0f67270266cf333c06071307768e1273d2369aa945320ae1ca"
print("recomputed:", rh)
print("ckpt      :", CKPT_NORM_HASH)
print("MATCH" if rh == CKPT_NORM_HASH else "MISMATCH")
if rh != CKPT_NORM_HASH:
    ck = load_ckpt()
    ck_sd = ck["state_dicts"]["model"]
    for k in ["offset", "scale"]:
        a = ck_sd[f"normalizer.params_dict.action.{k}"].numpy()
        b = norm.state_dict()[f"params_dict.action.{k}"].numpy()
        print(f"  action.{k}: ckpt[:4]={a[:4].tolist()} recomputed[:4]={b[:4].tolist()}")
        print(f"  action.{k} max abs diff: {np.abs(a - b).max():.2e}")

print("\n=== native policy on cuda ===")
ckpt = load_ckpt()
policy = build_policy(ckpt, device=DEVICE)
load_weights(policy, ckpt, which="model", strict=False)
policy.eval()
obs_dict_gpu = {k: v.to(DEVICE) for k, v in obs_dict.items()}

print("\n=== predict_action run 1 ===")
set_seed()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.no_grad():
    result = policy.predict_action(obs_dict_gpu)
t1 = time.time()
print(f"wall time: {t1 - t0:.4f} s")
print(f"peak CUDA mem allocated: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
print(f"returned keys: {sorted(result.keys())}")

full = result["action_pred"]
policy_act = result["action"]
executed = full[:, 2:4]
print(f"scheduler timesteps (K={policy.num_inference_steps}): "
      f"{policy.noise_scheduler.timesteps.detach().cpu().tolist()}")
print("full:")
stat("full", full[0])
print("policy:")
stat("policy_act", policy_act[0])
print("executed (full[:,2:4]):")
stat("executed", executed[0])

print("\n=== slice equality ===")
print("policy == full[:, 2:10] (exact):",
      torch.equal(policy_act, full[:, 2:10]))
print("max abs diff policy vs full[:,2:10]:", (policy_act - full[:, 2:10]).abs().max().item())
print("executed == full[:, 2:4] (exact):", torch.equal(executed, full[:, 2:4]))

print("\n=== run 2 (same process, same seeds, same input) ===")
set_seed()
with torch.no_grad():
    result2 = policy.predict_action(obs_dict_gpu)
full2 = result2["action_pred"]
policy_act2 = result2["action"]
print("full max abs diff run1 vs run2:", (full - full2).abs().max().item())
print("policy max abs diff run1 vs run2:", (policy_act - policy_act2).abs().max().item())
print("full exactly equal:", torch.equal(full, full2))
print("policy exactly equal:", torch.equal(policy_act, policy_act2))

print("\n=== obs tensor fingerprint ===")
for k in sorted(obs_dict.keys()):
    v = obs_dict[k]
    print(f"obs['{k}']: shape={tuple(v.shape)} dtype={v.dtype} "
          f"min={v.min().item():.4f} max={v.max().item():.4f} mean={v.mean().item():.4f} "
          f"sha256={tensor_sha256(v)}")
print(f"gt_action: shape={tuple(gt_action.shape)} sha256={tensor_sha256(gt_action)}")
