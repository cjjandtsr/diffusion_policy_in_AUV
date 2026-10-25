#!/usr/bin/env python
"""P4: native E5 single-sample deterministic smoke.

Fixed real sample from the frozen dataset (window 0, train split), fixed
seeds, batch-1 predict_action, no backward. Two repeats in one process.
Also recomputes the normalizer from the zarr to verify hash reproducibility.
"""
import hashlib
import os
import sys
import time

sys.path.insert(0, "/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from native_e5 import load_ckpt, build_policy, load_weights  # noqa: E402
from diffusion_policy_3d.dataset.drone_dataset import DroneDataset  # noqa: E402

ZARR = "/home/cjj/AUV/avoidance_dataset/merged_body_12.zarr"
DEVICE = "cuda:0"
SEED = 12345


def tensor_sha256(t):
    t = t.detach().cpu().float().numpy()
    return hashlib.sha256(t.tobytes()).hexdigest()


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


print("=== dataset / fixed sample ===")
ds = DroneDataset(
    zarr_path=ZARR, horizon=16, pad_before=2, pad_after=7, seed=42,
    val_ratio=0.1, max_train_episodes=None, task_name="drone_trajectory",
    num_points=4096, augment_pc=False,  # probe choice: no augmentation (determinism)
)
print(f"zarr: {ZARR}")
print(f"n_episodes={ds.replay_buffer.n_episodes} n_frames={len(ds.replay_buffer)}")
print(f"train windows={len(ds.sampler)}")

idx = 0
buf_start = int(ds.sampler.indices[idx, 0])
ep_ends = ds.replay_buffer.episode_ends
ep_idx = int(np.searchsorted(ep_ends, buf_start, side="right"))
ep_start = int(np.concatenate([[0], ep_ends[:-1]])[ep_idx])
print(f"sample window idx={idx}: buffer_start_idx={buf_start}, episode={ep_idx}, "
      f"frame_in_episode={buf_start - ep_start}, episode_frames=[{ep_start},{ep_ends[ep_idx]})")

sample = ds[idx]
obs = sample["obs"]
agent_pos = obs["agent_pos"].unsqueeze(0)          # (1,16,9)
point_cloud = obs["point_cloud"].unsqueeze(0)      # (1,16,4096,6)
obs_dict = {"agent_pos": agent_pos, "point_cloud": point_cloud}
gt_action = sample["action"].unsqueeze(0)          # (1,16,11)
print("obs sample shapes:", {k: tuple(v.shape) for k, v in obs_dict.items()})
print("gt action shape:", tuple(gt_action.shape))
print("sample['agent_pos'] dtype:", obs["agent_pos"].dtype,
      "point_cloud dtype:", obs["point_cloud"].dtype)

print("\n=== normalizer recompute check ===")
norm = ds.get_normalizer(mode="gaussian")  # same path as E5 training
h = hashlib.sha256()
sd = norm.state_dict()
for k in sorted(sd.keys()):
    t = sd[k]
    h.update(k.encode())
    h.update(str(t.dtype).encode())
    h.update(str(tuple(t.shape)).encode())
    h.update(t.numpy().tobytes())
print("recomputed normalizer sha256:", h.hexdigest())
print("ckpt normalizer     sha256: 1edbc81f78c66a0f67270266cf333c06071307768e1273d2369aa945320ae1ca")
print("MATCH" if h.hexdigest() == "1edbc81f78c66a0f67270266cf333c06071307768e1273d2369aa945320ae1ca" else "MISMATCH")

print("\n=== native policy on cuda ===")
ckpt = load_ckpt()
policy = build_policy(ckpt, device=DEVICE)
load_weights(policy, ckpt, which="model", strict=False)
policy.eval()
print(f"device: {next(policy.parameters()).device}")
print(f"dtype: {next(policy.parameters()).dtype}")

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

full = result["action_pred"]          # (1,16,11)
policy_act = result["action"]         # (1,8,11)
executed = full[:, 2:4]               # probe-defined executed slice
sched_timesteps = policy.noise_scheduler.timesteps.detach().cpu().tolist()
print(f"scheduler timesteps (after set_timesteps({policy.num_inference_steps})): {sched_timesteps}")

print("full:")
stat("full", full[0])
print("policy:")
stat("policy_act", policy_act[0])
print("executed (full[:,2:4]):")
stat("executed", executed[0])

print("\n=== slice equality ===")
print("policy == full[:, 2:10]:",
      torch.allclose(policy_act, full[:, 2:10], atol=0, rtol=0))
print("max abs diff (policy vs full[:,2:10]):",
      (policy_act - full[:, 2:10]).abs().max().item())
print("executed == full[:, 2:4]:",
      torch.allclose(executed, full[:, 2:4], atol=0, rtol=0))

print("\n=== predict_action run 2 (same process, same seeds, same input) ===")
set_seed()
with torch.no_grad():
    result2 = policy.predict_action(obs_dict_gpu)
full2 = result2["action_pred"]
policy_act2 = result2["action"]
print("full max abs diff run1 vs run2:", (full - full2).abs().max().item())
print("policy max abs diff run1 vs run2:", (policy_act - policy_act2).abs().max().item())
print("full exactly equal:", torch.equal(full, full2))

print("\n=== obs tensor fingerprint ===")
for k in sorted(obs_dict.keys()):
    v = obs_dict[k]
    print(f"obs['{k}']: shape={tuple(v.shape)} dtype={v.dtype} "
          f"min={v.min().item():.4f} max={v.max().item():.4f} mean={v.mean().item():.4f} "
          f"sha256={tensor_sha256(v)}")
