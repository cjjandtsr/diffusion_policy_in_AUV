#!/usr/bin/env python
"""P3 part 3: normalizer content hash + native construction/load test."""
import hashlib
import sys
import traceback

import torch

sys.path.insert(0, ".")
from native_e5 import load_ckpt, build_policy, load_weights  # noqa: E402

print("=== P3: normalizer content hash ===")
ckpt = load_ckpt()
cfg = ckpt["cfg"]
sd_model = ckpt["state_dicts"]["model"]

# stable content hash: sorted keys, bytes of (dtype + shape + tensor bytes)
h = hashlib.sha256()
for k in sorted(sd_model.keys()):
    if k.startswith("normalizer.params_dict."):
        t = sd_model[k]
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
print("normalizer content sha256 (all normalizer.params_dict keys):", h.hexdigest())

nkeys = sum(1 for k in sd_model if k.startswith("normalizer.params_dict."))
print(f"normalizer keys: {nkeys}")
print(f"action offset: {sd_model['normalizer.params_dict.action.offset'].tolist()}")
print(f"action scale[:5]: {sd_model['normalizer.params_dict.action.scale'][:5].tolist()} ...")
print(f"action std[:5]: {sd_model['normalizer.params_dict.action.input_stats.std'][:5].tolist()} ...")
print(f"action mean[:5]: {sd_model['normalizer.params_dict.action.input_stats.mean'][:5].tolist()} ...")

print()
print("=== P3: native construction via hydra.utils.instantiate(cfg.policy) ===")
try:
    policy = build_policy(ckpt, device="cpu")
    print("CONSTRUCTION_OK")
    print(f"policy class: {type(policy).__module__}.{type(policy).__name__}")
    print(f"scheduler class: {type(policy.noise_scheduler).__module__}.{type(policy.noise_scheduler).__name__}")
    print(f"scheduler num_train_timesteps: {policy.noise_scheduler.config.num_train_timesteps}")
    print(f"scheduler prediction_type: {policy.noise_scheduler.config.prediction_type}")
    print(f"num_inference_steps: {policy.num_inference_steps}")
    print(f"self.kwargs keys: {sorted(policy.kwargs.keys())}")
    print(f"n_obs_steps={policy.n_obs_steps} n_action_steps={policy.n_action_steps} "
          f"horizon={policy.horizon} action_dim={policy.action_dim}")
    print(f"obs_feature_dim={policy.obs_feature_dim} obs_as_global_cond={policy.obs_as_global_cond}")
    print(f"normalizer keys after construction: {sorted(policy.normalizer.params_dict.keys())}")

    print()
    print("=== P3: load state_dicts['model'] strict=False ===")
    missing, unexpected = load_weights(policy, ckpt, which="model", strict=False)
    print(f"missing: {len(missing)} -> {missing[:10]}")
    print(f"unexpected: {len(unexpected)} -> {unexpected[:10]}")

    print()
    print("=== P3: load state_dicts['ema_model'] strict=False (for contrast) ===")
    missing_e, unexpected_e = load_weights(policy, ckpt, which="ema_model", strict=False)
    print(f"missing: {len(missing_e)} -> {missing_e[:10]}")
    print(f"unexpected: {len(unexpected_e)} -> {unexpected_e[:10]}")
except Exception:
    print("NATIVE_LOAD_FAILED; full traceback:")
    traceback.print_exc()
    sys.exit(1)
