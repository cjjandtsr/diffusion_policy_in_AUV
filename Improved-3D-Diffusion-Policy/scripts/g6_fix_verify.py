"""g6_fix_verify.py — fast (no-rollout) verification of the G6 run-5 fix.

Run 5 of the gate test died at the first rollout step of the resumed
world: PolicyObsEncoder fed CPU observations into the CUDA pointnet.
Root cause: restore_training_state loads policy_state (saved as CPU
clones) and DictOfTensorMixin._load_from_state_dict REPLACES the
normalizer's params with them -> scale/offset end up on CPU ->
LinearNormalizer.normalize casts inputs to scale.device -> CPU obs.

This script reproduces the exact restore path (real checkpoint built
with the production _build_checkpoint_data + CheckpointBuilder.save)
without any rollout, and asserts the invariant the fix restores:
after restore_training_state, every normalizer param sits on the
policy model's device, and a PolicyObsEncoder forward on CUDA obs runs.

Usage: python scripts/g6_fix_verify.py   (~1 min: policy load + forward)
Exit 0 only when every check passes.
"""

import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from diffusion_policy_3d.policy.lora_adapters import insert_lora
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.critic import PolicyObsEncoder, ValueCritic
from diffusion_policy_3d.rl.dppo_uav.policy_loader import (
    extract_lora_state,
    load_ddpm_policy,
)
from diffusion_policy_3d.rl.dppo_uav.checkpoint import CheckpointBuilder
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    FORMAL_NUMERICAL_MODE,
    LoopConfig,
    _build_checkpoint_data,
    restore_training_state,
    resume_identity_for,
)
from diffusion_policy_3d.rl.dppo_uav.numerics import configure_numerical_mode
from diffusion_policy_3d.rl.dppo_uav.trainer import TrainerConfig

DEFAULT_BC_CKPT = (
    "data/outputs/2026.05.17/13.36.20_v8_phase2_sweep_lambda0.5"
    "/checkpoints/latest.ckpt"
)
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = "cuda:0"
policy = load_ddpm_policy(DEFAULT_BC_CKPT, device=device)
lora_info = insert_lora(policy.model, r=8, alpha=8, verbose=False)
contract = RLSystemContract()
critic = ValueCritic(contract, PolicyObsEncoder(policy)).to(device)
opt_a = torch.optim.Adam(lora_info["lora_params"], lr=1e-3)
opt_c = torch.optim.Adam(critic.value_head_parameters(), lr=1e-3)

# ── sanity: pre-restore the normalizer is on the model's device ─────
anchor = next(policy.model.parameters())
for p in policy.normalizer.parameters():
    assert p.device == anchor.device, f"pre-restore normalizer on {p.device}"
print(f"pre-restore: normalizer on {anchor.device} OK")

# ── build a real checkpoint exactly like run_e2e_batch does ─────────
outdir = REPO_ROOT / "temp" / "g6_fix_verify"
outdir.mkdir(parents=True, exist_ok=True)
runtime = {
    "device": device, "reward_version": "v16", "manifest_hash": "g6-fix-verify",
    "project_git_commit": "?", "project_tree_hash": "?",
    "source_file_hashes": {}, "bc_checkpoint_hash": "?", "world_assets_hash": "?",
    "timestamp": "x",
}
loop_config = LoopConfig(mode="one_update", n_episodes=1, max_ep_len=1,
                         device=device, output_dir=str(outdir))
trainer_config = TrainerConfig(batch_size=128, device=device)
# The resolved numerical mode is part of the config_hash layer — the
# save side and the resume side MUST encode the same dict (run_e2e_batch
# does this by construction; a missing resolved mode would hash `null`
# here and the resolved dict at restore, hard-failing the identity gate).
numerical_mode_resolved = configure_numerical_mode(FORMAL_NUMERICAL_MODE)
data = _build_checkpoint_data(
    policy, critic, contract, loop_config, trainer_config,
    metrics={"update_batches": 1, "next_value_batch_size": 1},
    env_steps=1, completed=1,
    lora_state={k: v.detach().cpu().clone()
                for k, v in extract_lora_state(lora_info).items()},
    runtime=runtime, actor_optimizer=opt_a, critic_optimizer=opt_c,
    numerical_mode_resolved=numerical_mode_resolved,
    # The live wiring (schema v3): without it the checkpoint records a
    # zero-rank lora_identity over a NON-empty lora_state, and the
    # restore side refuses (LoRA wiring inconsistency).
    lora_info=lora_info,
)
ckpt = outdir / "fix_verify.ckpt"
CheckpointBuilder.save(data, ckpt)
print(f"checkpoint written: {ckpt}")

# ── the fix path: restore into a FRESH world ────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
policy2 = load_ddpm_policy(DEFAULT_BC_CKPT, device=device)
lora2 = insert_lora(policy2.model, r=8, alpha=8, verbose=False)
critic2 = ValueCritic(contract, PolicyObsEncoder(policy2)).to(device)
opt2a = torch.optim.Adam(lora2["lora_params"], lr=1e-3)
opt2c = torch.optim.Adam(critic2.value_head_parameters(), lr=1e-3)

restore_training_state(
    str(ckpt), policy2, critic2, lora2, opt2a, opt2c,
    runtime_identity=resume_identity_for(
        contract, loop_config, trainer_config, runtime,
        numerical_mode_resolved, policy2,
        # The training_identity axes must match the save side
        # (schema v3, P0-2): lora wiring + optimizers with the same lr.
        lora_info=lora2,
        actor_optimizer=opt2a,
        critic_optimizer=opt2c,
    ),
)

anchor2 = next(policy2.model.parameters())
devs = {p.device for p in policy2.normalizer.parameters()}
assert devs == {anchor2.device}, f"normalizer after restore: {devs}"
print(f"post-restore: normalizer on {anchor2.device} OK (fix verified)")

# ── the exact call that crashed run 5: PolicyObsEncoder on CUDA obs ─
obs = {
    "point_cloud": torch.randn(1, 3, 4096, 6, device=device),
    "agent_pos": torch.randn(1, 3, 9, device=device),
}
g = PolicyObsEncoder(policy2)(obs)
assert g.device == torch.device(device), f"global_cond on {g.device}"
print(f"PolicyObsEncoder forward OK, global_cond {tuple(g.shape)} on "
      f"{g.device}")

print("G6 fix verify PASS")
sys.exit(0)
