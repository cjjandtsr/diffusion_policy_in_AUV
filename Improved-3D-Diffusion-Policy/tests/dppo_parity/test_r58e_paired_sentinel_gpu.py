"""R5.8-E (N11): real-GPU identical-policy paired sentinel.

The paired-RNG contract promises: same policy + same frozen bank +
same replica -> bitwise-identical episodes for BOTH sides. This gate
runs the REAL Phase1a policy (the frozen BC checkpoint, loaded fresh)
on BOTH sides of a real paired eval over the real frozen bank and the
real eval pool — per-episode outcome delta must be exactly 0.

Run (GPU): pytest -q tests/dppo_parity/test_r58e_paired_sentinel_gpu.py
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    manifest_record_key,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import (
    freeze_paired_bank,
    paired_bank_to_dict,
)
from diffusion_policy_3d.rl.dppo_uav.entrypoint_guard import file_sha256
from diffusion_policy_3d.rl.dppo_uav.integration import (
    build_adapter,
    build_routea_eval_pool,
    build_task_specs,
    entry_metadata_hash,
    manifest_eval_entries,
)
from diffusion_policy_3d.rl.dppo_uav.paired_rng import policy_noise_seed
from diffusion_policy_3d.rl.dppo_uav.policy_loader import load_ddpm_policy
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    LoopConfig,
    run_paired_eval,
)

from test_r3_rework3_gpu_gate import (  # noqa: F401  (shared real assets)
    _assets,
    _bank_specs,
    _release,
    MAX_EP_LEN,
    DEVICE,
)


@pytest.mark.gpu
def test_gpu_identical_policy_paired_delta_zero(tmp_path):
    """Identical real policy on both sides -> every episode outcome is
    bitwise-equal and every recorded noise seed re-derives from its own
    provenance."""
    manifest, worlds, worlds_val, worlds_test, bc_ckpt = _assets()
    contract = RLSystemContract()
    policy = load_ddpm_policy(bc_ckpt, device=DEVICE)

    entries = manifest_eval_entries(manifest)
    specs_eval, _ = build_task_specs(
        [dict(e, manifest_entry_hash=entry_metadata_hash(e))
         for e in entries]
    )
    pool = build_routea_eval_pool(
        manifest, worlds, worlds_val, worlds_test)
    records_eval = {
        manifest_record_key(s.world_uid, s.role, s.admission_id):
            s.manifest_entry_hash
        for s in specs_eval
    }

    # ── freeze the bank on a dedicated eval adapter (disturb OFF — the
    #    eval convention) ─────────────────────────────────────────────
    env_f = build_adapter(
        contract, worlds_dir=worlds, pool=pool,
        manifest_records=records_eval, device=DEVICE,
        max_episode_steps=MAX_EP_LEN, n_envs=1, disturb_enable=False,
    )
    bank_specs = _bank_specs(specs_eval)
    seeds = list(range(1, len(bank_specs) + 1))
    bank = freeze_paired_bank(env_f, bank_specs, seeds)
    _release(env_f)
    bank_dict = paired_bank_to_dict(
        bank, source_hash=file_sha256(Path(manifest)),
        creation_commit="r58e-paired-sentinel",
    )
    bank_id = bank_dict["bank_id"]

    loop_config = LoopConfig(
        mode="one_update", n_episodes=1, max_ep_len=MAX_EP_LEN,
        device=DEVICE, output_dir=str(tmp_path),
        eval_n_episodes=len(bank),
        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1,
    )
    eval_env = build_adapter(
        contract, worlds_dir=worlds, pool=pool,
        manifest_records=records_eval, device=DEVICE,
        max_episode_steps=MAX_EP_LEN, n_envs=1, disturb_enable=False,
    )
    records, invalid = run_paired_eval(
        eval_env, policy, policy, contract, loop_config, bank,
        bank_id=bank_id,
        policy_checkpoint_hash="ckpt-eval1",
    )
    _release(eval_env)
    assert invalid == 0
    assert len(records) == len(bank)
    for rec in records:
        assert rec.bc == rec.rl, (
            f"identical-policy sides diverged: bc={rec.bc} rl={rec.rl}"
        )
        expected_seed = policy_noise_seed(
            bank_id, rec.episode_config["episode_config_hash"],
            replica_id=rec.replica_id,
        )
        assert rec.policy_noise_seed == expected_seed, (
            "recorded noise seed does not re-derive from its provenance"
        )
