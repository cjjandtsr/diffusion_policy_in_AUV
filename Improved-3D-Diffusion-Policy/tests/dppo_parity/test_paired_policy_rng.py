"""E1-E3 (review R5.4 / §8.4): paired policy-noise RNG.

E1 identical-policy sentinel: BC and RL with the SAME weights and the
same derived policy-noise seed must produce bitwise-identical paired
episodes — the old code ran BC then RL on one shared RNG stream, so
even identical policies diverged (the comparison mixed environment and
policy-noise variance). E2/E3: the derivation formula and the artifact
RNG provenance fields.
"""

import torch

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    EpisodeTaskSpec,
)
from diffusion_policy_3d.rl.dppo_uav.evaluator import freeze_paired_bank
from diffusion_policy_3d.rl.dppo_uav.paired_rng import (
    RNG_CONTRACT_VERSION,
    policy_noise_seed,
)
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    LoopConfig,
    run_paired_eval,
)

from fake_policy import FakePolicy
from mock_uav_env import MockUAVEnv


def _specs():
    return [
        EpisodeTaskSpec(
            world_uid=f"mock_rng_{role}",
            obstacle_json="{}",
            role=role,
            split="rl_train",
            admission_id=f"mock_rng__{role}",
            generator_version="mock",
            manifest_entry_hash=f"hash_rng_{role}",
        )
        for role in ("anticipatory", "dense_recovery",
                     "support_expansion", "yaw_lateral_recovery")
    ]


def _cfg():
    return LoopConfig(
        mode="one_update", n_episodes=2, max_ep_len=12, device="cpu",
        output_dir="/tmp/none", eval_n_episodes=2,
        min_sampling_denoising_std=0.07,
        min_logprob_denoising_std=0.11,
    )


def test_seed_derivation_matches_spec():
    """E2: the derivation is uint64(SHA256(bank||entry||replica)[0:8])."""
    import hashlib

    s = policy_noise_seed("bank", "entry", 0)
    expected = int.from_bytes(
        hashlib.sha256(b"bank||entry||0").digest()[:8], "little"
    )
    assert s == expected
    assert policy_noise_seed("bank", "entry", 0) == policy_noise_seed(
        "bank", "entry", 0
    )
    assert policy_noise_seed("bank", "entry", 1) != s  # replica axis


def test_e1_identical_policy_pair_is_bitwise_identical():
    """E1: identical BC/RL weights + same derived noise seed -> identical
    paired episodes (outcome AND trace)."""
    contract = RLSystemContract()
    bank = freeze_paired_bank(
        MockUAVEnv(contract), _specs()[:2], (1001, 1002)
    )
    policy = FakePolicy(contract)
    records, invalid = run_paired_eval(
        MockUAVEnv(contract), policy, policy, contract, _cfg(), bank,
        bank_id="test-bank",
        policy_checkpoint_hash="ckpt-eval1",
    )
    assert invalid == 0
    assert len(records) == 2
    for rec in records:
        for key in ("success", "collision", "timeout", "return", "steps"):
            assert rec.bc[key] == rec.rl[key], (
                f"identical policies diverged on {key}: "
                f"{rec.bc[key]} vs {rec.rl[key]}"
            )
        bc_trace = rec.trace.get("bc", [])
        rl_trace = rec.trace.get("rl", [])
        assert len(bc_trace) == len(rl_trace), "trace length diverged"
        for a, b in zip(bc_trace, rl_trace):
            for tkey, va in a.items():
                if tkey == "elapsed":
                    # wall-clock timing is not part of the RNG contract
                    continue
                assert va == b.get(tkey), (
                    f"trace field {tkey} diverged: {va} vs {b.get(tkey)}"
                )


def test_e3_artifact_records_rng_contract():
    """E3: every record carries the derived seed + replica + contract."""
    contract = RLSystemContract()
    bank = freeze_paired_bank(
        MockUAVEnv(contract), _specs()[:2], (1001, 1002)
    )
    policy = FakePolicy(contract)
    records, _ = run_paired_eval(
        MockUAVEnv(contract), policy, policy, contract, _cfg(), bank,
        bank_id="test-bank",
        policy_checkpoint_hash="ckpt-eval1",
    )
    for rec in records:
        assert rec.policy_noise_seed == policy_noise_seed(
            "test-bank", rec.episode_config["episode_config_hash"], 0
        )
        assert rec.replica_id == 0
        assert rec.rng_contract_version == RNG_CONTRACT_VERSION
