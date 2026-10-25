"""Exact transition x denoising-step coverage for all reviewed modes."""

from collections import Counter

import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.pair_schedule import (
    iter_pair_passes,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.rollout_buffer import (
    OnPolicyRolloutBuffer,
)


def _buffer(size=1024):
    buffer = OnPolicyRolloutBuffer(size, outer_update_id=3)
    condition = {"state": torch.zeros(576), "global_cond": torch.zeros(576)}
    for index in range(size):
        buffer.add(
            condition=condition,
            chain=torch.zeros(11, 16, 11),
            old_logprobs=torch.zeros(10, 16, 11),
            reward=0.0, value=0.0, next_value=0.0,
            terminated=False, truncated=index == size - 1,
            transition_id="u3:t{}".format(index), role="anticipatory",
            outcome="none", provenance_hash="a" * 64,
        )
    buffer.finalize()
    return buffer


def _collect(buffer, mode, passes, logical):
    result = []
    for batch in iter_pair_passes(
        buffer, mode=mode, pair_passes=passes,
        logical_batch_pairs=logical, microbatch_pairs=128,
        generator=torch.Generator().manual_seed(7), device=torch.device("cpu"),
    ):
        result.extend(
            (batch.pass_id, int(b), int(k))
            for b, k in zip(batch.transition_indices, batch.denoising_inds)
        )
    return result


def test_full_joint_pass_has_exact_coverage():
    pairs = _collect(_buffer(), "full_joint_accum_batch1280", 1, 1280)
    plain = [(b, k) for _, b, k in pairs]
    assert len(plain) == 10240 and len(set(plain)) == 10240
    assert Counter(k for _, k in plain) == Counter({k: 1024 for k in range(10)})


def test_sampled_k_has_one_pair_per_transition_per_pass():
    pairs = _collect(_buffer(), "sampled_k", 5, 128)
    assert len(pairs) == 5120
    for pass_id in range(5):
        assert sorted(b for p, b, _ in pairs if p == pass_id) == list(range(1024))
