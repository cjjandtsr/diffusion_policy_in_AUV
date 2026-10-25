"""Logical commit/microbatch cardinalities are distinct audit quantities."""

import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.pair_schedule import iter_pair_passes
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.rollout_buffer import OnPolicyRolloutBuffer


def _buffer(size=1024):
    buffer = OnPolicyRolloutBuffer(size, outer_update_id=3)
    condition = {"state": torch.zeros(576), "global_cond": torch.zeros(576)}
    for index in range(size):
        buffer.add(
            condition=condition, chain=torch.zeros(11, 16, 11),
            old_logprobs=torch.zeros(10, 16, 11), reward=0.0, value=0.0,
            next_value=0.0, terminated=False, truncated=index == size - 1,
            transition_id="u3:t{}".format(index), role="anticipatory",
            outcome="none", provenance_hash="a" * 64,
        )
    buffer.finalize()
    return buffer


@pytest.mark.parametrize(
    "mode,logical,passes,want_pairs,want_commits,want_microbatches",
    [
        ("sampled_k", 128, 5, 5120, 40, 40),
        ("full_joint_flat", 128, 5, 51200, 400, 400),
        ("full_joint_accum_batch1280", 1280, 5, 51200, 40, 400),
    ],
)
def test_schedule_cardinalities(mode, logical, passes, want_pairs, want_commits, want_microbatches):
    batches = list(iter_pair_passes(
        _buffer(), mode=mode, pair_passes=passes,
        logical_batch_pairs=logical, microbatch_pairs=128,
        generator=torch.Generator().manual_seed(11), device=torch.device("cpu"),
    ))
    assert sum(batch.pair_count for batch in batches) == want_pairs
    assert len(batches) == want_commits
    assert sum(batch.microbatch_count for batch in batches) == want_microbatches


def test_illegal_schedule_parameters_fail_before_materialization():
    with pytest.raises(ValueError):
        list(iter_pair_passes(
            _buffer(16), mode="unknown", pair_passes=1,
            logical_batch_pairs=128, microbatch_pairs=128,
            generator=torch.Generator(), device=torch.device("cpu"),
        ))
