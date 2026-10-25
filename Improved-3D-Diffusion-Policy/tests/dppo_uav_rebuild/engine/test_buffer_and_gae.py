import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.gae import compute_gae
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.rollout_buffer import (
    OnPolicyRolloutBuffer,
)


def test_terminated_cuts_bootstrap_but_time_limit_does_not():
    reward = torch.tensor([1.0, 1.0])
    value = torch.zeros(2)
    next_value = torch.tensor([10.0, 10.0])
    terminated = torch.tensor([True, False])
    truncated = torch.tensor([False, True])
    advantage, _ = compute_gae(
        reward, value, next_value, terminated, truncated, 0.9, 0.0
    )
    torch.testing.assert_close(advantage, torch.tensor([1.0, 10.0]))


def test_buffer_emits_true_batch_128_and_full_action_chain():
    buffer = OnPolicyRolloutBuffer(128)
    condition = {"state": torch.zeros(576), "global_cond": torch.zeros(576)}
    for _ in range(128):
        buffer.add(
            condition=condition,
            chain=torch.zeros(11, 16, 11),
            old_logprobs=torch.zeros(10, 16, 11),
            reward=1.0,
            value=0.0,
            next_value=0.0,
            terminated=False,
            truncated=False,
        )
    buffer.finalize()
    batch = next(
        buffer.batches(128, generator=torch.Generator().manual_seed(3), device="cpu")
    )
    assert batch.chains_prev.shape == (128, 16, 11)
    assert batch.oldlogprobs.shape == (128, 16, 11)
    assert batch.condition["global_cond"].shape == (128, 576)
    audit_tensors = buffer.audit_tensors()
    assert set(audit_tensors) == {
        "rewards",
        "values",
        "next_values",
        "advantages",
        "returns",
    }
    assert audit_tensors["rewards"].shape == (128,)
    assert audit_tensors["advantages"].shape == (128,)
    assert all(not value.requires_grad for value in audit_tensors.values())


def test_formal_1024_rollout_emits_exactly_eight_true_minibatches():
    buffer = OnPolicyRolloutBuffer(1024)
    condition = {"state": torch.zeros(576), "global_cond": torch.zeros(576)}
    chain = torch.zeros(11, 16, 11)
    old_logprobs = torch.zeros(10, 16, 11)
    for index in range(1024):
        buffer.add(
            condition=condition,
            chain=chain,
            old_logprobs=old_logprobs,
            reward=float(index % 3),
            value=0.0,
            next_value=0.0,
            terminated=False,
            truncated=(index == 1023),
        )
    buffer.finalize()

    batches = list(
        buffer.batches(
            128,
            generator=torch.Generator().manual_seed(20260828),
            device="cpu",
        )
    )

    assert len(batches) == 8
    assert sum(batch.chains_prev.shape[0] for batch in batches) == 1024
    assert all(batch.chains_prev.shape == (128, 16, 11) for batch in batches)
