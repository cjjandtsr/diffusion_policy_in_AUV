from __future__ import annotations

import unittest

import torch

from diffusion_policy_3d.rl.reinflow_fm.rollout_buffer import (
    FlowRolloutBuffer,
    RolloutContractError,
)


def _batch(batch_size=2):
    return dict(
        condition={
            "state": torch.randn(batch_size, 12),
            "global_cond": torch.randn(batch_size, 12),
            "global_cond_uncond": torch.randn(batch_size, 12),
            "noise_cond": torch.randn(batch_size, 4),
        },
        chain=torch.randn(batch_size, 5, 16, 11),
        old_logprob=torch.randn(batch_size),
        reward=torch.randn(batch_size),
        value=torch.randn(batch_size),
        next_value=torch.randn(batch_size),
        terminated=torch.zeros(batch_size, dtype=torch.bool),
        truncated=torch.zeros(batch_size, dtype=torch.bool),
        world_ids=("train_dense_a",) * batch_size,
        task_ids=tuple(f"task-{index}" for index in range(batch_size)),
        reward_identity="a" * 64,
        action_identity="b" * 64,
        condition_identities=("c" * 64,) * batch_size,
        checkpoint_identity="d" * 64,
    )


class FlowRolloutBufferTests(unittest.TestCase):
    def test_valid_official_four_step_chains_round_trip(self):
        buffer = FlowRolloutBuffer(capacity=2)
        buffer.add_batch(**_batch())
        batch = buffer.as_batch(device="cpu")

        self.assertEqual(tuple(batch.chain.shape), (2, 5, 16, 11))
        self.assertEqual(tuple(batch.old_logprob.shape), (2,))
        self.assertEqual(batch.reward_identity, "a" * 64)
        self.assertEqual(batch.world_ids, ("train_dense_a", "train_dense_a"))

    def test_chain_shape_and_nonfinite_values_are_rejected(self):
        bad_shape = _batch()
        bad_shape["chain"] = torch.randn(2, 4, 16, 11)
        with self.assertRaisesRegex(RolloutContractError, "chain"):
            FlowRolloutBuffer(capacity=2).add_batch(**bad_shape)

        nonfinite = _batch()
        nonfinite["old_logprob"][0] = float("nan")
        with self.assertRaisesRegex(RolloutContractError, "non-finite"):
            FlowRolloutBuffer(capacity=2).add_batch(**nonfinite)

    def test_eval_only_worlds_are_rejected_from_training_buffer(self):
        for world in ("w4", "w8", "w11", "w12", "w18"):
            values = _batch()
            values["world_ids"] = (world, "train_dense_a")
            with self.subTest(world=world), self.assertRaisesRegex(
                RolloutContractError, "eval-only"
            ):
                FlowRolloutBuffer(capacity=2).add_batch(**values)

    def test_missing_or_malformed_identity_is_rejected(self):
        values = _batch()
        values["reward_identity"] = "missing"
        with self.assertRaisesRegex(RolloutContractError, "reward_identity"):
            FlowRolloutBuffer(capacity=2).add_batch(**values)

        values = _batch()
        values["condition_identities"] = ("c" * 64,)
        with self.assertRaisesRegex(RolloutContractError, "condition_identities"):
            FlowRolloutBuffer(capacity=2).add_batch(**values)


if __name__ == "__main__":
    unittest.main()
