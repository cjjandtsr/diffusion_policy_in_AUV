import math
from types import SimpleNamespace
import unittest

import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.official_loss_components import (
    AdvantageStats, compute_official_components)
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.trainer import TrainerConfig


class FakeRuntime:
    gamma_denoising = 1.
    ft_denoising_steps = 10
    clip_ploss_coef_base = .01
    clip_ploss_coef = .1
    clip_ploss_coef_rate = 3.
    clip_vloss_coef = None
    device = "cpu"

    def __init__(self, new):
        self.new = new
        self.critic = lambda _: torch.zeros(new.shape[0], 1)

    def get_logprobs_subsample(self, *args, **kwargs):
        return self.new, torch.zeros(self.new.shape[0])


class R1CreditTest(unittest.TestCase):
    def test_critic_history_never_reaches_actor_logprob_update(self):
        class StrictRuntime(FakeRuntime):
            def __init__(self, new):
                super().__init__(new)
                self.critic = self._strict_critic
                self.critic_calls = 0

            def get_logprobs_subsample(self, condition, *args, **kwargs):
                if set(condition) != {"state", "global_cond"}:
                    raise AssertionError("actor received critic-only beta")
                return super().get_logprobs_subsample(condition, *args, **kwargs)

            def _strict_critic(self, condition):
                self.critic_calls += 1
                if set(condition) != {"state", "global_cond", "beta"}:
                    raise AssertionError("critic lost history beta")
                if not torch.equal(condition["beta"], torch.tensor([[.8], [.6]])):
                    raise AssertionError("critic beta values changed")
                return torch.zeros(2, 1)

        new = torch.zeros(2, 16, 11)
        new[:, 2:6] = .1
        new[:, 6:10] = .3
        batch = SimpleNamespace(
            condition={"state": torch.zeros(2, 576),
                       "global_cond": torch.zeros(2, 576),
                       "beta": torch.tensor([[.8], [.6]])},
            chains_prev=torch.zeros(2, 16, 11), chains_next=torch.zeros(2, 16, 11),
            denoising_inds=torch.zeros(2, dtype=torch.long),
            returns=torch.zeros(2), oldvalues=torch.zeros(2),
            advantages=torch.tensor([1., 2.]), oldlogprobs=torch.zeros(2, 16, 11))
        runtime = StrictRuntime(new)
        parts = compute_official_components(
            runtime, batch, AdvantageStats.from_tensor(batch.advantages),
            reward_start=2, reward_horizon=8)
        self.assertEqual(runtime.critic_calls, 1)
        self.assertTrue(torch.allclose(parts.logratio, torch.full((2,), .2)))

    def test_eight_step_mean_pairs_same_indices_for_old_and_new(self):
        new = torch.zeros(2, 16, 11)
        new[:, 2:6] = .1
        new[:, 6:10] = .3
        new[:, 10:] = 1.9
        batch = SimpleNamespace(
            condition={"state": torch.zeros(2, 576), "global_cond": torch.zeros(2, 576)},
            chains_prev=torch.zeros(2, 16, 11), chains_next=torch.zeros(2, 16, 11),
            denoising_inds=torch.zeros(2, dtype=torch.long),
            returns=torch.zeros(2), oldvalues=torch.zeros(2),
            advantages=torch.tensor([1., 2.]), oldlogprobs=torch.zeros(2, 16, 11))
        stats = AdvantageStats.from_tensor(batch.advantages)
        a0 = compute_official_components(FakeRuntime(new), batch, stats,
                                         reward_start=2, reward_horizon=4)
        r1 = compute_official_components(FakeRuntime(new), batch, stats,
                                         reward_start=2, reward_horizon=8)
        self.assertTrue(torch.allclose(a0.logratio, torch.full((2,), .1)))
        self.assertTrue(torch.allclose(r1.logratio, torch.full((2,), .2)))
        self.assertTrue(torch.allclose(r1.ratio, torch.full((2,), math.exp(.2))))
        self.assertEqual(set(r1.slice_diagnostics), {"prefix", "tail", "full"})
        self.assertTrue(torch.allclose(r1.slice_diagnostics["prefix"]["logratio"],
                                       torch.full((2,), .1)))
        self.assertTrue(torch.allclose(r1.slice_diagnostics["tail"]["logratio"],
                                       torch.full((2,), .3)))

    def test_trainer_contract_allows_only_a0_or_r1_slice(self):
        self.assertEqual(TrainerConfig(reward_start=2, reward_horizon=8).reward_horizon, 8)
        with self.assertRaises(ValueError):
            TrainerConfig(reward_start=2, reward_horizon=6)


if __name__ == "__main__":
    unittest.main()
