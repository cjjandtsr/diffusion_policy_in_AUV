import importlib
import math
import unittest

import torch
from types import SimpleNamespace


class CreditProbeTest(unittest.TestCase):
    def _module(self):
        try:
            return importlib.import_module(
                "scripts.dppo_credit_support.credit_probe"
            )
        except ModuleNotFoundError:
            self.fail("credit_probe implementation is missing")

    def test_policy_credit_exposes_clamped_geometric_and_unclamped_joint_views(self):
        module = self._module()
        new = torch.tensor(
            [[[-9.0, -9.0], [-6.0, -4.0], [0.0, 1.0], [9.0, 9.0]]],
            requires_grad=True,
        )
        old = torch.tensor(
            [[[-9.0, -9.0], [-7.0, -5.0], [0.0, 0.0], [9.0, 9.0]]]
        )

        result = module.compute_policy_credit(
            new,
            old,
            advantages=torch.tensor([1.0]),
            denoising_inds=torch.tensor([9]),
            reward_start=1,
            reward_horizon=2,
            norm_adv=False,
            gamma_denoising=0.8,
            ft_denoising_steps=10,
            clip_ploss_coef=0.1,
            clip_ploss_coef_base=0.001,
            clip_ploss_coef_rate=3.0,
        )

        self.assertEqual(result["component_count"], 4)
        self.assertAlmostEqual(result["preclamp_joint_logratio"].item(), 3.0)
        self.assertAlmostEqual(result["clamped_joint_logratio"].item(), 2.0)
        self.assertAlmostEqual(result["clamped_mean_logratio"].item(), 0.5)
        self.assertAlmostEqual(
            result["geometric_ratio"].item(), math.exp(0.5), places=6
        )
        self.assertAlmostEqual(result["joint_ratio"].item(), math.exp(2.0), places=6)
        self.assertEqual(result["new_below_clamp_count"].item(), 1)
        self.assertEqual(result["old_below_clamp_count"].item(), 2)
        self.assertEqual(result["new_above_clamp_count"].item(), 0)
        self.assertEqual(result["old_above_clamp_count"].item(), 0)
        self.assertAlmostEqual(result["per_sample_policy_loss"].item(), -1.1)

    def test_policy_credit_matches_minibatch_normalization_and_denoising_discount(self):
        module = self._module()
        new = torch.zeros(2, 1, 1, requires_grad=True)
        old = torch.zeros_like(new)

        result = module.compute_policy_credit(
            new,
            old,
            advantages=torch.tensor([1.0, 3.0]),
            denoising_inds=torch.tensor([0, 9]),
            reward_start=0,
            reward_horizon=1,
            norm_adv=True,
            gamma_denoising=0.8,
            ft_denoising_steps=10,
            clip_ploss_coef=0.1,
            clip_ploss_coef_base=0.001,
            clip_ploss_coef_rate=3.0,
        )

        expected = 1.0 / math.sqrt(2.0)
        self.assertAlmostEqual(result["normalized_advantage"][0].item(), -expected)
        self.assertAlmostEqual(result["normalized_advantage"][1].item(), expected)
        self.assertAlmostEqual(result["denoising_discount"][0].item(), 0.8**9)
        self.assertAlmostEqual(result["denoising_discount"][1].item(), 1.0)
        self.assertAlmostEqual(
            result["discounted_advantage"][0].item(), -expected * 0.8**9
        )
        self.assertAlmostEqual(result["discounted_advantage"][1].item(), expected)

    def test_group_gradient_contributions_reconstruct_total_without_parameter_mutation(self):
        module = self._module()
        weight = torch.nn.Parameter(torch.tensor(2.0))
        losses = torch.stack((weight, 2 * weight, -weight, 4 * weight))
        before = weight.detach().clone()

        result = module.summarize_group_gradient_contributions(
            losses,
            (weight,),
            ("reached", "reached", "timeout", "timeout"),
        )

        self.assertAlmostEqual(result["overall"]["gradient_norm"], 1.5)
        self.assertAlmostEqual(result["groups"]["reached"]["gradient_norm"], 0.75)
        self.assertAlmostEqual(result["groups"]["timeout"]["gradient_norm"], 0.75)
        self.assertAlmostEqual(
            result["groups"]["reached"]["cosine_with_overall"], 1.0
        )
        self.assertAlmostEqual(
            result["groups"]["timeout"]["cosine_with_overall"], 1.0
        )
        self.assertLess(result["gradient_reconstruction_max_abs"], 1e-7)
        self.assertTrue(torch.equal(weight.detach(), before))
        self.assertIsNone(weight.grad)

    def test_shadow_schedule_matches_real_rollout_buffer_order_and_denoising_indices(self):
        module = self._module()
        from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.rollout_buffer import (
            OnPolicyRolloutBuffer,
        )

        buffer = OnPolicyRolloutBuffer(128)
        for index in range(128):
            buffer.add(
                condition={
                    "state": torch.full((576,), float(index)),
                    "global_cond": torch.full((576,), float(index)),
                },
                chain=torch.full((11, 16, 11), float(index)),
                old_logprobs=torch.zeros(10, 16, 11),
                reward=0.0,
                value=0.0,
                next_value=0.0,
                terminated=True,
                truncated=False,
            )
        buffer.finalize()
        real_generator = torch.Generator().manual_seed(12345)
        shadow_generator = torch.Generator().manual_seed(12345)

        real_batch = next(
            buffer.batches(128, generator=real_generator, device=torch.device("cpu"))
        )
        if not hasattr(module, "make_batch_schedule"):
            self.fail("make_batch_schedule implementation is missing")
        schedule = module.make_batch_schedule(
            128,
            batch_size=128,
            ft_denoising_steps=10,
            generator=shadow_generator,
        )
        shadow_indices, shadow_denoising = schedule[0]

        self.assertTrue(
            torch.equal(real_batch.chains_prev[:, 0, 0].long(), shadow_indices)
        )
        self.assertTrue(torch.equal(real_batch.denoising_inds, shadow_denoising))

    def test_probe_minibatch_measures_real_runtime_gradients_without_mutation(self):
        module = self._module()
        if not hasattr(module, "probe_minibatch"):
            self.fail("probe_minibatch implementation is missing")

        class TinyRuntime:
            norm_adv = False
            gamma_denoising = 0.8
            ft_denoising_steps = 10
            clip_ploss_coef = 0.1
            clip_ploss_coef_base = 0.001
            clip_ploss_coef_rate = 3.0
            clip_advantage_lower_quantile = 0.0
            clip_advantage_upper_quantile = 1.0

            def __init__(self, weight):
                self.weight = weight

            def get_logprobs_subsample(
                self, condition, chains_prev, chains_next, denoising_inds
            ):
                coefficients = condition["state"][:, :1].reshape(-1, 1, 1)
                return self.weight * coefficients.expand(-1, 16, 11)

        weight = torch.nn.Parameter(torch.tensor(0.25))
        batch = SimpleNamespace(
            condition={"state": torch.tensor([[1.0], [-1.0]])},
            chains_prev=torch.zeros(2, 16, 11),
            chains_next=torch.zeros(2, 16, 11),
            denoising_inds=torch.tensor([9, 9]),
            advantages=torch.tensor([1.0, 2.0]),
            oldlogprobs=torch.zeros(2, 16, 11),
        )
        before = weight.detach().clone()

        result = module.probe_minibatch(
            TinyRuntime(weight),
            batch,
            actor_parameters=(weight,),
            reward_start=2,
            reward_horizon=4,
            outcome_labels=("reached", "timeout"),
            role_labels=("anticipatory", "dense_recovery"),
        )

        self.assertEqual(result["credit"]["component_count"], 44)
        self.assertIn("reached", result["gradient_by_outcome"]["groups"])
        self.assertIn("dense_recovery", result["gradient_by_role"]["groups"])
        self.assertTrue(torch.equal(weight.detach(), before))
        self.assertIsNone(weight.grad)


if __name__ == "__main__":
    unittest.main()
