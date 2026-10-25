import random
import unittest

import numpy as np
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.plan_probe import (
    compare_probe_actions, sample_probe_action,
)


class _Sampler:
    def sample(self, condition, *, deterministic):
        if deterministic:
            raise AssertionError("probe must use the same stochastic policy path")
        action = torch.randn(1, 8, 11) + random.random() + np.random.random()
        return type("Rollout", (), {
            "action": type("Action", (), {"policy": action})(),
        })()


class R1PlanProbeTest(unittest.TestCase):
    def test_probe_repeats_without_advancing_training_rng(self):
        condition = {"state": torch.zeros(1, 576),
                     "global_cond": torch.zeros(1, 576)}
        torch.manual_seed(12)
        random.seed(12)
        np.random.seed(12)
        saved = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
        first = sample_probe_action(_Sampler(), condition, seed=99, device="cpu")
        second = sample_probe_action(_Sampler(), condition, seed=99, device="cpu")
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(saved[0], torch.get_rng_state()))
        self.assertEqual(saved[1], random.getstate())
        self.assertTrue(np.array_equal(saved[2][1], np.random.get_state()[1]))
        self.assertTrue(all(value == 0. for value in
                            compare_probe_actions(first, second).values()))


if __name__ == "__main__":
    unittest.main()
