from __future__ import annotations

import unittest

import torch
from torch import nn

from diffusion_policy_3d.rl.reinflow_fm.condition_adapter import FMConditionBatch
from diffusion_policy_3d.rl.reinflow_fm.policy_bridge import (
    IDP3FlowPolicyBridge,
    condition_to_official_dict,
)


class _VelocityPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(11, 11, bias=False)
        self.horizon = 16
        self.action_dim = 11
        self.sigma_min = 0.005
        self.fm_t_scale = 50
        self.cfg_lambda = 1.5
        self.obs_feature_dim = 4
        self.calls = []

    def _compute_cfg_velocity(
        self, action, t_input, global_cond, global_cond_uncond, local_cond
    ):
        self.calls.append(
            (t_input.detach().clone(), global_cond, global_cond_uncond, local_cond)
        )
        return action * 0.2 + t_input[:, None, None] * 0.01


def _condition(batch_size=2):
    return FMConditionBatch(
        cond=torch.ones(batch_size, 12),
        uncond=torch.zeros(batch_size, 12),
        noise_cond=torch.arange(batch_size * 4, dtype=torch.float32).reshape(batch_size, 4),
        raw_obs_sha256=tuple("a" * 64 for _ in range(batch_size)),
        encoder_sha256="b" * 64,
        normalizer_sha256="c" * 64,
        n_obs_steps=3,
    )


class PolicyBridgeTests(unittest.TestCase):
    def test_bridge_maps_official_unit_time_to_native_sigma_endpoint(self):
        policy = _VelocityPolicy()
        bridge = IDP3FlowPolicyBridge(policy, lora_bundle=None, lora_enabled=False)
        condition = _condition()
        official_cond = condition_to_official_dict(condition)
        action = torch.ones(2, 16, 11)
        official_time = torch.tensor([0.0, 0.25])

        velocity, time_embedding, condition_embedding = bridge(
            action, official_time, official_cond, output_embedding=True
        )

        expected_native_time = official_time * 0.995
        self.assertTrue(
            torch.allclose(policy.calls[0][0], expected_native_time * 50.0)
        )
        self.assertIs(policy.calls[0][1], official_cond["global_cond"])
        self.assertIs(policy.calls[0][2], official_cond["global_cond_uncond"])
        native_velocity = action * 0.2 + (expected_native_time * 50.0)[:, None, None] * 0.01
        self.assertTrue(torch.allclose(velocity, native_velocity * 0.995))
        self.assertTrue(torch.allclose(time_embedding[:, 0], expected_native_time))
        self.assertIs(condition_embedding, official_cond["noise_cond"])

    def test_four_official_steps_match_native_euler_endpoint_exactly(self):
        policy = _VelocityPolicy()
        bridge = IDP3FlowPolicyBridge(policy, lora_bundle=None, lora_enabled=False)
        cond = condition_to_official_dict(_condition(batch_size=1))
        initial = torch.randn(1, 16, 11, generator=torch.Generator().manual_seed(7))

        native = initial.clone()
        native_dt = 0.995 / 4
        for index in range(4):
            native_t = torch.tensor([index * native_dt])
            native_velocity = policy._compute_cfg_velocity(
                native, native_t * 50.0, cond["global_cond"], cond["global_cond_uncond"], None
            )
            native = native + native_velocity * native_dt

        adapted = initial.clone()
        for index in range(4):
            official_t = torch.tensor([index / 4])
            adapted = adapted + bridge(adapted, official_t, cond) / 4

        self.assertTrue(torch.allclose(native, adapted, atol=1e-7, rtol=1e-7))


if __name__ == "__main__":
    unittest.main()
