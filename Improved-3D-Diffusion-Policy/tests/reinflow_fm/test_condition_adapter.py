from __future__ import annotations

import unittest

import torch
from torch import nn

from diffusion_policy_3d.rl.reinflow_fm.condition_adapter import (
    ConditionEncodingError,
    encode_conditions,
)


class _IdentityNormalizer(nn.Module):
    def normalize(self, obs):
        return {key: value.clone() for key, value in obs.items()}


class _RecordingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("identity", torch.tensor([7.0]))
        self.inputs = []

    def forward(self, obs):
        self.inputs.append({key: value.clone() for key, value in obs.items()})
        state = obs["agent_pos"]
        point = obs["point_cloud"]
        return torch.stack(
            (
                state[:, 3],
                state[:, 4],
                point[..., 0].mean(dim=1),
                point[..., 1].mean(dim=1),
            ),
            dim=1,
        )


class _Policy:
    def __init__(self):
        self.normalizer = _IdentityNormalizer()
        self.obs_encoder = _RecordingEncoder()
        self.n_obs_steps = 3
        self.obs_feature_dim = 4
        self.cfg_lambda = 1.5
        self.use_pc_color = False
        self.device = torch.device("cpu")


class ConditionAdapterTests(unittest.TestCase):
    def _obs(self):
        agent = torch.zeros(2, 3, 9, dtype=torch.float32)
        agent[:, :, 3:7] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        points = torch.arange(2 * 3 * 5 * 6, dtype=torch.float32).reshape(2, 3, 5, 6)
        return {"agent_pos": agent, "point_cloud": points}

    def test_condition_matches_native_cfg_goal_drop_and_xyz_crop(self):
        policy = _Policy()
        obs = self._obs()
        original_goal = obs["agent_pos"][:, :, 3:7].clone()

        batch = encode_conditions(policy, obs, normalizer_sha256="a" * 64)

        self.assertEqual(tuple(batch.cond.shape), (2, 12))
        self.assertEqual(tuple(batch.uncond.shape), (2, 12))
        self.assertEqual(tuple(batch.noise_cond.shape), (2, 4))
        self.assertTrue(torch.equal(obs["agent_pos"][:, :, 3:7], original_goal))
        self.assertEqual(policy.obs_encoder.inputs[0]["point_cloud"].shape[-1], 3)
        self.assertTrue(torch.count_nonzero(batch.cond - batch.uncond) > 0)
        self.assertEqual(batch.raw_obs_sha256[0], encode_conditions(
            _Policy(), obs, normalizer_sha256="a" * 64
        ).raw_obs_sha256[0])

    def test_condition_fingerprint_changes_when_raw_observation_changes(self):
        obs = self._obs()
        first = encode_conditions(_Policy(), obs, normalizer_sha256="a" * 64)
        obs["point_cloud"][0, 0, 0, 0] += 1
        second = encode_conditions(_Policy(), obs, normalizer_sha256="a" * 64)
        self.assertNotEqual(first.raw_obs_sha256[0], second.raw_obs_sha256[0])
        self.assertEqual(first.raw_obs_sha256[1], second.raw_obs_sha256[1])

    def test_condition_rejects_history_or_identity_mismatch(self):
        obs = self._obs()
        with self.assertRaises(ConditionEncodingError):
            encode_conditions(_Policy(), {"agent_pos": obs["agent_pos"][:, :2], "point_cloud": obs["point_cloud"][:, :2]}, normalizer_sha256="a" * 64)
        with self.assertRaisesRegex(ConditionEncodingError, "normalizer_sha256"):
            encode_conditions(_Policy(), obs, normalizer_sha256="not-a-hash")


if __name__ == "__main__":
    unittest.main()
