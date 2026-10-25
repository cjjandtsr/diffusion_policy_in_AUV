import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.conditioning import (
    encode_e5_condition,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ObservationSpec


class _IdentityNormalizer:
    def normalize(self, observation):
        return {key: value.clone() for key, value in observation.items()}


class _FrameEncoder(torch.nn.Module):
    def forward(self, observation):
        marker = observation["agent_pos"][:, :1]
        return marker.repeat(1, 192)


class _Policy:
    def __init__(self):
        self.normalizer = _IdentityNormalizer()
        self.obs_encoder = _FrameEncoder()
        self.use_pc_color = False
        self.n_obs_steps = 3
        self.obs_feature_dim = 192


def test_e5_condition_uses_three_frames_and_preserves_chronological_order():
    agent_pos = torch.zeros((2, 3, 9), dtype=torch.float32)
    point_cloud = torch.zeros((2, 3, 4096, 3), dtype=torch.float32)
    agent_pos[:, 0, 0] = 5
    agent_pos[:, 1, 0] = 6
    agent_pos[:, 2, 0] = 7

    cond = encode_e5_condition(
        policy=_Policy(),
        policy_observation={"agent_pos": agent_pos, "point_cloud": point_cloud},
        spec=ObservationSpec.e5_route_a(),
    )

    assert set(cond) == {"state", "global_cond"}
    assert cond["global_cond"].shape == (2, 576)
    torch.testing.assert_close(cond["state"], cond["global_cond"])
    assert torch.equal(cond["global_cond"][:, 0], torch.tensor([5.0, 5.0]))
    assert torch.equal(cond["global_cond"][:, 192], torch.tensor([6.0, 6.0]))
    assert torch.equal(cond["global_cond"][:, 384], torch.tensor([7.0, 7.0]))
    assert cond["global_cond"].requires_grad is False
