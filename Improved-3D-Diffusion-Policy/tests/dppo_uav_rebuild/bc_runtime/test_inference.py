from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.bc_runtime.inference import (
    NativeInferenceError,
    ObservationBridge,
    run_native_inference,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import (
    ActionSpec,
    ObservationSpec,
)


def test_observation_bridge_selects_chronological_latest_three_and_xyz():
    spec = ObservationSpec.e5_route_a()
    bridge = ObservationBridge(spec)
    agent_pos = np.zeros((8, 9), dtype=np.float32)
    point_cloud = np.zeros((8, 4096, 6), dtype=np.float32)
    for frame in range(8):
        agent_pos[frame, 0] = frame
        point_cloud[frame, :, 0] = frame
        point_cloud[frame, :, 3:] = 99.0

    result = bridge.to_policy_observation(
        {"agent_pos": agent_pos, "point_cloud": point_cloud},
        device=torch.device("cpu"),
    )

    assert result["agent_pos"].shape == (1, 3, 9)
    assert result["point_cloud"].shape == (1, 3, 4096, 3)
    torch.testing.assert_close(result["agent_pos"][0, :, 0], torch.tensor([5.0, 6.0, 7.0]))
    torch.testing.assert_close(result["point_cloud"][0, :, 0, 0], torch.tensor([5.0, 6.0, 7.0]))


def test_observation_bridge_rejects_a_short_or_ambiguous_history():
    bridge = ObservationBridge(ObservationSpec.e5_route_a())

    with pytest.raises(NativeInferenceError, match="history"):
        bridge.to_policy_observation(
            {
                "agent_pos": np.zeros((2, 9), dtype=np.float32),
                "point_cloud": np.zeros((2, 4096, 3), dtype=np.float32),
            },
            device=torch.device("cpu"),
        )


class _FakeNativePolicy:
    def __init__(self):
        self.noise_scheduler = SimpleNamespace(
            timesteps=torch.tensor([45, 40, 35, 30, 25, 20, 15, 10, 5, 0])
        )
        self.seen_shapes = None

    def predict_action(self, observation):
        self.seen_shapes = {key: tuple(value.shape) for key, value in observation.items()}
        full = torch.arange(16 * 11, dtype=torch.float32).reshape(1, 16, 11)
        return {"action_pred": full, "action": full[:, 2:10, :]}


def test_native_inference_returns_all_named_action_protocol_views():
    policy = _FakeNativePolicy()
    runtime = SimpleNamespace(policy=policy, device=torch.device("cpu"))
    observation = {
        "agent_pos": np.zeros((8, 9), dtype=np.float32),
        "point_cloud": np.zeros((8, 4096, 3), dtype=np.float32),
    }

    result = run_native_inference(
        runtime=runtime,
        environment_observation=observation,
        observation_spec=ObservationSpec.e5_route_a(),
        action_spec=ActionSpec.e5(),
        seed=12345,
    )

    assert policy.seen_shapes == {
        "agent_pos": (1, 3, 9),
        "point_cloud": (1, 3, 4096, 3),
    }
    assert result.timesteps == (45, 40, 35, 30, 25, 20, 15, 10, 5, 0)
    assert result.actions.full.shape == (1, 16, 11)
    assert result.actions.policy.shape == (1, 8, 11)
    assert result.actions.deployment_executed.shape == (1, 2, 11)
    assert result.actions.rollout_executed.shape == (1, 4, 11)
