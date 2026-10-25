from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.bc_runtime.behavior_fingerprint import (
    BehaviorFingerprintError,
    fingerprint_runtime,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import (
    ActionSpec,
    ObservationSpec,
)


class _Policy:
    def __init__(self, drift=False):
        self.noise_scheduler = SimpleNamespace(
            timesteps=torch.tensor([45, 40, 35, 30, 25, 20, 15, 10, 5, 0])
        )
        self.calls = 0
        self.drift = drift

    def predict_action(self, observation):
        full = torch.arange(16 * 11, dtype=torch.float32).reshape(1, 16, 11)
        if self.drift:
            full = full + self.calls
        self.calls += 1
        return {"action_pred": full, "action": full[:, 2:10, :]}


def _observation():
    return {
        "agent_pos": np.arange(8 * 9, dtype=np.float32).reshape(8, 9),
        "point_cloud": np.zeros((8, 4096, 3), dtype=np.float32),
    }


def _runtime(policy):
    return SimpleNamespace(
        policy=policy,
        device=torch.device("cpu"),
        identity=SimpleNamespace(content_hash="a" * 64),
    )


def test_behavior_fingerprint_is_stable_and_records_all_action_views():
    fingerprint = fingerprint_runtime(
        runtime=_runtime(_Policy()),
        environment_observation=_observation(),
        observation_spec=ObservationSpec.e5_route_a(),
        action_spec=ActionSpec.e5(),
        seed=12345,
    )

    assert fingerprint.bc_identity_hash == "a" * 64
    assert fingerprint.timesteps == (45, 40, 35, 30, 25, 20, 15, 10, 5, 0)
    assert fingerprint.action_shapes == {
        "full": (1, 16, 11),
        "policy": (1, 8, 11),
        "deployment_executed": (1, 2, 11),
        "rollout_executed": (1, 4, 11),
    }
    assert set(fingerprint.action_sha256) == set(fingerprint.action_shapes)
    assert all(len(value) == 64 for value in fingerprint.action_sha256.values())


def test_behavior_fingerprint_rejects_same_seed_output_drift():
    with pytest.raises(BehaviorFingerprintError, match="not deterministic"):
        fingerprint_runtime(
            runtime=_runtime(_Policy(drift=True)),
            environment_observation=_observation(),
            observation_spec=ObservationSpec.e5_route_a(),
            action_spec=ActionSpec.e5(),
            seed=12345,
        )
