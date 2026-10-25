"""Point-cloud generation and immutable observation provenance gates."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from diffusion_policy_3d.model.common.synth_pointcloud import (
    DepthCameraIntrinsics,
    SynthPointCloud,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ObservationSpec


def _provenance():
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments import observation_provenance
    return observation_provenance


def _empty_world(tmp_path):
    path = tmp_path / "empty.world.obstacles.json"
    path.write_text(json.dumps({"spawn_world": [0, 0], "obstacles": []}), encoding="utf-8")
    return path


def test_generator_hash_and_metadata_come_from_the_same_call(tmp_path):
    synth = SynthPointCloud(
        str(_empty_world(tmp_path)), device="cpu",
        cam_intrinsics=DepthCameraIntrinsics(width=16, height=12),
        noise_cfg={"depth_noise_sigma": 0.0, "dropout_rate": 0.0},
    )
    points, meta = synth.generate_with_metadata(
        drone_pos_world=np.array([0.0, 0.0, 2.0]),
        drone_yaw=0.0,
        episode_identity_hash="a" * 64,
        decision_index=7,
        rng_seed=19,
    )
    assert points.shape == (4096, 3) and points.dtype == np.float32
    assert _provenance().canonical_array_hash(points) == meta.raw_array_hash
    assert meta.total_rays == 16 * 12
    assert 0 <= meta.valid_hit_rays <= meta.total_rays
    assert meta.decision_index == 7
    assert meta.episode_identity_hash == "a" * 64


def test_all_no_ray_far_plane_is_explicit_and_rejected(tmp_path):
    synth = SynthPointCloud(
        str(_empty_world(tmp_path)), device="cpu",
        cam_intrinsics=DepthCameraIntrinsics(width=8, height=6),
        noise_cfg={"depth_noise_sigma": 0.0, "dropout_rate": 0.0},
    )
    with pytest.warns(RuntimeWarning, match="no ray hit"):
        points, meta = synth.generate_with_metadata(
            np.array([0.0, 0.0, 2.0]), 0.0,
            drone_pitch=-math.pi / 2,
            episode_identity_hash="b" * 64,
            decision_index=0,
            rng_seed=3,
        )
    assert meta.used_far_plane_fallback and meta.valid_hit_rays == 0
    with pytest.raises(_provenance().PointCloudRejected, match="all_no_ray"):
        _provenance().require_pointcloud(points, meta, max_no_ray_hit_fraction=0.9)


def test_no_return_fallback_spans_camera_frustum(tmp_path):
    synth = SynthPointCloud(
        str(_empty_world(tmp_path)), device="cpu",
        cam_intrinsics=DepthCameraIntrinsics(width=16, height=12),
        noise_cfg={"depth_noise_sigma": 0.0, "dropout_rate": 0.0},
    )
    with pytest.warns(RuntimeWarning, match="no ray hit"):
        points, meta = synth.generate_with_metadata(
            np.array([0.0, 0.0, 6.0]), 0.0,
            episode_identity_hash="f" * 64,
            decision_index=0,
            rng_seed=3,
        )
    assert meta.used_far_plane_fallback and meta.valid_hit_rays == 0
    assert np.all(np.ptp(points, axis=0) > 1.0)


def test_verified_no_return_can_be_explicitly_allowed(tmp_path):
    synth = SynthPointCloud(
        str(_empty_world(tmp_path)), device="cpu",
        cam_intrinsics=DepthCameraIntrinsics(width=16, height=12),
        noise_cfg={"depth_noise_sigma": 0.0, "dropout_rate": 0.0},
    )
    with pytest.warns(RuntimeWarning, match="no ray hit"):
        points, meta = synth.generate_with_metadata(
            np.array([0.0, 0.0, 6.0]), 0.0,
            episode_identity_hash="f" * 64,
            decision_index=0,
            rng_seed=3,
        )
    _provenance().require_pointcloud(
        points,
        meta,
        max_no_ray_hit_fraction=0.9,
        reject_all_no_ray_hit=False,
    )


class _Normalizer:
    def __init__(self):
        self.scale = torch.tensor(1.0)

    def state_dict(self):
        return {"scale": self.scale}

    def normalize(self, observation):
        return {key: value * self.scale for key, value in observation.items()}


class _Encoder(torch.nn.Module):
    def forward(self, observation):
        marker = observation["agent_pos"][:, :1]
        return marker.repeat(1, 192)


class _Policy:
    def __init__(self):
        self.normalizer = _Normalizer()
        self.obs_encoder = _Encoder()
        self.use_pc_color = False
        self.n_obs_steps = 3
        self.obs_feature_dim = 192


def _adapter_inputs():
    provenance = _provenance()
    point_cloud = np.ones((8, 4096, 3), np.float32)
    agent_pos = np.zeros((8, 9), np.float32)
    agent_pos[:, 0] = np.arange(8)
    metas = tuple(
        provenance.PointCloudGenerationMeta(
            total_rays=100,
            valid_hit_rays=90,
            no_ray_hit_fraction=0.1,
            used_far_plane_fallback=False,
            finite=True,
            episode_identity_hash="c" * 64,
            decision_index=index,
            raw_array_hash=provenance.canonical_array_hash(point_cloud[index]),
        )
        for index in range(8)
    )
    return {"point_cloud": point_cloud, "agent_pos": agent_pos}, metas


def test_adapter_builds_and_verifies_all_provenance_layers():
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.observation_adapter import (
        E5ObservationAdapter,
    )
    observation, metas = _adapter_inputs()
    adapter = E5ObservationAdapter(_Policy(), torch.device("cpu"), ObservationSpec.e5_route_a())
    encoded = adapter.encode(
        observation, metas, episode_identity_hash="c" * 64, decision_index=7
    )
    assert encoded.condition["global_cond"].shape == (1, 576)
    adapter.verify(encoded)
    assert encoded.provenance.raw_frame_hash == metas[-1].raw_array_hash


@pytest.mark.parametrize("attack", ["raw", "order", "normalizer", "condition"])
def test_provenance_attacks_fail_closed_before_consumption(attack):
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.observation_adapter import (
        E5ObservationAdapter,
    )
    provenance = _provenance()
    observation, metas = _adapter_inputs()
    policy = _Policy()
    adapter = E5ObservationAdapter(policy, torch.device("cpu"))
    if attack == "raw":
        observation["point_cloud"][-1, 0, 0] = np.nextafter(np.float32(1), np.float32(2))
        with pytest.raises(provenance.ObservationIdentityMismatch, match="raw_frame_hash"):
            adapter.encode(observation, metas, episode_identity_hash="c" * 64, decision_index=7)
        return
    if attack == "order":
        metas = metas[:-2] + (metas[-1], metas[-2])
        with pytest.raises(provenance.ObservationIdentityMismatch, match="history_order"):
            adapter.encode(observation, metas, episode_identity_hash="c" * 64, decision_index=7)
        return
    encoded = adapter.encode(observation, metas, episode_identity_hash="c" * 64, decision_index=7)
    if attack == "normalizer":
        policy.normalizer.scale = torch.tensor(2.0)
        with pytest.raises(provenance.ObservationIdentityMismatch, match="normalizer_identity"):
            adapter.verify(encoded)
    else:
        encoded.condition["global_cond"][0, 0] = 123.0
        with pytest.raises(provenance.ObservationIdentityMismatch, match="policy_input_hash"):
            adapter.verify(encoded)


@pytest.mark.parametrize(
    "points,reason",
    [
        (np.ones((4095, 3), np.float32), "point_shape"),
        (np.ones((4097, 3), np.float32), "point_shape"),
        (np.full((4096, 3), np.nan, np.float32), "non_finite"),
        (np.zeros((4096, 3), np.float32), "all_zero"),
        (np.full((4096, 3), 100.0, np.float32), "distance_exceeds_limit"),
    ],
)
def test_pointcloud_rejection_reason_is_specific(points, reason):
    provenance = _provenance()
    meta = provenance.PointCloudGenerationMeta(
        total_rays=10, valid_hit_rays=9, no_ray_hit_fraction=0.1,
        used_far_plane_fallback=False, finite=bool(np.isfinite(points).all()),
        episode_identity_hash="d" * 64, decision_index=1,
        raw_array_hash=provenance.canonical_array_hash(points),
    )
    with pytest.raises(provenance.PointCloudRejected, match=reason):
        provenance.require_pointcloud(points, meta, max_no_ray_hit_fraction=0.5)


def test_metadata_rejects_type_drift_and_nonfinite_fraction():
    provenance = _provenance()
    common = dict(
        valid_hit_rays=9, no_ray_hit_fraction=0.1,
        used_far_plane_fallback=False, finite=True,
        episode_identity_hash="d" * 64, decision_index=1,
        raw_array_hash="e" * 64,
    )
    with pytest.raises(ValueError, match="total_rays"):
        provenance.PointCloudGenerationMeta(total_rays=10.5, **common)
    with pytest.raises(TypeError, match="used_far_plane_fallback"):
        provenance.PointCloudGenerationMeta(
            total_rays=10, **{**common, "used_far_plane_fallback": 0}
        )
    with pytest.raises(ValueError, match="no_ray_hit_fraction"):
        provenance.PointCloudGenerationMeta(
            total_rays=10, **{**common, "no_ray_hit_fraction": float("nan")}
        )
