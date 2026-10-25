from copy import deepcopy

import pytest

from diffusion_policy_3d.rl.dppo_uav_rebuild.bc_runtime.checkpoint_contract import (
    BCIdentity,
    BCIdentityError,
    EXPECTED_E5_CHECKPOINT_SHA256,
    EXPECTED_E5_NORMALIZER_SHA256,
)


def _e5_config():
    shape_meta = {
        "obs": {
            "point_cloud": {"shape": [4096, 6], "type": "point_cloud"},
            "agent_pos": {"shape": [9], "type": "low_dim"},
        },
        "action": {"shape": [11]},
    }
    return {
        "_target_": "diffusion_policy_3d.workspace.idp3_workspace.iDP3Workspace",
        "n_obs_steps": 3,
        "horizon": 16,
        "n_action_steps": 8,
        "shape_meta": shape_meta,
        "policy": {
            "_target_": (
                "diffusion_policy_3d.policy.diffusion_pointcloud_policy."
                "DiffusionPointcloudPolicy"
            ),
            "horizon": 16,
            "n_action_steps": 8,
            "n_obs_steps": 3,
            "num_inference_steps": 10,
            "obs_as_global_cond": True,
            "shape_meta": deepcopy(shape_meta),
            "use_pc_color": False,
            "pointnet_type": "multi_stage_pointnet",
            "pointcloud_encoder_cfg": {
                "in_channels": 3,
                "out_channels": 128,
                "num_points": 4096,
            },
            "noise_scheduler": {
                "_target_": "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
                "num_train_timesteps": 50,
                "prediction_type": "sample",
            },
        },
    }


def _identity(config=None, **overrides):
    values = {
        "resolved_config": _e5_config() if config is None else config,
        "checkpoint_relpath": (
            "data/outputs/2026.08.16/08.53.26_e5_lambda0p5/"
            "checkpoints/latest.ckpt"
        ),
        "checkpoint_sha256": EXPECTED_E5_CHECKPOINT_SHA256,
        "normalizer_sha256": EXPECTED_E5_NORMALIZER_SHA256,
        "weight_source": "model",
        "evidence_caveats": (
            "historical Gazebo deployment log is unavailable; code path uses model",
        ),
    }
    values.update(overrides)
    return BCIdentity.from_resolved_config(**values)


def test_e5_identity_freezes_the_native_checkpoint_contract():
    identity = _identity()

    assert identity.weight_source == "model"
    assert identity.policy_target.endswith("DiffusionPointcloudPolicy")
    assert identity.scheduler_target.endswith("DDIMScheduler")
    assert identity.prediction_type == "sample"
    assert (identity.num_train_timesteps, identity.num_inference_steps) == (50, 10)
    assert identity.observation_spec.policy_condition_dim == 576
    assert identity.action_spec.full_horizon == 16
    assert identity.action_spec.rollout_exec_horizon == 4
    assert len(identity.config_sha256) == 64
    assert len(identity.content_hash) == 64


@pytest.mark.parametrize(
    "path,value,match",
    [
        (
            ("policy", "_target_"),
            "diffusion_policy_3d.policy.phase1a_policy.Phase1aPolicy",
            "policy target",
        ),
        (
            ("policy", "noise_scheduler", "_target_"),
            "diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
            "scheduler target",
        ),
        (
            ("policy", "noise_scheduler", "prediction_type"),
            "epsilon",
            "prediction_type",
        ),
        (("n_obs_steps",), 8, "n_obs_steps"),
    ],
)
def test_e5_identity_rejects_behavior_drift(path, value, match):
    config = _e5_config()
    cursor = config
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value

    with pytest.raises(BCIdentityError, match=match):
        _identity(config=config)


def test_e5_identity_rejects_implicit_or_ema_weight_source():
    with pytest.raises(BCIdentityError, match="weight_source"):
        _identity(weight_source="ema_model")


def test_e5_identity_rejects_checkpoint_path_escape():
    with pytest.raises(BCIdentityError, match="checkpoint_relpath"):
        _identity(checkpoint_relpath="../latest.ckpt")
