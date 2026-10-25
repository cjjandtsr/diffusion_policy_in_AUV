from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.bc_runtime.checkpoint_contract import (
    EXPECTED_E5_CHECKPOINT_SHA256,
    EXPECTED_E5_NORMALIZER_SHA256,
    EXPECTED_POLICY_TARGET,
    EXPECTED_SCHEDULER_TARGET,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.bc_runtime import native_loader


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


class _FakeNormalizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.ones(1))


class _FakePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.normalizer = _FakeNormalizer()
        self.noise_scheduler = SimpleNamespace(
            config=SimpleNamespace(num_train_timesteps=50, prediction_type="sample")
        )
        self.num_inference_steps = 10
        self.n_obs_steps = 3
        self.n_action_steps = 8
        self.horizon = 16
        self.action_dim = 11
        self.obs_feature_dim = 192
        self.obs_as_global_cond = True


def _patch_valid_payload(monkeypatch, policy, payload):
    monkeypatch.setattr(native_loader, "_sha256_file", lambda _: EXPECTED_E5_CHECKPOINT_SHA256)
    monkeypatch.setattr(
        native_loader,
        "hash_normalizer_state_dict",
        lambda _: EXPECTED_E5_NORMALIZER_SHA256,
    )
    monkeypatch.setattr(native_loader.torch, "load", lambda *args, **kwargs: payload)
    monkeypatch.setattr(native_loader.hydra.utils, "instantiate", lambda _: policy)
    monkeypatch.setattr(
        native_loader,
        "_qualified_name",
        lambda value: (
            EXPECTED_POLICY_TARGET
            if value is policy
            else EXPECTED_SCHEDULER_TARGET
        ),
    )


def test_native_loader_uses_native_config_and_freezes_the_loaded_base(tmp_path, monkeypatch):
    checkpoint = tmp_path / "latest.ckpt"
    checkpoint.write_bytes(b"fixture")
    policy = _FakePolicy()
    payload = {"cfg": _e5_config(), "state_dicts": {"model": policy.state_dict()}}
    _patch_valid_payload(monkeypatch, policy, payload)

    runtime = native_loader.load_native_e5(
        checkpoint_path=checkpoint,
        checkpoint_relpath=(
            "data/outputs/2026.08.16/08.53.26_e5_lambda0p5/"
            "checkpoints/latest.ckpt"
        ),
        device="cpu",
    )

    assert runtime.identity.weight_source == "model"
    assert runtime.policy is policy
    assert runtime.device == torch.device("cpu")
    assert not policy.training
    assert all(not parameter.requires_grad for parameter in policy.parameters())
    assert all(tensor.device.type == "cpu" for tensor in policy.normalizer.state_dict().values())


def test_native_loader_rejects_missing_or_unexpected_weight_keys(tmp_path, monkeypatch):
    checkpoint = tmp_path / "latest.ckpt"
    checkpoint.write_bytes(b"fixture")
    policy = _FakePolicy()
    payload = {
        "cfg": _e5_config(),
        "state_dicts": {"model": {"unexpected": torch.ones(1)}},
    }
    _patch_valid_payload(monkeypatch, policy, payload)

    with pytest.raises(native_loader.NativeE5LoadError, match="missing=.*unexpected="):
        native_loader.load_native_e5(
            checkpoint_path=checkpoint,
            checkpoint_relpath="data/outputs/x/checkpoints/latest.ckpt",
            device="cpu",
        )


def test_native_loader_rejects_checkpoint_bytes_before_deserialization(tmp_path, monkeypatch):
    checkpoint = tmp_path / "latest.ckpt"
    checkpoint.write_bytes(b"wrong checkpoint")
    called = {"torch_load": False}

    def _unexpected_load(*args, **kwargs):
        called["torch_load"] = True
        raise AssertionError("torch.load must not run for an identity mismatch")

    monkeypatch.setattr(native_loader.torch, "load", _unexpected_load)

    with pytest.raises(native_loader.NativeE5LoadError, match="checkpoint SHA256 mismatch"):
        native_loader.load_native_e5(
            checkpoint_path=checkpoint,
            checkpoint_relpath="data/outputs/x/checkpoints/latest.ckpt",
            device="cpu",
        )

    assert called["torch_load"] is False
