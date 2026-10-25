from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from diffusion_policy_3d.rl.reinflow_fm.checkpoint_adapter import (
    NativeFMLoadError,
    load_native_fm,
)
from diffusion_policy_3d.rl.reinflow_fm.checkpoint_contract import (
    FMCheckpointIdentity,
    FMIdentityError,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.ones(1, dtype=torch.float32))

    def normalize(self, value):
        return value


class _FakePhase1aPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.normalizer = _Normalizer()
        self.obs_encoder = nn.Linear(1, 1, bias=False)
        self.num_fm_steps = 4
        self.sigma_min = 0.005
        self.fm_t_scale = 50
        self.cfg_lambda = 1.5
        self.horizon = 16
        self.action_dim = 11
        self.n_obs_steps = 3
        self.n_action_steps = 8
        self.use_pc_color = False

    def _compute_cfg_velocity(self, *args, **kwargs):
        return args[0]

    def predict_action(self, obs_dict):
        anchor_idx = None
        return {"anchor_idx": anchor_idx, "obs": obs_dict}


def _qualified_name(value) -> str:
    return f"{type(value).__module__}.{type(value).__name__}"


class CheckpointContractTests(unittest.TestCase):
    def _identity_kwargs(self):
        return dict(
            checkpoint_sha256="1" * 64,
            policy_target="project.Phase1aPolicyFM",
            policy_source_sha256="2" * 64,
            normalizer_sha256="3" * 64,
            weight_source="model",
            num_fm_steps=4,
            sigma_min=0.005,
            fm_t_scale=50.0,
            cfg_lambda=1.5,
            anchor_enabled=False,
            horizon_steps=16,
            action_dim=11,
            n_obs_steps=3,
            n_action_steps=8,
            execution_slice=(2, 10),
        )

    def test_identity_accepts_the_required_four_step_fp32_contract(self):
        identity = FMCheckpointIdentity(**self._identity_kwargs())
        self.assertEqual(len(identity.content_hash), 64)

    def test_identity_rejects_drift_in_fixed_fm_values(self):
        for field, bad_value in (
            ("num_fm_steps", 5),
            ("sigma_min", 0.01),
            ("fm_t_scale", 1.0),
            ("cfg_lambda", 0.0),
            ("anchor_enabled", True),
            ("horizon_steps", 15),
            ("action_dim", 4),
        ):
            kwargs = self._identity_kwargs()
            kwargs[field] = bad_value
            with self.subTest(field=field), self.assertRaises(FMIdentityError):
                FMCheckpointIdentity(**kwargs)

    def test_identity_rejects_execution_slice_not_derived_from_runtime(self):
        kwargs = self._identity_kwargs()
        kwargs["execution_slice"] = (0, 8)
        with self.assertRaisesRegex(FMIdentityError, "execution_slice"):
            FMCheckpointIdentity(**kwargs)


class NativeCheckpointAdapterTests(unittest.TestCase):
    def _checkpoint(self, policy=None):
        policy = policy or _FakePhase1aPolicy()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "fm.ckpt"
        target = _qualified_name(policy)
        torch.save(
            {
                "cfg": {"policy": {"_target_": target}},
                "state_dicts": {"model": policy.state_dict()},
            },
            path,
        )
        return path, policy, target

    def test_native_loader_uses_cfg_policy_and_exact_model_state(self):
        path, policy, target = self._checkpoint()

        runtime = load_native_fm(
            path,
            expected_sha256=_sha256(path),
            device="cpu",
            expected_policy_target=target,
            instantiate_policy=lambda policy_cfg: policy,
        )

        self.assertIs(runtime.policy, policy)
        self.assertEqual(runtime.identity.policy_target, target)
        self.assertEqual(runtime.identity.execution_slice, (2, 10))
        self.assertFalse(any(parameter.requires_grad for parameter in policy.parameters()))
        self.assertFalse(policy.training)

    def test_native_loader_rejects_checkpoint_hash_mismatch(self):
        path, policy, target = self._checkpoint()
        with self.assertRaisesRegex(NativeFMLoadError, "checkpoint SHA256 mismatch"):
            load_native_fm(
                path,
                expected_sha256="0" * 64,
                device="cpu",
                expected_policy_target=target,
                instantiate_policy=lambda policy_cfg: policy,
            )

    def test_native_loader_rejects_runtime_class_substitution(self):
        path, _, target = self._checkpoint()

        class Substitute(_FakePhase1aPolicy):
            pass

        with self.assertRaisesRegex(NativeFMLoadError, "policy class mismatch"):
            load_native_fm(
                path,
                expected_sha256=_sha256(path),
                device="cpu",
                expected_policy_target=target,
                instantiate_policy=lambda policy_cfg: Substitute(),
            )

    def test_native_loader_rejects_runtime_shape_drift(self):
        path, policy, target = self._checkpoint()
        policy.horizon = 15
        with self.assertRaisesRegex(NativeFMLoadError, "horizon"):
            load_native_fm(
                path,
                expected_sha256=_sha256(path),
                device="cpu",
                expected_policy_target=target,
                instantiate_policy=lambda policy_cfg: policy,
            )


if __name__ == "__main__":
    unittest.main()
