from types import SimpleNamespace
import unittest

import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.observation_provenance import (
    EncodedObservation,
    ObservationProvenance,
)
from scripts.evaluate_dppo_uav_formal_endpoint import _E5Policy


class _ObservationAdapter:
    @staticmethod
    def encode(_observation, _metadata, *, episode_identity_hash, decision_index):
        digest = "b" * 64
        return EncodedObservation(
            condition={"state": torch.zeros(1, 3, 192)},
            provenance=ObservationProvenance(
                raw_frame_hash=digest,
                raw_history_hash=digest,
                normalizer_identity_hash=digest,
                normalized_history_hash=digest,
                policy_input_hash=digest,
                episode_identity_hash=episode_identity_hash,
                decision_index=decision_index,
            ),
            normalized_history={"state": torch.zeros(1, 3, 192)},
        )

    @staticmethod
    def verify(_encoded):
        return None


class _ActionNormalizer:
    @staticmethod
    def unnormalize(value):
        return value


class _Normalizer:
    def __getitem__(self, key):
        if key != "action":
            raise KeyError(key)
        return _ActionNormalizer()


class _VendorRuntimeWithFrozenBug:
    def __init__(self):
        self.return_chain = None
        self.use_base_policy = None

    def __call__(
        self, *, cond, deterministic, return_chain, use_base_policy
    ):
        self.return_chain = return_chain
        self.use_base_policy = use_base_policy
        if not return_chain:
            raise AttributeError("'NoneType' object has no attribute 'append'")
        return SimpleNamespace(trajectories=torch.zeros(1, 16, 11), chains=object())


class ReturnChainCompatibilityTest(unittest.TestCase):
    def test_endpoint_sampler_requests_chain_to_avoid_frozen_vendor_bug(self):
        runtime = _VendorRuntimeWithFrozenBug()
        system = SimpleNamespace(
            observation_adapter=_ObservationAdapter(),
            official_runtime=runtime,
            bc_runtime=SimpleNamespace(
                policy=SimpleNamespace(normalizer=_Normalizer())
            ),
        )
        policy = _E5Policy(system, use_base_policy=True, identity_hash="a" * 64)

        action = policy.predict(
            {},
            {"pointcloud_meta_history": (), "episode_identity_hash": "a" * 64, "decision_index": 0},
            action_seed=7,
        )

        self.assertTrue(runtime.return_chain)
        self.assertTrue(runtime.use_base_policy)
        self.assertEqual(tuple(action.full.shape), (1, 16, 11))


if __name__ == "__main__":
    unittest.main()
