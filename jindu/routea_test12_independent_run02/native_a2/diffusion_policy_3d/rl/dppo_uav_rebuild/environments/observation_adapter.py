"""Route A 8-frame observation to native E5 latest-3 condition."""

from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.conditioning import (
    encode_e5_condition_with_normalized,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.bc_runtime.inference import (
    ObservationBridge,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ObservationSpec
from .observation_provenance import (
    EncodedObservation,
    ObservationIdentityMismatch,
    ObservationProvenance,
    canonical_array_hash,
    canonical_mapping_hash,
    normalizer_identity_hash,
    validate_metadata_history,
)


class E5ObservationAdapter:
    def __init__(
        self,
        policy,
        device,
        spec=None,
        max_no_ray_hit_fraction=0.99,
        reject_all_no_ray_hit=True,
    ):
        self.policy = policy
        self.device = device
        self.spec = spec or ObservationSpec.e5_route_a()
        self.bridge = ObservationBridge(self.spec)
        self.max_no_ray_hit_fraction = float(max_no_ray_hit_fraction)
        if not isinstance(reject_all_no_ray_hit, bool):
            raise TypeError("reject_all_no_ray_hit must be boolean")
        self.reject_all_no_ray_hit = reject_all_no_ray_hit

    def encode(
        self,
        environment_observation,
        pointcloud_meta_history,
        *,
        episode_identity_hash,
        decision_index,
    ):
        validate_metadata_history(
            environment_observation["point_cloud"],
            tuple(pointcloud_meta_history),
            episode_identity_hash=episode_identity_hash,
            decision_index=decision_index,
            max_no_ray_hit_fraction=self.max_no_ray_hit_fraction,
            reject_all_no_ray_hit=self.reject_all_no_ray_hit,
        )
        policy_observation = self.bridge.to_policy_observation(
            environment_observation, self.device
        )
        condition, normalized = encode_e5_condition_with_normalized(
            self.policy, policy_observation, self.spec
        )
        provenance = ObservationProvenance(
            raw_frame_hash=pointcloud_meta_history[-1].raw_array_hash,
            raw_history_hash=canonical_array_hash(
                environment_observation["point_cloud"]
            ),
            normalizer_identity_hash=normalizer_identity_hash(
                self.policy.normalizer
            ),
            normalized_history_hash=canonical_mapping_hash(normalized),
            policy_input_hash=canonical_mapping_hash(condition),
            episode_identity_hash=episode_identity_hash,
            decision_index=int(decision_index),
        )
        return EncodedObservation(condition, provenance, normalized)

    def verify(self, encoded: EncodedObservation) -> None:
        if normalizer_identity_hash(self.policy.normalizer) != (
            encoded.provenance.normalizer_identity_hash
        ):
            raise ObservationIdentityMismatch("normalizer_identity")
        if canonical_mapping_hash(encoded.normalized_history) != (
            encoded.provenance.normalized_history_hash
        ):
            raise ObservationIdentityMismatch("normalized_history_hash")
        if canonical_mapping_hash(encoded.condition) != (
            encoded.provenance.policy_input_hash
        ):
            raise ObservationIdentityMismatch("policy_input_hash")
