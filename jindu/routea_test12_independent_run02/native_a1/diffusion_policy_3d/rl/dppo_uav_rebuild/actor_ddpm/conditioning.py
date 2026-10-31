"""Native E5 observation encoding for the official DPPO condition dict."""

from __future__ import annotations

from typing import Dict, Mapping

import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ObservationSpec


class ConditioningError(RuntimeError):
    """Raised when E5 conditioning would use a different observation contract."""


def encode_e5_condition(
    policy,
    policy_observation: Mapping[str, torch.Tensor],
    spec: ObservationSpec,
) -> Dict[str, torch.Tensor]:
    """Encode `(B,3,...)` with the frozen native normalizer and obs encoder."""
    condition, _ = encode_e5_condition_with_normalized(
        policy, policy_observation, spec
    )
    return condition


def encode_e5_condition_with_normalized(
    policy,
    policy_observation: Mapping[str, torch.Tensor],
    spec: ObservationSpec,
):
    """Return the exact normalized encoder input together with its condition."""
    if set(policy_observation) != {"agent_pos", "point_cloud"}:
        raise ConditioningError("policy observation keys must be agent_pos and point_cloud")
    agent_pos = policy_observation["agent_pos"]
    point_cloud = policy_observation["point_cloud"]
    if tuple(agent_pos.shape[1:]) != (spec.policy_history, spec.state_dim):
        raise ConditioningError(
            "agent_pos must have shape (B,{},{})".format(
                spec.policy_history, spec.state_dim
            )
        )
    if point_cloud.ndim != 4:
        raise ConditioningError("point_cloud must have shape (B,T,N,C)")
    if tuple(point_cloud.shape[1:3]) != (
        spec.policy_history,
        spec.point_count,
    ):
        raise ConditioningError(
            "point_cloud must use policy_history={} and point_count={}".format(
                spec.policy_history, spec.point_count
            )
        )
    if point_cloud.shape[-1] not in (
        spec.policy_point_dim,
        spec.checkpoint_point_dim,
    ):
        raise ConditioningError("point_cloud channels must be 3 or 6")
    if agent_pos.shape[0] != point_cloud.shape[0]:
        raise ConditioningError("agent_pos and point_cloud batch mismatch")
    if getattr(policy, "n_obs_steps", None) != spec.policy_history:
        raise ConditioningError("policy n_obs_steps does not match ObservationSpec")
    if getattr(policy, "obs_feature_dim", None) != spec.feature_dim_per_frame:
        raise ConditioningError("policy obs_feature_dim does not match ObservationSpec")
    if getattr(policy, "use_pc_color", None) is not False:
        raise ConditioningError("E5 conditioning requires use_pc_color=False")

    normalized = policy.normalizer.normalize(
        {"agent_pos": agent_pos, "point_cloud": point_cloud}
    )
    normalized = dict(normalized)
    normalized["point_cloud"] = normalized["point_cloud"][
        ..., : spec.policy_point_dim
    ]
    batch = agent_pos.shape[0]
    flat_observation = {
        key: value.reshape(-1, *value.shape[2:])
        for key, value in normalized.items()
    }
    with torch.no_grad():
        frame_features = policy.obs_encoder(flat_observation)
    expected_frame_shape = (
        batch * spec.policy_history,
        spec.feature_dim_per_frame,
    )
    if tuple(frame_features.shape) != expected_frame_shape:
        raise ConditioningError(
            "obs_encoder output mismatch: expected {}, got {}".format(
                expected_frame_shape, tuple(frame_features.shape)
            )
        )
    global_cond = frame_features.reshape(batch, spec.policy_condition_dim).detach()
    if not torch.isfinite(global_cond).all():
        raise ConditioningError("global_cond contains NaN or Inf")

    # Official VPG forward requires `state` to infer B. The E5 actor backend
    # consumes `global_cond`; both keys intentionally reference one tensor.
    return {"state": global_cond, "global_cond": global_cond}, normalized
