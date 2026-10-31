"""Observation bridging and deterministic native-E5 inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import (
    ActionBundle,
    ActionSpec,
    ObservationSpec,
)


EXPECTED_E5_TIMESTEPS = (45, 40, 35, 30, 25, 20, 15, 10, 5, 0)


class NativeInferenceError(RuntimeError):
    """Raised when observation or native E5 output violates its identity."""


@dataclass(frozen=True)
class NativeInferenceResult:
    actions: ActionBundle
    timesteps: Tuple[int, ...]
    seed: int


class ObservationBridge:
    """Convert an 8-frame Route A observation to E5 chronological latest-3."""

    def __init__(self, spec: ObservationSpec):
        self.spec = spec

    def to_policy_observation(
        self,
        environment_observation: Mapping[str, Any],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(environment_observation, Mapping):
            raise NativeInferenceError("environment observation must be a mapping")
        if set(environment_observation) != {"agent_pos", "point_cloud"}:
            raise NativeInferenceError(
                "environment observation keys must be agent_pos and point_cloud"
            )

        agent_pos = torch.as_tensor(
            environment_observation["agent_pos"], dtype=torch.float32
        )
        point_cloud = torch.as_tensor(
            environment_observation["point_cloud"], dtype=torch.float32
        )
        if agent_pos.ndim == 2 and point_cloud.ndim == 3:
            agent_pos = agent_pos.unsqueeze(0)
            point_cloud = point_cloud.unsqueeze(0)
        elif agent_pos.ndim != 3 or point_cloud.ndim != 4:
            raise NativeInferenceError(
                "observation must be unbatched (T,...) or batched (B,T,...)"
            )

        if agent_pos.shape[0] != point_cloud.shape[0]:
            raise NativeInferenceError("agent_pos and point_cloud batch mismatch")
        if agent_pos.shape[1] != point_cloud.shape[1]:
            raise NativeInferenceError("agent_pos and point_cloud history mismatch")
        if agent_pos.shape[1] != self.spec.environment_history:
            raise NativeInferenceError(
                "environment history must be exactly {}; got {}".format(
                    self.spec.environment_history, agent_pos.shape[1]
                )
            )
        if tuple(agent_pos.shape[-1:]) != (self.spec.state_dim,):
            raise NativeInferenceError("agent_pos must end with state_dim=9")
        if point_cloud.shape[-2] != self.spec.point_count:
            raise NativeInferenceError("point_cloud must contain 4096 points")
        if point_cloud.shape[-1] not in (
            self.spec.policy_point_dim,
            self.spec.checkpoint_point_dim,
        ):
            raise NativeInferenceError("point_cloud channels must be 3 or 6")
        if not torch.isfinite(agent_pos).all() or not torch.isfinite(point_cloud).all():
            raise NativeInferenceError("observation contains NaN or Inf")

        start = self.spec.environment_history - self.spec.policy_history
        policy_agent = agent_pos[:, start:, :]
        policy_points = point_cloud[:, start:, :, : self.spec.policy_point_dim]
        return {
            "agent_pos": policy_agent.to(device=device).contiguous(),
            "point_cloud": policy_points.to(device=device).contiguous(),
        }


def _scheduler_timesteps(policy: Any) -> Tuple[int, ...]:
    raw = getattr(getattr(policy, "noise_scheduler", None), "timesteps", None)
    if raw is None:
        raise NativeInferenceError("native policy did not expose scheduler timesteps")
    values = tuple(int(item.item()) if hasattr(item, "item") else int(item) for item in raw)
    if values != EXPECTED_E5_TIMESTEPS:
        raise NativeInferenceError(
            "E5 timestep identity mismatch: expected {}, got {}".format(
                EXPECTED_E5_TIMESTEPS, values
            )
        )
    return values


def run_native_inference(
    runtime: Any,
    environment_observation: Mapping[str, Any],
    observation_spec: ObservationSpec,
    action_spec: ActionSpec,
    seed: int,
) -> NativeInferenceResult:
    """Run native predict_action with an owned, non-leaking RNG seed."""
    if not isinstance(seed, int) or seed < 0:
        raise NativeInferenceError("seed must be a non-negative integer")
    device = torch.device(runtime.device)
    observation = ObservationBridge(observation_spec).to_policy_observation(
        environment_observation, device=device
    )
    fork_devices = []
    if device.type == "cuda":
        fork_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]

    with torch.random.fork_rng(devices=fork_devices, enabled=True):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        with torch.inference_mode():
            output = runtime.policy.predict_action(observation)

    if not isinstance(output, Mapping):
        raise NativeInferenceError("native predict_action must return a mapping")
    if "action_pred" not in output or "action" not in output:
        raise NativeInferenceError(
            "native predict_action must return action_pred and action"
        )
    full = output["action_pred"]
    expected_policy, deployment, rollout = action_spec.split(full)
    actions = ActionBundle(
        spec=action_spec,
        full=full,
        policy=output["action"],
        deployment_executed=deployment,
        rollout_executed=rollout,
    )
    if tuple(expected_policy.shape) != tuple(actions.policy.shape):
        raise NativeInferenceError("native policy action shape mismatch")

    return NativeInferenceResult(
        actions=actions,
        timesteps=_scheduler_timesteps(runtime.policy),
        seed=seed,
    )
