"""Shared checkpoint loader for evaluation and Gazebo/PX4 deployment."""

from dataclasses import dataclass
from pathlib import Path

import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.restore import (
    restore_checkpoint,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.contracts import ActionBundle
from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.build_system import build_system
from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import (
    build_runtime_identity,
)


@dataclass
class DeploymentPolicy:
    system: object
    checkpoint_header: object

    @torch.no_grad()
    def predict(self, environment_observation, observation_context, *, action_seed=None):
        if action_seed is not None:
            torch.manual_seed(int(action_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(action_seed))
        encoded = self.system.observation_adapter.encode(
            environment_observation,
            observation_context["pointcloud_meta_history"],
            episode_identity_hash=observation_context["episode_identity_hash"],
            decision_index=observation_context["decision_index"],
        )
        self.system.observation_adapter.verify(encoded)
        condition = encoded.condition
        samples = self.system.official_runtime(
            cond=condition,
            deterministic=True,
            return_chain=False,
            use_base_policy=False,
        )
        full = self.system.bc_runtime.policy.normalizer["action"].unnormalize(
            samples.trajectories
        )
        return ActionBundle.from_full(full)


def load_deployment_policy(config, checkpoint_path, repo_root: Path):
    system = build_system(config, repo_root)
    manifest = Path(config.admission_manifest)
    if not manifest.is_absolute():
        manifest = repo_root / manifest
    identity = build_runtime_identity(config, repo_root, system, manifest.resolve())
    header = restore_checkpoint(
        checkpoint_path,
        runtime_identity=identity,
        lora_identity=system.lora_bundle.identity,
        lora_bundle=system.lora_bundle,
        critic=system.critic,
        actor_optimizer=system.actor_optimizer,
        critic_optimizer=system.critic_optimizer,
        counters=system.trainer.counters,
        map_location=system.bc_runtime.device,
    )
    return DeploymentPolicy(system=system, checkpoint_header=header)
