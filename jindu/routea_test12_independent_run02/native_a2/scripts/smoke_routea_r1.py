#!/usr/bin/env python3
"""One admitted train-domain R1 decision with the real E5 actor; no optimizer."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
from omegaconf import OmegaConf
import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.critic_state import with_history_scalar, with_episode_context
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.initialization_parity import (
    _restore_rng, _rng_hash, _rng_snapshot, compare_a0_update0_state,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.plan_probe import sample_probe_action
from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.build_system import build_system
from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.config_schema_v2 import validate_v2_config
from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import build_runtime_identity
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.task_sampler import TaskSampler


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--a0-update0", type=Path,
                        help="Optional existing A0 checkpoint_update_000000 directory")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(args.config)
    validate_v2_config(config)
    protocol = config.get("task_protocol")
    if str(config.reward_version) != "v2":
        raise ValueError("R1 smoke requires Reward v2")
    seed = int(config.run.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    system = build_system(config, repo_root)
    from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.factory import RouteAEnvironmentFactory
    manifest = Path(config.admission_manifest)
    if not manifest.is_absolute():
        manifest = repo_root / manifest
    worlds = Path(config.worlds_train)
    if not worlds.is_absolute():
        worlds = repo_root / worlds
    factory = RouteAEnvironmentFactory(
        manifest_path=manifest, worlds_train_dir=worlds,
        reward=system.reward, device=str(system.bc_runtime.device),
        base_seed=seed,
        max_episode_steps=int(config.environment.max_episode_steps),
        disturb_enable=bool(config.environment.disturb_enable),
        disturb_prob=float(config.environment.disturb_prob),
        disturb_pos_scale=float(config.environment.disturb_pos_scale),
        exec_noise_scale=float(config.environment.exec_noise_scale),
        mid_state_init_ratio=float(config.environment.mid_state_init_ratio),
        mid_state_pos_scale=float(config.environment.mid_state_pos_scale),
        max_reset_attempts=int(config.environment.max_reset_attempts),
        reward_config=OmegaConf.to_container(config.reward, resolve=True),
        terminate_at_space_limits=protocol is None,
        space_cost_budget=None if protocol is None else float(protocol.space_cost_budget),
    )
    a0_update0_parity = None
    if args.a0_update0 is not None:
        a0_path = args.a0_update0 / "state.pt"
        if not a0_path.is_file():
            raise FileNotFoundError("A0 update-0 state.pt missing: {}".format(a0_path))
        rng_before_load = _rng_snapshot(torch.device(system.bc_runtime.device))
        a0_state = torch.load(a0_path, map_location="cpu", weights_only=False)
        a0_update0_parity = compare_a0_update0_state(
            a0_state, system.lora_bundle, system.critic, rng_before_load)
        _restore_rng(rng_before_load, torch.device(system.bc_runtime.device))
    task = TaskSampler(factory.tasks, seed=seed + 17).sample()
    adapter = factory.make(task)
    obs, context = adapter.reset(seed=task.seed)
    before_history = dict(adapter.environment._reward_history)
    encoded = system.observation_adapter.encode(
        obs, context["pointcloud_meta_history"],
        episode_identity_hash=context["episode_identity_hash"],
        decision_index=context["decision_index"])
    system.observation_adapter.verify(encoded)
    probe_rng_before = _rng_snapshot(torch.device(system.bc_runtime.device))
    probe_first = sample_probe_action(
        system.rollout_sampler, encoded.condition, seed=seed + 981731,
        device=system.bc_runtime.device)
    probe_second = sample_probe_action(
        system.rollout_sampler, encoded.condition, seed=seed + 981731,
        device=system.bc_runtime.device)
    if not torch.equal(probe_first, probe_second) or _rng_hash(probe_rng_before) != _rng_hash(
            _rng_snapshot(torch.device(system.bc_runtime.device))):
        raise RuntimeError("R1 production actor probe is not repeatable or changed RNG")
    with torch.no_grad():
        critic_condition = (with_episode_context(encoded.condition, adapter.environment) if protocol is not None
                            else with_history_scalar(encoded.condition, before_history))
        value_before = system.critic(critic_condition)
        rollout = system.rollout_sampler.sample(encoded.condition, deterministic=False)
    next_obs, reward, terminated, truncated, info = adapter.step(rollout.action)
    next_encoded = system.observation_adapter.encode(
        next_obs, info["pointcloud_meta_history"],
        episode_identity_hash=info["episode_identity_hash"],
        decision_index=info["decision_index"])
    system.observation_adapter.verify(next_encoded)
    with torch.no_grad():
        next_critic_condition = (with_episode_context(next_encoded.condition, adapter.environment) if protocol is not None
                                 else with_history_scalar(next_encoded.condition, adapter.environment._reward_history))
        value_after = system.critic(next_critic_condition)
    if rollout.action.policy.shape != (1, 8, 11):
        raise RuntimeError("R1 policy action is not 8x11")
    if rollout.action.rollout_executed.shape != (1, 4, 11):
        raise RuntimeError("R1 physical action is not H4")
    if abs(float(reward) - sum(info["reward_terms"].values())) > 1e-8:
        raise RuntimeError("R1 reward sum mismatch")
    if not (np.isfinite(reward) and torch.isfinite(value_before).all()
            and torch.isfinite(value_after).all()):
        raise RuntimeError("R1 smoke produced nonfinite reward/value")
    identity = build_runtime_identity(config, repo_root, system, manifest)
    print(json.dumps({
        "event": "R1_SMOKE_COMPLETE", "seed": seed,
        "task_id": task.task_id, "task_role": task.role,
        "runtime_identity_hash": identity.content_hash,
        "initialization_parity": system.initialization_parity,
        "a0_update0_parity": a0_update0_parity,
        "probe_repeat_equal_and_rng_isolated": True,
        "action_policy_shape": list(rollout.action.policy.shape),
        "action_executed_shape": list(rollout.action.rollout_executed.shape),
        "beta_before": before_history["best_distance"] / before_history["initial_distance"],
        "beta_after": adapter.environment._reward_history["best_distance"] / before_history["initial_distance"],
        "value_before": float(value_before.reshape(-1)[0]),
        "value_after": float(value_after.reshape(-1)[0]),
        "task_protocol": OmegaConf.to_container(protocol, resolve=True) if protocol is not None else None,
        "critic_episode_context_before": (critic_condition["episode_context"][0].tolist()
                                           if protocol is not None else None),
        "critic_episode_context_after": (next_critic_condition["episode_context"][0].tolist()
                                          if protocol is not None else None),
        "reward": float(reward), "reward_terms": info["reward_terms"],
        "reward_metrics": info["reward_metrics"],
        "done_reason": info["done_reason"],
        "terminated": bool(terminated), "truncated": bool(truncated),
        "executed_path": np.asarray(info["executed_path"]).tolist(),
        "event_segment": info["event_segment"],
        "event_fraction": info["event_fraction"],
        "pointcloud_meta_history_count": len(info["pointcloud_meta_history"]),
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
