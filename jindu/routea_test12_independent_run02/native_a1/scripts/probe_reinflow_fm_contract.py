#!/usr/bin/env python3
"""R0-R3 probe for the pinned official ReinFlow × native Phase1aPolicyFM core."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy_3d.rl.reinflow_fm.audit import assert_first_ratio
from diffusion_policy_3d.rl.reinflow_fm.build_system import (
    DEFAULT_WSL_REPO_ROOT,
    EXPECTED_REWARD_V13_SHA256,
    PreflightInputs,
    SystemBuildConfig,
    build_core_system,
    inventory_lora_targets,
    validate_preflight,
)
from diffusion_policy_3d.rl.reinflow_fm.checkpoint_adapter import (
    EXPECTED_POLICY_TARGET,
    load_native_fm,
    sha256_file,
)
from diffusion_policy_3d.rl.reinflow_fm.condition_adapter import encode_conditions
from diffusion_policy_3d.rl.reinflow_fm.official_core.vendor_identity import (
    verify_vendor_snapshot,
)
from diffusion_policy_3d.rl.reinflow_fm.policy_bridge import (
    condition_to_official_dict,
)
from diffusion_policy_3d.rl.reinflow_fm.runtime import PPOFlowBuildConfig


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_observation(path: Path, expected_sha256: str):
    if not path.is_file():
        raise FileNotFoundError(f"observation fixture is missing: {path}")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"observation SHA256 mismatch: expected {expected_sha256}, got {actual}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and "observation" in payload:
        payload = payload["observation"]
    if not isinstance(payload, Mapping):
        raise ValueError("observation fixture must be a tensor mapping")
    return dict(payload), actual


def _ppo_config(config: Mapping[str, Any], device: str) -> PPOFlowBuildConfig:
    ppo = config["ppo"]
    return PPOFlowBuildConfig(
        device=device,
        act_min=float(ppo["act_min"]),
        act_max=float(ppo["act_max"]),
        obs_dim=int(ppo["obs_dim"]),
        cond_steps=int(ppo["cond_steps"]),
        noise_scheduler_type=str(ppo["noise_scheduler_type"]),
        inference_steps=int(ppo["inference_steps"]),
        ft_denoising_steps=int(ppo["ft_denoising_steps"]),
        randn_clip_value=float(ppo["randn_clip_value"]),
        min_sampling_denoising_std=float(ppo["min_sampling_denoising_std"]),
        min_logprob_denoising_std=float(ppo["min_logprob_denoising_std"]),
        max_logprob_denoising_std=float(ppo["max_logprob_denoising_std"]),
        logprob_min=float(ppo["logprob_min"]),
        logprob_max=float(ppo["logprob_max"]),
        clip_ploss_coef=float(ppo["clip_ploss_coef"]),
        clip_ploss_coef_base=float(ppo["clip_ploss_coef_base"]),
        clip_ploss_coef_rate=float(ppo["clip_ploss_coef_rate"]),
        clip_vloss_coef=float(ppo["clip_vloss_coef"]),
        denoised_clip_value=float(ppo["denoised_clip_value"]),
        time_dim_explore=int(ppo["time_dim_explore"]),
        learn_explore_time_embedding=bool(ppo["learn_explore_time_embedding"]),
        use_time_independent_noise=bool(ppo["use_time_independent_noise"]),
        noise_hidden_dims=tuple(int(value) for value in ppo["noise_hidden_dims"]),
        logprob_debug_sample=bool(ppo["logprob_debug_sample"]),
        logprob_debug_recalculate=bool(ppo["logprob_debug_recalculate"]),
        explore_net_activation_type=str(ppo["explore_net_activation_type"]),
    )


def _system_config(args, config, preflight) -> SystemBuildConfig:
    optimizer = config["optimizer"]
    critic = config["critic"]
    lora = config["lora"]
    return SystemBuildConfig(
        preflight=preflight,
        device=args.device,
        expected_policy_target=args.expected_policy_target,
        lora_rank=int(lora["rank"]),
        lora_alpha=float(lora["alpha"]),
        expected_linear_targets=int(args.expected_linear),
        expected_conv1d_targets=int(args.expected_conv1d),
        actor_lr=float(optimizer["actor_lr"]),
        critic_lr=float(optimizer["critic_lr"]),
        weight_decay=float(optimizer["weight_decay"]),
        critic_hidden_dim=int(critic["hidden_dim"]),
        ppo=_ppo_config(config, args.device),
    )


def _json_write(result: Mapping[str, Any], output: Path | None) -> None:
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")


@torch.no_grad()
def _deterministic_parity(core, official_condition, *, seed: int) -> float:
    policy = core.native_runtime.policy
    batch = official_condition["state"].shape[0]
    generator = torch.Generator(device=core.native_runtime.device).manual_seed(seed)
    initial = torch.randn(
        batch,
        core.native_runtime.identity.horizon_steps,
        core.native_runtime.identity.action_dim,
        generator=generator,
        device=core.native_runtime.device,
        dtype=torch.float32,
    )
    path_scale = 1.0 - float(policy.sigma_min)
    native_dt = path_scale / int(policy.num_fm_steps)
    native = initial.clone()
    for index in range(int(policy.num_fm_steps)):
        native_time = torch.full(
            (batch,), index * native_dt, device=native.device, dtype=native.dtype
        )
        velocity = policy._compute_cfg_velocity(
            native,
            (native_time * float(policy.fm_t_scale)).float(),
            official_condition["global_cond"],
            official_condition["global_cond_uncond"],
            None,
        )
        native = native + velocity * native_dt

    adapted = initial.clone()
    for index in range(int(policy.num_fm_steps)):
        official_time = torch.full(
            (batch,),
            index / int(policy.num_fm_steps),
            device=adapted.device,
            dtype=adapted.dtype,
        )
        adapted = adapted + core.bridge(
            adapted, official_time, official_condition
        ) / int(policy.num_fm_steps)
    return float((native - adapted).abs().max())


def _gradient_reachability(core, official_condition, chain, old_logprob):
    runtime = core.official_runtime
    batch = chain.shape[0]
    if batch < 2:
        raise ValueError("R3 gradient probe requires observation batch size >= 2")
    runtime.actor_ft.zero_grad(set_to_none=True)
    core.critic.zero_grad(set_to_none=True)
    values = core.critic(official_condition).reshape(-1)
    advantages = torch.linspace(
        -1.0, 1.0, batch, device=values.device, dtype=values.dtype
    )
    returns = values.detach() + advantages
    result = runtime.loss(
        official_condition,
        chain,
        returns,
        values.detach(),
        advantages,
        old_logprob,
        use_bc_loss=False,
        normalize_denoising_horizon=False,
        normalize_act_space_dimension=False,
        verbose=False,
        clip_intermediate_actions=False,
        account_for_initial_stochasticity=True,
    )
    policy_loss, entropy_loss, value_loss = result[:3]
    objective = policy_loss + 0.01 * entropy_loss + 0.5 * value_loss
    objective.backward()

    def grad_norm(named_parameters, predicate=lambda name: True):
        total = 0.0
        for name, parameter in named_parameters:
            if predicate(name) and parameter.grad is not None:
                total += float(parameter.grad.detach().float().pow(2).sum())
        return total ** 0.5

    adapter_norm = grad_norm(
        runtime.actor_ft.policy.named_parameters(),
        lambda name: name.endswith(("lora_a", "lora_b")),
    )
    noise_norm = grad_norm(runtime.actor_ft.explore_noise_net.named_parameters())
    critic_norm = grad_norm(core.critic.named_parameters())
    if min(adapter_norm, noise_norm, critic_norm) <= 0.0:
        raise RuntimeError(
            "R3 gradient reachability failed: "
            f"adapter={adapter_norm}, noise={noise_norm}, critic={critic_norm}"
        )
    return {
        "objective": float(objective.detach()),
        "adapter_grad_norm": adapter_norm,
        "noise_grad_norm": noise_norm,
        "critic_grad_norm": critic_norm,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "reinflow_fm" / "parity.yaml",
    )
    parser.add_argument(
        "--repo-root", type=Path, default=DEFAULT_WSL_REPO_ROOT
    )
    parser.add_argument("--vendor-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-policy-target", default=EXPECTED_POLICY_TARGET)
    parser.add_argument("--expected-linear", type=int)
    parser.add_argument("--expected-conv1d", type=int)
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--observation-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    config = _load_yaml(args.config.resolve())
    repo_root = args.repo_root.resolve()
    vendor_root = (
        args.vendor_root.resolve()
        if args.vendor_root is not None
        else repo_root / "third_party" / "reinflow_official_e722e15"
    )
    preflight = validate_preflight(
        PreflightInputs(
            checkpoint=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            vendor_root=vendor_root,
            reward_source_sha256=EXPECTED_REWARD_V13_SHA256,
        )
    )
    reward_path = repo_root / "diffusion_policy_3d" / "env" / "reward_v13.py"
    if not reward_path.is_file():
        raise FileNotFoundError(f"Reward v1.3 source is missing: {reward_path}")
    reward_hash = _sha256(reward_path)
    if reward_hash != EXPECTED_REWARD_V13_SHA256:
        raise ValueError(
            f"Reward v1.3 hash mismatch: expected {EXPECTED_REWARD_V13_SHA256}, "
            f"got {reward_hash}"
        )

    vendor_identity = verify_vendor_snapshot(preflight.vendor_root)
    native = load_native_fm(
        preflight.checkpoint,
        expected_sha256=preflight.checkpoint_sha256,
        device=args.device,
        expected_policy_target=args.expected_policy_target,
    )
    inventory = inventory_lora_targets(native.policy.model)
    result = {
        "status": "INVENTORY_ONLY",
        "R0": {
            "status": "PASS",
            "vendor_commit": vendor_identity.commit,
            "vendor_manifest_sha256": vendor_identity.manifest_sha256,
            "vendor_file_count": vendor_identity.file_count,
            "reward_v13_source_sha256": reward_hash,
        },
        "checkpoint": {
            "path": str(preflight.checkpoint),
            "sha256": sha256_file(preflight.checkpoint),
            "identity_hash": native.identity.content_hash,
            "policy_target": native.identity.policy_target,
            "policy_source_sha256": native.identity.policy_source_sha256,
            "normalizer_sha256": native.identity.normalizer_sha256,
            "n_obs_steps": native.identity.n_obs_steps,
            "n_action_steps": native.identity.n_action_steps,
            "execution_slice": list(native.identity.execution_slice),
        },
        "lora_inventory": {
            "linear_count": inventory.linear_count,
            "conv1d_count": inventory.conv1d_count,
            "content_hash": inventory.content_hash,
            "linear_names": list(inventory.linear_names),
            "conv1d_names": list(inventory.conv1d_names),
        },
    }
    counts = (args.expected_linear, args.expected_conv1d)
    if any(value is None for value in counts):
        result["next_action"] = (
            "rerun with --expected-linear and --expected-conv1d set to this "
            "checkpoint's measured inventory"
        )
        _json_write(result, args.output)
        return 0
    measured_counts = (inventory.linear_count, inventory.conv1d_count)
    if counts != measured_counts:
        raise ValueError(
            f"explicit LoRA inventory {counts} does not match measured {measured_counts}"
        )

    core = build_core_system(_system_config(args, config, preflight))
    result["status"] = "CORE_BUILT"
    result["official_runtime"] = {
        "ppo_class": (
            f"{type(core.official_runtime).__module__}."
            f"{type(core.official_runtime).__name__}"
        ),
        "inference_steps": core.official_runtime.inference_steps,
        "chain_shape": [None, 5, 16, 11],
        "trainable_actor_parameters": sum(
            parameter.numel()
            for parameter in core.official_runtime.actor_ft.parameters()
            if parameter.requires_grad
        ),
    }
    if args.observation is None or args.observation_sha256 is None:
        result["next_action"] = (
            "provide --observation and --observation-sha256 for R1-R3"
        )
        _json_write(result, args.output)
        return 0

    observation, observation_hash = _load_observation(
        args.observation, args.observation_sha256
    )
    condition = encode_conditions(
        core.native_runtime.policy,
        observation,
        normalizer_sha256=core.native_runtime.identity.normalizer_sha256,
    )
    official_condition = condition_to_official_dict(condition)
    parity_error = _deterministic_parity(
        core, official_condition, seed=args.seed
    )
    if parity_error > 1.0e-6:
        raise RuntimeError(
            f"R1 native/adapted deterministic parity failed: {parity_error}"
        )
    result["R1"] = {
        "status": "PASS",
        "observation_sha256": observation_hash,
        "raw_observation_sha256": list(condition.raw_obs_sha256),
        "encoder_sha256": condition.encoder_sha256,
        "condition_shape": list(condition.cond.shape),
        "noise_condition_shape": list(condition.noise_cond.shape),
        "max_abs_parity_error": parity_error,
        "threshold": 1.0e-6,
    }

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    runtime = core.official_runtime
    _, chain, old_logprob = runtime.get_actions(
        official_condition,
        eval_mode=False,
        save_chains=True,
        normalize_denoising_horizon=False,
        normalize_act_space_dimension=False,
        clip_intermediate_actions=False,
        account_for_initial_stochasticity=True,
    )
    new_logprob = runtime.get_logprobs(
        official_condition,
        chain,
        get_entropy=False,
        normalize_denoising_horizon=False,
        normalize_act_space_dimension=False,
        clip_intermediate_actions=False,
        verbose_entropy_stats=False,
        account_for_initial_stochasticity=True,
        get_chains_stds=False,
    )
    ratio_error = assert_first_ratio(old_logprob, new_logprob, tolerance=1.0e-6)
    result["R2"] = {
        "status": "PASS",
        "chain_shape": list(chain.shape),
        "old_logprob_shape": list(old_logprob.shape),
        "max_first_ratio_error": ratio_error,
        "threshold": 1.0e-6,
        "sampling_owner": "official PPOFlow.get_actions",
        "recompute_owner": "official PPOFlow.get_logprobs",
    }
    result["R3"] = {
        "status": "PASS",
        **_gradient_reachability(
            core, official_condition, chain, old_logprob.detach()
        ),
    }
    result["status"] = "R0_R3_PASS"
    _json_write(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
