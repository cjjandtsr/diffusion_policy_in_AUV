"""Evaluation-only actor loading and common environment construction."""
from __future__ import annotations

from dataclasses import asdict
import copy
from pathlib import Path

import numpy as np

from .core import canonical_hash, file_hash, read_json, verify_checkpoint


def resolve(root, value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (Path(root) / p).resolve()


def catalogue(root):
    p = Path(root) / "scripts/routea_paired_support/policy_catalog_v1.json"
    result = read_json(p)
    body = {k: v for k, v in result.items() if k != "content_hash"}
    if canonical_hash(body) != result["content_hash"]:
        raise ValueError("catalogue content hash mismatch")
    return result


def model_name(descriptor):
    if descriptor.get('use_base_policy'):
        return 'bc'
    seed = descriptor.get('training_seed')
    stem = (descriptor['family'] + '_seed' + str(seed) if seed is not None else
            'rl_' + descriptor['runtime_identity_hash'][:12])
    return stem + '_u' + str(descriptor['update'])


def resolve_catalogue(root, checkpoint_paths):
    """Read actual checkpoint metadata; callers only supply weight addresses."""
    accepted = catalogue(root)
    cat = copy.deepcopy(accepted)
    cat.pop('content_hash')
    cat['schema'] = 'routea_paired_catalog_v3'
    cat['terminate_at_space_limits'] = False
    cat['evaluation_config'] = accepted['policies']['bc']
    cat['policies'] = {'bc': accepted['policies']['bc']}
    known = {d['runtime_identity_hash']: d for d in accepted['policies'].values()
             if not d['use_base_policy']}
    for value in checkpoint_paths:
        p = resolve(root, value)
        if p.suffix == '.ckpt':
            if file_hash(p) != cat['common']['bc_file_sha256']:
                raise ValueError('this evaluator uses the shared E5 BC checkpoint')
            cat['common']['bc_checkpoint_path'] = str(p)
            continue
        h, sums = read_json(p / 'header.json'), read_json(p / 'SHA256.json')
        d = copy.deepcopy(known.get(h['runtime_identity_hash'], {
            'family': 'rl', 'training_seed': None, 'use_base_policy': False,
            'config_path': cat['evaluation_config']['config_path'],
            'config_sha256': cat['evaluation_config']['config_sha256']}))
        d.update(checkpoint_path=str(p), update=h['update_index'],
                 header_sha256=sums['header.json'], state_sha256=sums['state.pt'],
                 runtime_identity_hash=h['runtime_identity_hash'], runtime_identity=h['runtime_identity'])
        verify_checkpoint(p, d)
        if h['lora_identity_hash'] != cat['common']['lora_identity_hash']:
            raise ValueError('checkpoint has different actor targets')
        for key, expected in cat['common']['shared_runtime_fields'].items():
            if h['runtime_identity'][key] != expected:
                raise ValueError('incompatible checkpoint: ' + key)
        name = model_name(d)
        if name in cat['policies'] and cat['policies'][name]['state_sha256'] != d['state_sha256']:
            raise ValueError('two different weights have the same training identity and update')
        cat['policies'][name] = d
    cat['content_hash'] = canonical_hash(cat)
    return cat


def evaluation_sources(root):
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import _iter_source_paths
    paths = [Path(root) / "scripts/run_routea_paired_comparison.py",
             Path(root) / "scripts/evaluate_dppo_uav_formal_endpoint.py"]
    paths += [p for p in sorted((Path(root) / "scripts/routea_paired_support").glob("*.py"))
              if p.name not in ("analysis.py","statistics.py")]
    paths += [Path(root) / "scripts/routea_paired_support/policy_catalog_v1.json"]
    paths += [Path(root) / "scripts/routea_paired_support/paper_bank_spec_v1.json"]
    paths += [Path(root) / "scripts/analyze_routea_trajectory_behavior.py",
              Path(root) / "scripts/routea_paired_support/recovery_bank_v1.json"]
    # Recheck the existing training source closure at every public collection
    # stage; helper-only hashes would miss changes to the common physics/sampler.
    paths += list(_iter_source_paths(Path(root)))
    return {str(p.relative_to(root)).replace("\\", "/"): file_hash(p) for p in sorted(set(paths))}


def config(root, descriptor):
    from omegaconf import OmegaConf
    p = resolve(root, descriptor["config_path"])
    if file_hash(p) != descriptor["config_sha256"]:
        raise ValueError("config differs from accepted receipt: " + str(p))
    return OmegaConf.load(p)


def common_system(root, cat):
    import torch
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.build_system import build_system
    if file_hash(resolve(root, cat["common"]["bc_checkpoint_path"])) != cat["common"]["bc_file_sha256"]:
        raise ValueError("BC bytes changed after preflight")
    c = config(root, cat.get('evaluation_config', cat['policies']['bc']))
    c.checkpoint = str(resolve(root, cat['common']['bc_checkpoint_path']))
    torch.manual_seed(int(c.run.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(c.run.seed))
    system = build_system(c, Path(root))
    system.official_runtime.eval()
    system.bc_runtime.policy.eval()
    for parameter in system.bc_runtime.policy.parameters():
        parameter.requires_grad_(False)
    if system.bc_runtime.identity.content_hash != cat["common"]["bc_identity_hash"]:
        raise ValueError("actual BC identity differs")
    return system, c


def validate_actor(bundle, tensors):
    import torch
    expected = dict(bundle.named_adapter_parameters())
    if set(tensors) != set(expected):
        raise ValueError("actor target names differ")
    for name, tensor in tensors.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                or tensor.shape != expected[name].shape or not torch.isfinite(tensor).all()):
            raise ValueError("invalid actor tensor: " + name)


def load_actor(root, system, descriptor):
    import torch
    p = resolve(root, descriptor["checkpoint_path"])
    header = verify_checkpoint(p, descriptor)
    if header["lora_identity_hash"] != system.lora_bundle.identity.content_hash:
        raise ValueError("actual LoRA target identity differs")
    state = torch.load(p / "state.pt", map_location="cpu", weights_only=True)
    transitions, commits = 1024, 40  # Accepted legacy S2 checkpoints.
    if "training_config" in state:
        from omegaconf import OmegaConf
        from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.config_schema_v2 import validate_v2_config
        from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import _canonical_hash
        saved = state["training_config"]
        if _canonical_hash(saved) != header["runtime_identity"]["config_hash"]:
            raise ValueError("saved training config differs from checkpoint identity")
        contract = validate_v2_config(OmegaConf.create(saved))
        transitions = int(saved["run"]["rollout_transitions"])
        commits = contract.expected_optimizer_commits_per_outer_update
    if (state["counters"]["env_steps"] != descriptor["update"] * transitions
            or state["counters"]["committed_updates"] != descriptor["update"] * commits):
        raise ValueError("actual saved weight state is not the complete training budget")
    validate_actor(system.lora_bundle, state["lora"])
    system.lora_bundle.load_adapter_state_dict(state["lora"])
    # Critic, optimizers, counters and training RNG are deliberately not restored.
    return header


def preflight(root, cat):
    """Verify compatible actors once using the common evaluation environment."""
    from omegaconf import OmegaConf
    import torch
    system, c = common_system(root, cat)
    checkpoints = []
    for name, descriptor in cat["policies"].items():
        if name != "bc":
            header = load_actor(root, system, descriptor)
            checkpoints.append({"policy": name, "update": header["update_index"],
                                "state_sha256": descriptor["state_sha256"],
                                "original_training_identity": header["runtime_identity_hash"]})
    return {"status": "pass", "catalogue_hash": cat["content_hash"],
            "terminate_at_space_limits": False,
            "evaluation_sources": evaluation_sources(Path(root)),
            "checkpoints": checkpoints, "common_config": OmegaConf.to_container(c, resolve=True),
            "versions": {"torch": torch.__version__, "numpy": np.__version__,
                         "cuda": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            "load_scope": "actor-only evaluation; no training restore"}


def make_adapter(system, c, world, role):
    # Every selected policy uses the same evaluation event rule.
    from scripts.evaluate_dppo_uav_formal_endpoint import _build_adapter
    if file_hash(world["obstacle_path"]) != world["obstacle_sha256"]:
        raise ValueError("live obstacle bytes changed after bank freeze")
    adapter = _build_adapter(system, c, world["world_id"], Path(world["obstacle_path"]), role)
    env = adapter.environment
    env.exec_noise_scale = 0.
    env.disturb_enable = False
    env.mid_state_init_ratio = 0.
    env.terminate_at_space_limits = False
    if (env.reward_version != "v2" or env.n_exec_steps != 4 or env.rl_action_steps != 8
            or env.traj_dt != .3 or env.max_episode_steps != 60 or env.collision_thresh != .1):
        raise ValueError("common environment is not v2/H4/8")
    return adapter


def metadata(info):
    return [asdict(m) for m in info["pointcloud_meta_history"]]
