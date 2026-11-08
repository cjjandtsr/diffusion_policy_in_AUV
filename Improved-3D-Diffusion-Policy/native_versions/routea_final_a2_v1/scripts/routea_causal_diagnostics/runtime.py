"""Checkpoint comparability, paired forward, and read-only runtime guards."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import gc
import hashlib
from pathlib import Path
import random
import re
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch

from .contracts import (
    DiagnosticFailure,
    canonical_array_sha256,
    canonical_tree_sha256,
    capture_rng_state,
    load_strict_json,
    restore_rng_state,
    rng_state_sha256,
)


LORA_MARKERS = ("lora_A", "lora_B", ".lora_", "lora_down", "lora_up")


@dataclass(frozen=True)
class CheckpointContract:
    label: str
    update: int
    source_set_hash: str
    actor_state: Mapping[str, torch.Tensor]
    lora_state: Mapping[str, torch.Tensor]
    runtime_contract: Mapping[str, Any]
    checkpoint_identity: Mapping[str, Any]


@dataclass(frozen=True)
class ForwardRecord:
    variant: str
    full_action: np.ndarray
    normalized_chain: np.ndarray
    initial_chain_sha256: str
    post_call_rng_sha256: str
    input_sha256: str
    condition_sha256: str
    policy_input_hash: str | None


def _cpu_state_dict(state: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not torch.is_tensor(value):
            raise DiagnosticFailure(
                "CHECKPOINT_CONTRACT_MISMATCH", f"state tensor {name} is {type(value).__name__}"
            )
        result[str(name)] = value.detach().cpu().contiguous().clone()
    return result


def _tensor_record(value: torch.Tensor) -> Dict[str, Any]:
    array = value.detach().cpu().contiguous().numpy()
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": canonical_array_sha256(array),
    }


def _is_lora_name(name: str) -> bool:
    return any(marker in name for marker in LORA_MARKERS)


def validate_zero_effect_lora_state(
    state: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    """Validate the production LoRA initialization contract.

    The E5 adapters initialize ``lora_a`` with Kaiming weights and
    ``lora_b`` with exact zeros.  The adapter residual is therefore exactly
    zero even though the complete adapter state is intentionally nonzero.
    """

    if not state:
        raise DiagnosticFailure(
            "CHECKPOINT_CONTRACT_MISMATCH", "empty LoRA adapter state", exit_code=11
        )
    factors: Dict[str, set[str]] = {}
    counts = {"a": 0, "b": 0}
    for name, value in sorted(state.items()):
        if not torch.is_tensor(value):
            raise DiagnosticFailure(
                "CHECKPOINT_CONTRACT_MISMATCH",
                f"LoRA state tensor {name} is {type(value).__name__}",
                exit_code=11,
            )
        if not torch.isfinite(value.detach()).all().item():
            raise DiagnosticFailure(
                "CHECKPOINT_CONTRACT_MISMATCH",
                f"non-finite LoRA tensor {name}",
                exit_code=11,
            )
        lowered = name.lower()
        if lowered.endswith("lora_a"):
            factor = "a"
            prefix = lowered[: -len("lora_a")]
        elif lowered.endswith("lora_b"):
            factor = "b"
            prefix = lowered[: -len("lora_b")]
        else:
            raise DiagnosticFailure(
                "CHECKPOINT_CONTRACT_MISMATCH",
                f"unknown LoRA tensor name {name}",
                exit_code=11,
            )
        factors.setdefault(prefix, set()).add(factor)
        counts[factor] += 1
        if factor == "b" and torch.count_nonzero(value.detach().cpu()).item() != 0:
            raise DiagnosticFailure(
                "CHECKPOINT_CONTRACT_MISMATCH",
                f"nonzero lora_b tensor {name}",
                exit_code=11,
            )
    incomplete = sorted(prefix for prefix, pair in factors.items() if pair != {"a", "b"})
    if incomplete:
        raise DiagnosticFailure(
            "CHECKPOINT_CONTRACT_MISMATCH",
            f"unpaired LoRA factors: {incomplete}",
            exit_code=11,
        )
    return {
        "adapter_count": len(factors),
        "lora_a_tensor_count": counts["a"],
        "lora_b_tensor_count": counts["b"],
        "status": "ZERO_EFFECT_LORA",
    }


def compare_checkpoint_contracts(
    contracts: Sequence[CheckpointContract],
) -> Dict[str, Any]:
    if not contracts:
        raise DiagnosticFailure("CHECKPOINT_CONTRACT_MISMATCH", "empty checkpoint matrix")
    by_label = {contract.label: contract for contract in contracts}
    if len(by_label) != len(contracts) or "bc" not in by_label or "u0" not in by_label:
        raise DiagnosticFailure(
            "CHECKPOINT_CONTRACT_MISMATCH", "matrix requires unique labels including bc and u0"
        )
    baseline = by_label["bc"]
    baseline_non_lora = {
        name: value for name, value in baseline.actor_state.items() if not _is_lora_name(name)
    }
    baseline_lora_keys = set(baseline.lora_state)
    mismatches: list[Dict[str, Any]] = []
    runtime_hash = canonical_tree_sha256(baseline.runtime_contract)
    for contract in contracts:
        if contract.label == "bc":
            expected_update = 0
        elif contract.label == "u0":
            expected_update = 0
        else:
            match = re.fullmatch(r"seed[123]-u(8|16|24|32)", contract.label)
            expected_update = None if match is None else int(match.group(1))
        if expected_update is None or int(contract.update) != expected_update:
            mismatches.append(
                {
                    "label": contract.label,
                    "field": "update_label",
                    "expected": expected_update,
                    "actual": int(contract.update),
                }
            )
        # Production source_set_hash folds in the resolved config, so the
        # expected value is seed-specific.  Each contract was already matched
        # fail-closed against its own reconstructed executed-source manifest.
        actual_runtime_hash = canonical_tree_sha256(contract.runtime_contract)
        if actual_runtime_hash != runtime_hash:
            mismatches.append(
                {
                    "label": contract.label,
                    "field": "runtime_contract",
                    "expected_sha256": runtime_hash,
                    "actual_sha256": actual_runtime_hash,
                }
            )
        if set(contract.lora_state) != baseline_lora_keys:
            mismatches.append(
                {
                    "label": contract.label,
                    "field": "lora_key_set",
                    "expected": sorted(baseline_lora_keys),
                    "actual": sorted(contract.lora_state),
                }
            )
        actual_non_lora = {
            name: value for name, value in contract.actor_state.items() if not _is_lora_name(name)
        }
        if set(actual_non_lora) != set(baseline_non_lora):
            mismatches.append(
                {
                    "label": contract.label,
                    "field": "non_lora_key_set",
                    "missing": sorted(set(baseline_non_lora) - set(actual_non_lora)),
                    "extra": sorted(set(actual_non_lora) - set(baseline_non_lora)),
                }
            )
        for name in sorted(set(actual_non_lora).intersection(baseline_non_lora)):
            expected = _tensor_record(baseline_non_lora[name])
            actual = _tensor_record(actual_non_lora[name])
            if expected != actual:
                mismatches.append(
                    {
                        "label": contract.label,
                        "field": "non_lora_actor_tensor",
                        "tensor": name,
                        "expected": expected,
                        "actual": actual,
                    }
                )
    u0 = by_label["u0"]
    for name in sorted(set(u0.lora_state).intersection(baseline.lora_state)):
        expected = _tensor_record(baseline.lora_state[name])
        actual = _tensor_record(u0.lora_state[name])
        if expected != actual:
            mismatches.append(
                {
                    "label": "u0",
                    "field": "lora_initial_identity",
                    "tensor": name,
                    "expected": expected,
                    "actual": actual,
                }
            )
    try:
        zero_effect = validate_zero_effect_lora_state(u0.lora_state)
    except DiagnosticFailure as error:
        zero_effect = {"status": "CHECKPOINT_CONTRACT_MISMATCH"}
        mismatches.append(
            {
                "label": "u0",
                "field": "lora_zero_effect",
                "detail": error.detail,
            }
        )
    return {
        "schema": "routea_checkpoint_comparability_v2",
        "status": "COMPARABLE" if not mismatches else "CHECKPOINT_CONTRACT_MISMATCH",
        "baseline": baseline.label,
        "labels": [contract.label for contract in contracts],
        "source_set_hash": baseline.source_set_hash,
        "runtime_contract_sha256": runtime_hash,
        "non_lora_tensor_count": len(baseline_non_lora),
        "lora_key_count": len(baseline_lora_keys),
        "u0_lora_zero_effect": zero_effect,
        "mismatches": mismatches,
        "contracts": [
            {
                "label": contract.label,
                "update": contract.update,
                "source_set_hash": contract.source_set_hash,
                "checkpoint_identity": dict(contract.checkpoint_identity),
                "actor_state_sha256": canonical_tree_sha256(contract.actor_state),
                "lora_state_sha256": canonical_tree_sha256(contract.lora_state),
            }
            for contract in contracts
        ],
    }


def _jsonable_runtime_value(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable_runtime_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable_runtime_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_runtime_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return _tensor_record(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_source_hash(header: Mapping[str, Any]) -> str:
    for candidate in (
        header.get("source_set_hash"),
        (header.get("runtime_identity") or {}).get("source_set_hash"),
        (header.get("identity") or {}).get("source_set_hash"),
    ):
        if isinstance(candidate, str) and len(candidate) == 64:
            return candidate.lower()
    raise DiagnosticFailure(
        "CHECKPOINT_CONTRACT_MISMATCH", "checkpoint header has no source_set_hash"
    )


def _reconstruct_executed_source_hash(
    root: Path, config_path: Path | str
) -> str:
    """Rebuild the config-folded source identity recorded by training."""

    from omegaconf import OmegaConf

    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.identity import (
        build_executed_source_manifest,
    )

    config = OmegaConf.load(Path(config_path))
    manifest = config.get("admission_manifest")
    if not isinstance(manifest, str) or not manifest:
        raise DiagnosticFailure(
            "CHECKPOINT_CONTRACT_MISMATCH",
            "config has no admission_manifest; cannot reconstruct executed source manifest",
            exit_code=11,
        )
    try:
        result = build_executed_source_manifest(root, config, manifest)
        content_hash = result["content_hash"]
    except DiagnosticFailure:
        raise
    except Exception as exc:
        raise DiagnosticFailure(
            "CHECKPOINT_CONTRACT_MISMATCH",
            f"cannot reconstruct executed source manifest: {type(exc).__name__}: {exc}",
            exit_code=11,
        ) from exc
    if not isinstance(content_hash, str) or len(content_hash) != 64:
        raise DiagnosticFailure(
            "CHECKPOINT_CONTRACT_MISMATCH",
            "reconstructed executed source manifest has invalid content_hash",
            exit_code=11,
        )
    return content_hash.lower()


def inspect_checkpoint_contract(
    label: str,
    config_path: Path | str,
    checkpoint_path: Path | str | None,
    repo_root: Path | str,
) -> CheckpointContract:
    """Build and inspect one system, restoring outer RNG before returning."""

    root = Path(repo_root).resolve()
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path).resolve()
    actual_source_hash = _reconstruct_executed_source_hash(root, config_path)
    header_payload: Dict[str, Any] = {}
    if checkpoint is not None:
        header_payload = load_strict_json(checkpoint / "header.json", max_bytes=1024 * 1024)
        expected = _expected_source_hash(header_payload)
        if expected != actual_source_hash:
            raise DiagnosticFailure(
                "CHECKPOINT_CONTRACT_MISMATCH",
                f"{label}: checkpoint source={expected} reconstructed={actual_source_hash}",
                exit_code=11,
            )

    outer_rng = capture_rng_state()
    system = None
    try:
        from omegaconf import OmegaConf
        from diffusion_policy_3d.rl.dppo_uav_rebuild.deployment.loader import (
            load_deployment_policy,
        )
        from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.build_system import (
            build_system,
        )

        config = OmegaConf.load(Path(config_path))
        random.seed(int(config.run.seed))
        np.random.seed(int(config.run.seed))
        torch.manual_seed(int(config.run.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config.run.seed))
        if checkpoint is None:
            system = build_system(config, root)
            update = 0
        else:
            deployment = load_deployment_policy(config, checkpoint, root)
            system = deployment.system
            update = int(deployment.checkpoint_header.update_index)
        system.bc_runtime.policy.model.eval()
        system.critic.eval()
        actor_state = _cpu_state_dict(system.bc_runtime.policy.model.state_dict())
        lora_state = _cpu_state_dict(system.lora_bundle.adapter_state_dict())
        action_spec = system.rollout_sampler.action_spec
        scheduler = system.bc_runtime.policy.noise_scheduler
        normalizer = system.bc_runtime.policy.normalizer
        resolved_config = OmegaConf.to_container(config, resolve=True)
        # Seed-specific training configs legitimately differ in output_dir and
        # run.seed.  Those fields do not enter an already-restored actor
        # forward.  Record their full hash in checkpoint_identity, but exclude
        # only these two pre-registered paths from the inference comparability
        # contract; every other resolved field remains exact.
        inference_config = dict(resolved_config)
        inference_config.pop("output_dir", None)
        inference_config["run"] = dict(inference_config.get("run") or {})
        inference_config["run"].pop("seed", None)
        runtime_contract = {
            "inference_config": inference_config,
            "allowed_seed_config_differences": ["output_dir", "run.seed"],
            "action_spec": asdict(action_spec),
            "action_spec_hash": action_spec.content_hash,
            "actor_dtype": str(system.bc_runtime.policy.dtype),
            "actor_training": bool(system.bc_runtime.policy.model.training),
            "critic_training": bool(system.critic.training),
            "device": str(system.bc_runtime.device),
            "scheduler_class": type(scheduler).__module__ + "." + type(scheduler).__name__,
            "scheduler_config": _jsonable_runtime_value(getattr(scheduler, "config", {})),
            "timesteps": _jsonable_runtime_value(getattr(system.official_runtime, "ddim_t", [])),
            "normalizer": _jsonable_runtime_value(
                normalizer.state_dict() if hasattr(normalizer, "state_dict") else vars(normalizer)
            ),
            "bc_identity": _jsonable_runtime_value(system.bc_runtime.identity),
            "vendor_identity": _jsonable_runtime_value(system.vendor_identity),
        }
        checkpoint_identity = {
            "label": str(label),
            "path": None if checkpoint is None else str(checkpoint),
            "header_sha256": None if checkpoint is None else _file_sha256(checkpoint / "header.json"),
            "integrity_sha256": None if checkpoint is None else _file_sha256(checkpoint / "SHA256.json"),
            "update": update,
            "resolved_config_sha256": canonical_tree_sha256(resolved_config),
        }
        return CheckpointContract(
            label=str(label),
            update=update,
            source_set_hash=actual_source_hash,
            actor_state=actor_state,
            lora_state=lora_state,
            runtime_contract=runtime_contract,
            checkpoint_identity=checkpoint_identity,
        )
    finally:
        restore_rng_state(outer_rng)
        del system
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_read_only_system(
    config_path: Path | str,
    checkpoint_path: Path | str | None,
    repo_root: Path | str,
):
    """Load one production system after source identity has been checked."""

    root = Path(repo_root).resolve()
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path).resolve()
    actual_source_hash = _reconstruct_executed_source_hash(root, config_path)
    if checkpoint is not None:
        header = load_strict_json(checkpoint / "header.json", max_bytes=1024 * 1024)
        expected = _expected_source_hash(header)
        if expected != actual_source_hash:
            raise DiagnosticFailure(
                "CHECKPOINT_CONTRACT_MISMATCH",
                f"checkpoint source={expected} reconstructed={actual_source_hash}",
                exit_code=11,
            )
    from omegaconf import OmegaConf
    from diffusion_policy_3d.rl.dppo_uav_rebuild.deployment.loader import (
        load_deployment_policy,
    )
    from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.build_system import (
        build_system,
    )

    config = OmegaConf.load(Path(config_path))
    _seed_all(int(config.run.seed))
    if checkpoint is None:
        system = build_system(config, root)
    else:
        system = load_deployment_policy(config, checkpoint, root).system
    system.bc_runtime.policy.model.eval()
    system.critic.eval()
    if system.bc_runtime.policy.dtype != torch.float32:
        raise DiagnosticFailure(
            "CHECKPOINT_CONTRACT_MISMATCH", "diagnostic runtime must be FP32", exit_code=11
        )
    return config, system


def _normalized_state_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous()
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if isinstance(value, Mapping):
        pairs = []
        for key, item in value.items():
            key_token = f"{type(key).__name__}:{key}"
            pairs.append((key_token, _normalized_state_tree(item)))
        return {key: item for key, item in sorted(pairs)}
    if isinstance(value, tuple):
        return tuple(_normalized_state_tree(item) for item in value)
    if isinstance(value, list):
        return [_normalized_state_tree(item) for item in value]
    if is_dataclass(value):
        return _normalized_state_tree(asdict(value))
    if hasattr(value, "__dict__"):
        return _normalized_state_tree(vars(value))
    return value


def _system_state_hashes(system: Any) -> Dict[str, str]:
    actor = system.bc_runtime.policy.model.state_dict()
    critic = system.critic.state_dict()
    counters = vars(system.trainer.counters)
    hashes = {
        "actor": canonical_tree_sha256(_normalized_state_tree(actor)),
        "lora": canonical_tree_sha256(
            _normalized_state_tree(system.lora_bundle.adapter_state_dict())
        ),
        "critic": canonical_tree_sha256(_normalized_state_tree(critic)),
        "actor_optimizer": canonical_tree_sha256(
            _normalized_state_tree(system.actor_optimizer.state_dict())
        ),
        "critic_optimizer": canonical_tree_sha256(
            _normalized_state_tree(system.critic_optimizer.state_dict())
        ),
        "counters": canonical_tree_sha256(_normalized_state_tree(counters)),
    }
    runtime = getattr(system, "official_runtime", None)
    if runtime is not None and hasattr(runtime, "state_dict"):
        hashes["official_runtime"] = canonical_tree_sha256(
            _normalized_state_tree(runtime.state_dict())
        )
    sampler = getattr(system, "rollout_sampler", None)
    if sampler is not None and hasattr(sampler, "state_dict"):
        hashes["rollout_sampler"] = canonical_tree_sha256(
            _normalized_state_tree(sampler.state_dict())
        )
    scheduler = getattr(getattr(system, "bc_runtime", None), "policy", None)
    scheduler = getattr(scheduler, "noise_scheduler", None)
    if scheduler is not None:
        hashes["scheduler"] = canonical_tree_sha256(
            _normalized_state_tree(
                {
                    "config": _jsonable_runtime_value(getattr(scheduler, "config", {})),
                    "timesteps": _jsonable_runtime_value(getattr(scheduler, "timesteps", [])),
                }
            )
        )
    trainer = getattr(system, "trainer", None)
    for name in ("scaler", "schedules"):
        value = getattr(trainer, name, None)
        if value is not None:
            payload = value.state_dict() if hasattr(value, "state_dict") else value
            hashes[f"trainer_{name}"] = canonical_tree_sha256(
                _normalized_state_tree(payload)
            )
    return hashes


class TripwireGuard:
    def __init__(self, system: Any, *, immutable_files: Sequence[Path | str] = ()):
        self.system = system
        self.before = _system_state_hashes(system)
        self.rng_before = rng_state_sha256(capture_rng_state())
        self.files_before = {
            str(Path(path).resolve()): _file_sha256(Path(path).resolve())
            for path in immutable_files
        }
        self.forbidden_calls: list[str] = []

    def forbid(self, name: str):
        def blocked(*unused_args: Any, **unused_kwargs: Any) -> None:
            self.forbidden_calls.append(name)
            raise DiagnosticFailure("FORBIDDEN_TRAINING_CALL", name, exit_code=13)

        return blocked

    def verify(self, *, require_rng_unchanged: bool = False) -> Dict[str, Any]:
        after = _system_state_hashes(self.system)
        categories = {name: self.before[name] == after[name] for name in self.before}
        files_after = {
            path: _file_sha256(Path(path)) for path in self.files_before
        }
        files_unchanged = files_after == self.files_before
        rng_after = rng_state_sha256(capture_rng_state())
        rng_unchanged = rng_after == self.rng_before
        if not all(categories.values()) or not files_unchanged or (
            require_rng_unchanged and not rng_unchanged
        ):
            raise DiagnosticFailure(
                "INVALID_STATE_MUTATION",
                f"categories={categories} files_unchanged={files_unchanged} rng_unchanged={rng_unchanged}",
                exit_code=13,
            )
        return {
            "schema": "routea_read_only_tripwire_v1",
            "status": "READ_ONLY_VERIFIED",
            "categories": categories,
            "immutable_files_unchanged": files_unchanged,
            "rng_unchanged": rng_unchanged,
            "rng_required_unchanged": bool(require_rng_unchanged),
            "forbidden_call_count": len(self.forbidden_calls),
            "forbidden_calls": list(self.forbidden_calls),
        }


def install_read_only_tripwires(
    system: Any, *, immutable_files: Sequence[Path | str] = ()
) -> TripwireGuard:
    guard = TripwireGuard(system, immutable_files=immutable_files)
    system.actor_optimizer.step = guard.forbid("actor_optimizer.step")
    system.critic_optimizer.step = guard.forbid("critic_optimizer.step")
    system.trainer.update = guard.forbid("trainer.update")
    for name in ("save", "save_checkpoint", "checkpoint_save"):
        if hasattr(system, name):
            setattr(system, name, guard.forbid(f"system.{name}"))
    return guard


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _variant_condition(system: Any, observation: Mapping[str, Any], variant: Mapping[str, Any]):
    if "condition" in variant:
        return variant["condition"], variant.get("policy_input_hash")
    raw = {
        "agent_pos": np.asarray(variant["agent_pos"]),
        "point_cloud": np.asarray(
            variant.get("point_cloud", observation["point_cloud"])
        ),
    }
    encoded = system.observation_adapter.encode(
        raw,
        observation["pointcloud_meta_history"],
        episode_identity_hash=observation["episode_identity_hash"],
        decision_index=int(observation["decision_index"]),
    )
    system.observation_adapter.verify(encoded)
    provenance = getattr(encoded, "provenance", None)
    return encoded.condition, getattr(provenance, "policy_input_hash", None)


def paired_forward(
    system: Any,
    observation: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
    policy_seed: int,
) -> tuple[ForwardRecord, ...]:
    outer_rng = capture_rng_state()
    try:
        _seed_all(int(policy_seed))
        paired_state = capture_rng_state()
        records: list[ForwardRecord] = []
        for variant in variants:
            restore_rng_state(paired_state)
            condition, provenance_hash = _variant_condition(system, observation, variant)
            with torch.no_grad():
                rollout = system.rollout_sampler.sample(condition, deterministic=False)
            post_hash = rng_state_sha256(capture_rng_state())
            chain = rollout.normalized_chain.detach().cpu().contiguous().numpy()
            full = rollout.action.full.detach().cpu().contiguous().numpy()
            variant_agent_pos = np.asarray(
                variant.get("agent_pos", observation["agent_pos"])
            )
            variant_pc = np.asarray(
                variant.get("point_cloud", observation["point_cloud"])
            )
            records.append(
                ForwardRecord(
                    variant=str(variant["name"]),
                    full_action=full,
                    normalized_chain=chain,
                    initial_chain_sha256=canonical_array_sha256(chain[:, 0]),
                    post_call_rng_sha256=post_hash,
                    input_sha256=canonical_tree_sha256(
                        {"agent_pos": variant_agent_pos, "point_cloud": variant_pc}
                    ),
                    condition_sha256=canonical_tree_sha256(condition),
                    policy_input_hash=provenance_hash,
                )
            )
        return tuple(records)
    finally:
        restore_rng_state(outer_rng)


def verify_forward_pairing(records: Sequence[ForwardRecord]) -> Dict[str, Any]:
    initial = {record.initial_chain_sha256 for record in records}
    post = {record.post_call_rng_sha256 for record in records}
    if len(initial) != 1 or len(post) != 1:
        raise DiagnosticFailure(
            "INVALID_NOISE_PAIRING",
            f"initial_chain_hashes={sorted(initial)} post_rng_hashes={sorted(post)}",
            exit_code=12,
        )
    return {
        "status": "VALID_NOISE_PAIRING",
        "variant_count": len(records),
        "initial_chain_sha256": next(iter(initial)),
        "post_call_rng_sha256": next(iter(post)),
    }
