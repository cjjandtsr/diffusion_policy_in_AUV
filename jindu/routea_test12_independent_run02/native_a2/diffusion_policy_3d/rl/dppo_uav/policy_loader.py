"""BC checkpoint loader + LoRA state extraction.

load_ddpm_policy replicates the frozen eval loader
(scripts/eval_ddpm_bc_baseline.py:65-162, proven E040 path) inside the
dppo_uav package so the RL core never imports scripts/. The frozen policy
file itself is untouched.

extract_lora_state pulls the trainable LoRA params from the insert_lora
return dict: the adapter modules are NOT attached to the model (lora_adapters
docstring), so named_parameters() never sees them — the state must come from
the insertion report's lora_params list (§4.16).
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy_3d.policy.phase1a_policy import Phase1aPolicy

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_VALID_SCHEDULER_KEYS = {
    "num_train_timesteps", "beta_start", "beta_end", "beta_schedule",
    "trained_betas", "variance_type", "clip_sample", "prediction_type",
    "thresholding", "dynamic_thresholding_ratio", "clip_sample_range",
    "sample_max_value", "timestep_spacing", "steps_offset",
    "rescale_betas_zero_snr",
}

# E040-era auxiliary keys not present in this checkpoint (frozen loader filter)
_DUMMY_KEY_FILTER = (
    "_dummy_variable", "anchor_head", "k_stab", "lambda_jerk",
)

# ───────────────────────────────────────────────────────────────────
# R6-E §11.6.1 (F-06 → R6-LOAD-1): the weight-load contract.
# ───────────────────────────────────────────────────────────────────

# SHA256 of the REGISTERED BC checkpoint FILE BYTES — the R7/G8 base
# actor e5_lambda0p5 (E5 retrain, V11-fixed collision loss lambda_col=0.5,
# 2026-08-16): data/outputs/2026.08.16/08.53.26_e5_lambda0p5/checkpoints/
# latest.ckpt, measured 2026-08-22 (user ruling: e5 replaces the v8
# lambda0.5 baseline). The allowlist below is bound to THIS hash — a
# different checkpoint must be registered, not silently allowed through.
REGISTERED_BC_CHECKPOINT_HASH = (
    "c1fca4ac2735928b18fa7a9c069f90ad181eb84092acee379b073dc6a61bfc9a"
)

# Loader/schema version of the allowlist contract — a head-layout change
# must bump it (and re-measure the allowlist), never silently edit.
# v1: v8 baseline carried the 7-key anchor auxiliary head.
# v2: e5_lambda0p5 (192 state_dict keys, measured 2026-08-22) has NO
# anchor keys (cfg.policy.anchor_bank_path=None) — allowlist re-measured
# and emptied.
LOAD_CONTRACT_SCHEMA_VERSION = "v2"

# The EXACT unexpected keys of the registered checkpoint's model
# state_dict vs the runtime policy. e5_lambda0p5 measured 2026-08-22:
# 192 keys, zero anchor keys — the runtime Phase1aPolicy constructed from
# its own cfg.policy carries every key (missing=[], unexpected=[]), so
# the allowlist is EMPTY. Literal names only — wildcards are forbidden.
BC_ANCHOR_ALLOWLIST = ()

# Why the excluded keys never enter current inference — frozen into the
# contract at load time and into the eval artifact by the validator
# (§11.6.1 "被纳入 resolved config/hash/artifact"). A bare key list is
# not an explanation.
BC_COMPATIBILITY_MANIFEST = {
    "loader_schema_version": LOAD_CONTRACT_SCHEMA_VERSION,
    "summary": (
        "Registered base actor: e5_lambda0p5 (2026-08-16 E5 retrain, "
        "V11-fixed collision loss lambda_col=0.5; adopted 2026-08-22 by "
        "user ruling, Gazebo 3-way 79/100 vs v8 lambda0.5 54/100). "
        "Measured state_dict: 192 keys, no anchor keys "
        "(cfg.policy.anchor_bank_path=None) — the runtime policy "
        "constructed from this checkpoint's own cfg.policy loads with "
        "missing=[] and unexpected=[] (strict=False complete). The "
        "allowlist is empty; the strict=False load is provably total."
    ),
    "excluded_keys": {},
}


@dataclass
class PolicyLoadAudit:
    """Result of the R6-E load-contract validation (§11.6.1)."""

    missing_keys: tuple
    unexpected_keys: tuple
    allowlisted_keys: tuple
    bc_checkpoint_hash: str
    loader_schema_version: str
    compatibility_manifest: dict

    def to_artifact_dict(self) -> Dict[str, Any]:
        """The serializable load-contract carried by eval artifacts."""
        return {
            "bc_checkpoint_hash": self.bc_checkpoint_hash,
            "loader_schema_version": self.loader_schema_version,
            "compatibility_manifest": self.compatibility_manifest,
        }


def _file_sha256(path: str) -> str:
    """SHA256 of the checkpoint FILE BYTES (bound to the file — a
    repacked checkpoint changes the hash even if tensors are identical)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_policy_load_contract(
    *,
    bc_checkpoint_hash: str,
    missing_keys: Sequence[str],
    unexpected_keys: Sequence[str],
    compatibility_manifest: Dict[str, Any],
) -> PolicyLoadAudit:
    """§11.6.1 (R6-LOAD-1): enforce the weight-load contract.

    Called by BOTH the training and the evaluation entries — same
    function, same rule: ``real_missing == []`` AND
    ``set(unexpected) == BC_ANCHOR_ALLOWLIST``, with the manifest frozen
    into the audit. Any violation raises BEFORE LoRA insertion / env
    construction / rollout.

    Raises:
        ValueError: missing UNet key / unregistered BC hash / unexpected
            key not in the exact allowlist / allowlist subset (a missing
            anchor — no "may be present or absent") / drifted manifest.
    """
    if missing_keys:
        raise ValueError(
            f"load contract: {len(missing_keys)} missing weight keys "
            f"(first: {list(missing_keys)[:3]}) — real_missing must be []"
        )
    if bc_checkpoint_hash != REGISTERED_BC_CHECKPOINT_HASH:
        raise ValueError(
            f"unregistered BC checkpoint (sha256={bc_checkpoint_hash}): "
            "the allowlist is bound to the registered checkpoint "
            f"{REGISTERED_BC_CHECKPOINT_HASH}"
        )
    unexpected = tuple(unexpected_keys)
    if set(unexpected) != set(BC_ANCHOR_ALLOWLIST):
        missing_anchors = sorted(set(BC_ANCHOR_ALLOWLIST) - set(unexpected))
        extra = sorted(set(unexpected) - set(BC_ANCHOR_ALLOWLIST))
        raise ValueError(
            f"load contract: unexpected keys do not match the exact "
            f"allowlist (missing anchors: {missing_anchors}, extra: "
            f"{extra}) — exact-set allowlist, no wildcards"
        )
    if compatibility_manifest != BC_COMPATIBILITY_MANIFEST:
        raise ValueError(
            "load contract: compatibility manifest drifted from the "
            "frozen constant — every allowlisted key must keep its "
            "explanation"
        )
    return PolicyLoadAudit(
        missing_keys=tuple(missing_keys),
        unexpected_keys=unexpected,
        allowlisted_keys=BC_ANCHOR_ALLOWLIST,
        bc_checkpoint_hash=bc_checkpoint_hash,
        loader_schema_version=LOAD_CONTRACT_SCHEMA_VERSION,
        compatibility_manifest=dict(BC_COMPATIBILITY_MANIFEST),
    )


def load_ddpm_policy(ckpt_path: str, device: str = "cuda:0") -> Phase1aPolicy:
    """Load Phase1aPolicy (DDPM epsilon-pred UNet) from a BC checkpoint.

    Faithful port of the frozen eval loader (eval_ddpm_bc_baseline.py:65-162):
    manual construction from cfg.policy because cfg.policy._target_ may point
    at DiffusionPointcloudPolicy while the state_dict has Phase1aPolicy keys.

    Side effect: attaches ``policy._dppo_missing_keys``,
    ``policy._dppo_unexpected_keys`` and ``policy._dppo_load_contract``
    (additive attributes; the first two feed the §4.17 artifact's
    missing/unexpected weight-key fields, the third carries the R6-E
    §11.6.1 load contract into the artifact).
    """
    ckpt_path = os.path.join(_PROJECT_ROOT, ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    logger.info("loading BC checkpoint %s", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    policy_cfg = cfg.get("policy", cfg)
    if not policy_cfg:
        raise ValueError("Checkpoint missing 'cfg.policy' key")

    n_obs = ckpt.get("n_obs_steps", policy_cfg.get("n_obs_steps", 3))
    shape_meta = policy_cfg.get("shape_meta")
    if shape_meta is None:
        raise ValueError("Checkpoint missing shape_meta")

    # The runtime policy instance must run at the formal CFG pin 1.5.
    # Fast-fail pre-construction on the checkpoint's configured value; the
    # post-construction instance check below is the audit-mandated runtime
    # verification.
    cfg_lambda_cfg = policy_cfg.get("cfg_lambda", 1.5)
    if cfg_lambda_cfg != 1.5:
        raise ValueError(
            "formal CFG lambda is pinned to 1.5; checkpoint cfg.policy"
            f" cfg_lambda = {cfg_lambda_cfg}"
        )

    ns_cfg = policy_cfg.get("noise_scheduler")
    if ns_cfg is None:
        raise ValueError("Checkpoint missing noise_scheduler config")
    ns_dict = {
        k: v for k, v in ns_cfg.items()
        if not k.startswith("_") and k in _VALID_SCHEDULER_KEYS
    }
    noise_scheduler = DDPMScheduler(**ns_dict)

    policy = Phase1aPolicy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=policy_cfg.get("horizon", 16),
        n_action_steps=policy_cfg.get("n_action_steps", 8),
        n_obs_steps=n_obs,
        num_inference_steps=policy_cfg.get("num_inference_steps", 10),
        obs_as_global_cond=policy_cfg.get("obs_as_global_cond", True),
        diffusion_step_embed_dim=policy_cfg.get("diffusion_step_embed_dim", 256),
        down_dims=policy_cfg.get("down_dims", (256, 512, 1024)),
        kernel_size=policy_cfg.get("kernel_size", 5),
        n_groups=policy_cfg.get("n_groups", 8),
        condition_type=policy_cfg.get("condition_type") or "film",
        use_down_condition=policy_cfg.get("use_down_condition", True),
        use_mid_condition=policy_cfg.get("use_mid_condition", True),
        use_up_condition=policy_cfg.get("use_up_condition", True),
        use_pc_color=policy_cfg.get("use_pc_color", False),
        pointnet_type=policy_cfg.get("pointnet_type", "pointnet"),
        pointcloud_encoder_cfg=policy_cfg.get("pointcloud_encoder_cfg"),
        point_downsample=policy_cfg.get("point_downsample", False),
        anchor_bank_path=policy_cfg.get("anchor_bank_path"),
        anchor_loss_weight=policy_cfg.get("anchor_loss_weight", 0.5),
        cfg_dropout_prob=policy_cfg.get("cfg_dropout_prob", 0.15),
        cfg_lambda=cfg_lambda_cfg,
        truncated_steps=policy_cfg.get("truncated_steps", 50),
        truncation_ratio=policy_cfg.get("truncation_ratio", 1.0),
        use_projection=policy_cfg.get("use_projection", True),
        z_min=policy_cfg.get("z_min", 1.0),
        z_max=policy_cfg.get("z_max", 5.0),
        yaw_max_per_step=policy_cfg.get("yaw_max_per_step", 0.524),
        yaw_max_accumulated=policy_cfg.get("yaw_max_accumulated", 3.14159),
        use_obstacle_projection=policy_cfg.get("use_obstacle_projection", False),
        use_collision_loss=policy_cfg.get("use_collision_loss", False),
        training_phase=policy_cfg.get("training_phase", "joint"),
    )

    # Audit-mandated runtime instance check — verify the constructed policy
    # actually carries the formal pin (a constructor-side override would be
    # caught here, not at rollout).
    if policy.cfg_lambda != 1.5:
        raise ValueError(
            "loaded policy runtime instance cfg_lambda != 1.5 (got"
            f" {policy.cfg_lambda}) — formal CFG pin violated"
        )

    state_dicts = ckpt.get("state_dicts", {})
    state_dict = state_dicts.get("model", state_dicts.get("ema_model", {}))
    if not state_dict:
        raise ValueError("Checkpoint missing state_dicts.model")

    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    real_missing = [
        k for k in missing
        if not any(tag in k for tag in _DUMMY_KEY_FILTER)
    ]
    if real_missing:
        logger.warning(
            "missing %d non-dummy keys (first 3: %s)",
            len(real_missing), real_missing[:3],
        )
    # R6-E §11.6.1 (F-06 → R6-LOAD-1): the weight-load contract — the
    # SAME validation as the eval entry, before LoRA insertion / env
    # construction / rollout. Any violation raises here.
    audit = validate_policy_load_contract(
        bc_checkpoint_hash=_file_sha256(ckpt_path),
        missing_keys=real_missing,
        unexpected_keys=list(unexpected),
        compatibility_manifest=BC_COMPATIBILITY_MANIFEST,
    )
    policy._dppo_missing_keys = real_missing  # additive, for §4.17 artifact
    policy._dppo_unexpected_keys = list(unexpected)
    policy._dppo_load_contract = audit.to_artifact_dict()  # §11.6.1 artifact

    policy.to(device)
    policy.eval()
    logger.info(
        "loaded policy: %d params, device=%s",
        sum(p.numel() for p in policy.parameters()), device,
    )
    return policy


def extract_lora_state(lora_info: Dict[str, Any]) -> Dict[str, Any]:
    """Detached CPU clone of the trainable LoRA params (checkpoint §4.16).

    Args:
        lora_info: the dict returned by insert_lora (must contain
            "lora_params").

    Returns:
        {"lora_<i>": tensor} — the snapshot stored in the checkpoint and
        hashed for the eval artifact.

    [ASSUMPTION] Call this AFTER the PPO update: the checkpoint must
    hold the trained (live) state, never an insertion-time snapshot.
    """
    return {
        f"lora_{i}": p.detach().cpu().clone()
        for i, p in enumerate(lora_info["lora_params"])
    }


def restore_lora_state(lora_info: Dict[str, Any], lora_state: Dict[str, Any]) -> None:
    """Copy a saved LoRA snapshot back into the live adapter params.

    The adaptors are forward-hook closures invisible to policy.state_dict(),
    so the checkpoint's LoRA tensors are the only restorable copy of the
    trained actor. Copies in-place; deterministic given the insertion order.

    Args:
        lora_info: the dict returned by insert_lora (must contain
            "lora_params").
        lora_state: {"lora_<i>": tensor} as saved by extract_lora_state.

    Raises:
        ValueError: when the state keys do not match the adapter count.
    """
    params = lora_info["lora_params"]
    expected = [f"lora_{i}" for i in range(len(params))]
    if list(lora_state.keys()) != expected:
        raise ValueError(
            f"lora_state keys {list(lora_state.keys())} do not match "
            f"expected {expected}"
        )
    with torch.no_grad():
        for i, p in enumerate(params):
            p.copy_(lora_state[f"lora_{i}"].to(p.device))
