#!/usr/bin/env python3
"""artifact_identity.py — P0 artifact identity and config-freeze gate (Task 2).

Computes SHA-256 hashes for critical training artifacts, captures environment
information (PyTorch, CUDA, GPU), and compares per-rank training configs to
verify that no hyperparameter other than ``ft_lora_rank`` (and its derived
fields) differs across rank 4/16/32 runs.

Sub-report output file: ``artifact_hashes.json`` (written to ``<output_dir>/``).

Sub-report contract (imported from aggregator)
===============================================
    schema_version (str)
    run_id         (str)
    gate_pass      (bool)

Additional fields (this sub-report)
====================================
    artifacts      (dict):   name -> {path, sha256, bytes, note?}
    environment    (dict):   python, torch, cuda, gpu_name, gpu_mem_mib,
                             vcs, git_commit, dirty
    config_diff    (dict):   per-rank configs, differing keys, allowlist,
                             flagged_confounds
    notes          (list):   free-form notes / warnings
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.capacity_budget_p0.aggregator import (
    SCHEMA_VERSION,
    GATE_PASS_KEY,
    SUB_REPORT_CONTRACT,
)

# ── Constants ──────────────────────────────────────────────────────────────

ALLOWLISTED_RANK_DIFF_KEYS: List[str] = [
    "ft_lora_rank",
    "lora_param_shapes",
    "lora_trainable_count",
]
"""Keys that are PERMITTED to differ across rank configs.

``lora_effective_scale`` is deliberately excluded from this list — it must
appear in ``flagged_confounds["pending_task3_scaling_adjudication"]`` instead.
"""

# ── Helpers ────────────────────────────────────────────────────────────────


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return hex SHA-256 digest of the file at *path*.

    Reads the file in *chunk_size*-byte chunks (default 1 MiB).
    Raises ``FileNotFoundError`` if *path* does not exist.
    """
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _capture_environment() -> Dict[str, Any]:
    """Capture Python / PyTorch / CUDA / GPU environment info."""
    info: Dict[str, Any] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": None,
        "cuda": None,
        "gpu_name": None,
        "gpu_mem_mib": None,
        "vcs": "none",
        "git_commit": None,
        "dirty": None,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_mem_mib"] = int(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            )
    except ImportError:
        pass
    return info


def _hash_artifact(path_str: str) -> Dict[str, Any]:
    """Hash a single artifact path. Returns {path, sha256?, bytes?, error?}."""
    result: Dict[str, Any] = {"path": path_str}
    p = Path(path_str)
    if not p.is_file():
        result["error"] = "file_not_found"
        return result
    try:
        b = p.stat().st_size
        h = sha256_file(p)
        result["sha256"] = h
        result["bytes"] = b
    except (OSError, PermissionError) as e:
        result["error"] = str(e)
    return result


def _build_rank_configs(
    rank_configs: Dict[str, Dict],
) -> Dict[str, Dict]:
    """Build per-rank configs including LoRA-derived fields.

    For each entry in *rank_configs* (keyed like ``"rank4"``, ``"rank16"``,
    ``"rank32"``), computes:

        lora_effective_scale = 8.0 / ft_lora_rank

    If ``lora_param_shapes`` and ``lora_trainable_count`` are not already
    present in the config dict, they are computed via
    ``_compute_lora_params_analytic``.
    """
    built = {}
    for key, cfg in rank_configs.items():
        c = dict(cfg)
        rank = c.get("ft_lora_rank", 4)
        # LoRA effective scale = alpha / r; alpha is hard-coded as 8 in
        # create_actor_wrapper's adapter_config (train_a2_10k_capability.py:463).
        c["lora_effective_scale"] = 8.0 / max(rank, 1)

        if "lora_param_shapes" not in c or "lora_trainable_count" not in c:
            analytic = _compute_lora_params_analytic(rank)
            c["lora_param_shapes"] = analytic["shapes"]
            c["lora_trainable_count"] = analytic["trainable_count"]
        built[key] = c
    return built


def _compute_lora_params_analytic(rank: int) -> Dict:
    """Analytically compute LoRA parameter shapes and trainable count.

    Uses the known architecture of Phase1aPolicyFMV92:
      - ConditionalUNet1D with down_dims=(256, 512, 1024),
        kernel_size=5, n_groups=8 (Conv1dBlock).
      - LoRA targets (TARGET_PATTERNS_V1): ``mid_modules``, ``up_modules``,
        ``final_conv``.
      - Each matched ``nn.Conv1d(kernel=1)`` or ``nn.Linear`` gets an adapter.

    COUNTED TARGET GROUPS
    ---------------------
    * ``mid_modules`` — 2 residual blocks × 2 Conv1dBlocks × 1 Conv1d each
        (all 1024-channel): 4 Conv1d adapters.
    * ``up_modules`` — 2 decoder levels × 2 residual blocks × 2 Conv1dBlocks
        × 1 Conv1d each: 8 Conv1d adapters.  Dims per level derived from
        reversed ``in_out[1:]``.
    * ``final_conv`` — ``Conv1dBlock(start_dim→start_dim)`` followed by
        plain ``Conv1d(start_dim→input_dim, 1)``.  The Conv1dBlock contains
        1 Conv1d (k=5); the trailing plain Conv1d (k=1) is also a target.
        This gives 2 adapters at [start_dim×2, start_dim, input_dim].

    NOT COUNTED (also name-matched by TARGET_PATTERNS_V1 at runtime)
    ----------------------------------------------------------------
    * FiLM scale/shift Linear layers inside each ConditionalResidualBlock1D
        — they reside under ``.cond_encoder.*`` names and are *not* matched
        by the ``mid_modules`` / ``up_modules`` / ``final_conv`` substrings.
        If the runtime model has any such layers with matching names, the
        actual count will exceed this estimate.
    * ``down_modules`` layers — not in TARGET_PATTERNS_V1.
    * The ``Conv1dBlock``'s GroupNorm/Mish submodules (non-parameter layers).

    The v9_2 BC model uses n_obs_steps=8, which produces a
    global_cond_dim=1536 (12 features × 8 steps × 16 embedding) and
    input_dim (obs_feature_dim) of 11 (action dim, used as input to the
    final_conv output projection).

    Returns
    -------
    dict with keys:
        shapes: list of {in_dim, out_dim, module_name} entries
        trainable_count: total trainable parameters across all adapters
    """
    shapes: List[Dict] = []

    # ── mid_modules: 2 blocks, each (1024→1024) × 2 convs = 4 adapters ──
    mid_dim = 1024
    for blk_idx in range(2):
        shapes.append({
            "module_name": f"mid_modules.{blk_idx}.block.conv1d.0",
            "module_type": "Conv1d",
            "in_dim": mid_dim,
            "out_dim": mid_dim,
        })
        shapes.append({
            "module_name": f"mid_modules.{blk_idx}.block.conv1d.1",
            "module_type": "Conv1d",
            "in_dim": mid_dim,
            "out_dim": mid_dim,
        })

    # ── up_modules: decoder blocks ──
    # in_out from down_dims: [(input_dim, 256), (256, 512), (512, 1024)]
    # reversed(in_out[1:]) = [(512, 1024), (256, 512)]
    # For level pair (dim_in, dim_out):
    #   block0: ConditionalResidualBlock1D(dim_out*2, dim_in)
    #     → 2 Conv1d: (dim_out*2→dim_in), (dim_in→dim_in)
    #   block1: ConditionalResidualBlock1D(dim_in, dim_in)
    #     → 2 Conv1d: (dim_in→dim_in), (dim_in→dim_in)
    in_out_from_downs = [(256, 256), (256, 512), (512, 1024)]
    reversed_pairs = list(reversed(in_out_from_downs[1:]))  # [(512, 1024), (256, 512)]

    for pair_idx, (dim_in, dim_out) in enumerate(reversed_pairs):
        # block0: in=dim_out*2, out=dim_in
        b0_in = dim_out * 2
        b0_out = dim_in
        shapes.append({
            "module_name": f"up_modules.{pair_idx}.0.block.conv1d.0",
            "module_type": "Conv1d",
            "in_dim": b0_in,
            "out_dim": b0_out,
        })
        shapes.append({
            "module_name": f"up_modules.{pair_idx}.0.block.conv1d.1",
            "module_type": "Conv1d",
            "in_dim": b0_out,
            "out_dim": b0_out,
        })
        # block1: in=dim_in, out=dim_in
        b1_dim = dim_in
        shapes.append({
            "module_name": f"up_modules.{pair_idx}.1.block.conv1d.0",
            "module_type": "Conv1d",
            "in_dim": b1_dim,
            "out_dim": b1_dim,
        })
        shapes.append({
            "module_name": f"up_modules.{pair_idx}.1.block.conv1d.1",
            "module_type": "Conv1d",
            "in_dim": b1_dim,
            "out_dim": b1_dim,
        })

    # ── final_conv: Conv1dBlock(start_dim→start_dim) + Conv1d(start_dim→input_dim) ──
    start_dim = 256
    input_dim = 11  # v9_2 action dim (11d control)
    shapes.append({
        "module_name": "final_conv.0.block.conv1d.0",
        "module_type": "Conv1d",
        "in_dim": start_dim,
        "out_dim": start_dim,
    })
    shapes.append({
        "module_name": "final_conv.1",
        "module_type": "Conv1d",
        "in_dim": start_dim,
        "out_dim": input_dim,
    })

    # Each adapter has r * (in_dim + out_dim) trainable params
    total_trainable = 0
    for s in shapes:
        total_trainable += rank * (s["in_dim"] + s["out_dim"])

    return {
        "shapes": shapes,
        "trainable_count": total_trainable,
        "_note": "Estimated analytically from UNet architecture. "
                 "Exact count depends on input_dim and FiLM linear layer membership "
                 "of ConditionalResidualBlock1D. Counted target groups: "
                 "mid_modules (4 adapters), up_modules (8 adapters), "
                 "final_conv (2 adapters).",
    }


def _pairwise_diff_config_allowlist(
    rank_configs: Dict[str, Dict],
) -> Dict:
    """Compute pairwise differences across rank configs with allowlist.

    Parameters
    ----------
    rank_configs : dict
        Mapping from rank label (e.g. ``"rank4"``) to resolved config dict.

    Returns
    -------
    dict with keys:
        per_rank_config (dict): the full config for each rank (scalar-only subset).
        differing_keys (list): keys that differ across at least one pair.
        allowlisted_diff_keys (list): differing keys that ARE in the allowlist.
        non_allowlisted_diff_keys (list): differing keys NOT in the allowlist.
        flagged_confounds (list): entries with {key, reason} for confounds that
            need task-level adjudication.
    """
    labels = list(rank_configs.keys())
    if len(labels) < 2:
        return {
            "per_rank_config": dict(rank_configs),
            "differing_keys": [],
            "allowlisted_diff_keys": [],
            "non_allowlisted_diff_keys": [],
            "flagged_confounds": [],
        }

    # Collect all keys across all configs
    all_keys: set = set()
    for cfg in rank_configs.values():
        all_keys.update(cfg.keys())

    # Determine which keys differ across ranks
    differing_keys: List[str] = []
    for key in sorted(all_keys):
        values = [rank_configs[l].get(key) for l in labels]
        if len(set(str(v) for v in values)) > 1:
            differing_keys.append(key)

    # ── Flagged confounds (must be determined BEFORE split) ──────────
    # ``lora_effective_scale`` differs by definition across ranks but is
    # NOT an accidental misconfiguration — it is a direct mathematical
    # consequence of rank.  Task 3 (LoRA scaling audit) adjudicates it.
    FLAGGED_CONFOUND_KEYS = {"lora_effective_scale"}
    flagged_confounds: List[Dict] = []
    for fk in FLAGGED_CONFOUND_KEYS:
        if fk in differing_keys:
            flagged_confounds.append({
                "key": fk,
                "reason": "pending_task3_scaling_adjudication",
                "values": {l: rank_configs[l].get(fk) for l in labels},
            })

    # Split into allowlisted vs non-allowlisted (flagged confounds excluded
    # from the non-allowlisted set — they do NOT fail the gate).
    allow_set = set(ALLOWLISTED_RANK_DIFF_KEYS)
    flagged_set = FLAGGED_CONFOUND_KEYS
    allowlisted = [k for k in differing_keys if k in allow_set]
    non_allowlisted = [
        k for k in differing_keys
        if k not in allow_set and k not in flagged_set
    ]

    # Extract per-rank scalar configs for the report
    per_rank_config = {}
    for label in labels:
        cfg = dict(rank_configs[label])
        # Keep only JSON-serializable scalar values
        clean = {}
        for k, v in cfg.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                clean[k] = v
            elif isinstance(v, dict):
                # Only store small dicts (like role_ratios)
                if all(isinstance(sv, (str, int, float, bool)) for sv in v.values()):
                    clean[k] = v
        per_rank_config[label] = clean

    return {
        "per_rank_config": per_rank_config,
        "differing_keys": differing_keys,
        "allowlisted_diff_keys": allowlisted,
        "non_allowlisted_diff_keys": non_allowlisted,
        "flagged_confounds": flagged_confounds,
    }


# ── Public entry point ────────────────────────────────────────────────────


def run_artifact_identity_audit(config: dict) -> dict:
    """Run the artifact-identity gate and return the sub-report dict.

    Parameters
    ----------
    config : dict
        Required keys:
            output_dir (str) — Directory for the output ``artifact_hashes.json``.
            run_id (str) — Run identifier.

            artifact_paths (dict) — Mapping of artifact name (str) to filesystem
                path (str). Expected names: ``bc_ckpt``, ``b1_critic``,
                ``b0_replay``, ``reward_norm``, ``training_config_source``,
                ``world_manifest``, ``current_bank``.

            rank_configs (dict) — Mapping from rank label (e.g. ``"rank4"``)
                to config dict.  Each inner dict must contain at least
                ``ft_lora_rank`` and all hyperparameters to be compared.
                The audit computes ``lora_effective_scale`` automatically.

    Returns
    -------
    dict
        Sub-report dict also saved to ``<output_dir>/artifact_hashes.json``.
    """
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = config.get("run_id", output_dir.name)

    # ── 1. Hash artifacts ──────────────────────────────────────────────
    artifact_paths: Dict[str, str] = config.get("artifact_paths", {})
    artifacts: Dict[str, Dict] = {}
    all_hashed_ok = True

    for name, path_str in sorted(artifact_paths.items()):
        info = _hash_artifact(path_str)
        artifacts[name] = info
        if "error" in info:
            all_hashed_ok = False

    # ── 2. Capture environment ─────────────────────────────────────────
    environment = _capture_environment()

    # ── 3. Build per-rank configs and diff ──────────────────────────────
    raw_rank_configs: Dict[str, Dict] = config.get("rank_configs", {})
    rank_configs_resolved = _build_rank_configs(raw_rank_configs)
    config_diff = _pairwise_diff_config_allowlist(rank_configs_resolved)

    # ── 4. Compute gate pass ───────────────────────────────────────────
    non_allowlisted = config_diff.get("non_allowlisted_diff_keys", [])
    no_artifact_errors = all(
        "error" not in info for info in artifacts.values()
    )
    gate_pass = (len(non_allowlisted) == 0) and no_artifact_errors

    # ── 5. Build report ────────────────────────────────────────────────
    notes: List[str] = []

    # Check all artifacts
    for name, info in artifacts.items():
        if "error" in info:
            notes.append(f"Artifact '{name}': {info['error']}")

    # Check C2-OOC note (rank hardcode fixed 2026-07-14)
    for label, cfg in raw_rank_configs.items():
        if cfg.get("ft_c2_ooc"):
            notes.append(
                f"C2-OOC mode active for {label}: ft_lora_rank is now RESPECTED from "
                f"CLI (rank={cfg.get('ft_lora_rank')}). The prior hardcode "
                f"(ft_lora_rank=16 regardless of CLI) was fixed on 2026-07-14 via a "
                f"sentinel default in run_routea_rl_formal_train, so rank in {{4,16,32}} "
                f"variants are no longer collapsed to 16."
            )
            break

    # Manifest ambiguity note
    wm_info = artifacts.get("world_manifest", {})
    if "note" in wm_info:
        notes.append(wm_info["note"])

    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        GATE_PASS_KEY: gate_pass,
        "artifacts": artifacts,
        "environment": environment,
        "config_diff": config_diff,
        "notes": notes if notes else None,
    }

    # ── 6. Persist ─────────────────────────────────────────────────────
    out_path = output_dir / "artifact_hashes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result
