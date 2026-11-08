#!/usr/bin/env python3
"""lora_scaling.py — LoRA scaling audit for P0 capacity budget.

Two entry points:
  1. evaluate_scaling_gate(...) — pure function: aggregation + gate logic.
  2. run_lora_scaling_audit(config) — real GPU probe: loads policy,
     runs K updates per rank, records metrics, writes JSON.
"""

import copy
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Imports from the project
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # ~/AUV/.../Improved-3D-Diffusion-Policy
sys.path.insert(0, str(_REPO_ROOT))

from scripts.capacity_budget_p0 import SCHEMA_VERSION, GATE_PASS_KEY

# These are imported lazily inside run_lora_scaling_audit to avoid eager
# CUDA imports when only the pure function is needed.

# ---------------------------------------------------------------------------
# Constants for the probe
# ---------------------------------------------------------------------------

PROBE_RANKS = [4, 16, 32]
PROBE_UPDATES_K = 30
PROBE_BATCH_INDICES = [0, 1]  # deterministic indices into replay buffer
ACTOR_LR = 5e-6  # formal --ft_actor_lr default (train_a2_10k_capability.py:4628)
OPTIMIZER_KWARGS = {}  # torch.optim.Adam default (no weight_decay, no betas override)
RATIO_MAX = 4.0
MIN_UPDATE_NORM_FRACTION = 0.25  # high-rank must be >= 25% of rank4

# ---------------------------------------------------------------------------
# Pure gate / aggregation logic (testable without GPU)
# ---------------------------------------------------------------------------


def evaluate_scaling_gate(
    per_rank_u_tilde: Dict[int, float],
    per_rank_update_norm: Dict[int, float],
    non_finite_count: int,
    insertion_sets: Dict[int, Set[str]],
) -> Dict[str, Any]:
    """Evaluate the four LoRA scaling gate conditions.

    Parameters
    ----------
    per_rank_u_tilde : dict
        Maps rank -> median u_r across identical inserted module paths.
    per_rank_update_norm : dict
        Maps rank -> total actor update norm over the K updates.
    non_finite_count : int
        Number of non-finite values encountered during the probe.
    insertion_sets : dict
        Maps rank -> set of module paths that received adapters.

    Returns
    -------
    dict
        {
            "gate_pass": bool,
            "ratio": float,
            "min_u_tilde": float,
            "max_u_tilde": float,
            "update_norm_ratios_vs_rank4": dict or None,
            "gate_conditions": {
                "ratio_leq_4": bool,
                "high_rank_update_norm_sufficient": bool,
                "non_finite_ok": bool,
                "insertion_sets_identical": bool,
            },
            "next_action": str or None,
            "stop_gpu_main_matrix": bool,
        }
    """
    # --- Condition 1: ratio <= 4.0 ---
    min_u = min(per_rank_u_tilde.values()) if per_rank_u_tilde else 0.0
    max_u = max(per_rank_u_tilde.values()) if per_rank_u_tilde else 0.0
    ratio = max_u / max(min_u, 1e-12)
    ratio_leq_4 = ratio <= RATIO_MAX

    # --- Condition 2: high-rank update norm not systematically below 25% of rank4 ---
    update_norm_ratios = None
    high_rank_update_norm_sufficient = True
    if 4 in per_rank_update_norm and per_rank_update_norm[4] > 0:
        norm_r4 = per_rank_update_norm[4]
        update_norm_ratios = {}
        for r in PROBE_RANKS:
            update_norm_ratios[r] = per_rank_update_norm[r] / norm_r4
        for r in [16, 32]:
            if r in update_norm_ratios and update_norm_ratios[r] < MIN_UPDATE_NORM_FRACTION:
                high_rank_update_norm_sufficient = False
                break
    else:
        high_rank_update_norm_sufficient = False

    # --- Condition 3: non-finite count = 0 ---
    non_finite_ok = non_finite_count == 0

    # --- Condition 4: insertion sets are identical across ranks ---
    insertion_sets_identical = True
    if insertion_sets:
        sets_list = list(insertion_sets.values())
        if sets_list:
            first_set = sets_list[0]
            for s in sets_list[1:]:
                if s != first_set:
                    insertion_sets_identical = False
                    break

    gate_pass = (
        ratio_leq_4
        and high_rank_update_norm_sufficient
        and non_finite_ok
        and insertion_sets_identical
    )

    next_action = None
    stop_gpu_main_matrix = False
    if not gate_pass:
        next_action = "DESIGN_SEPARATE_SCALING_ABLATION"
        stop_gpu_main_matrix = True

    return {
        GATE_PASS_KEY: gate_pass,
        "ratio": ratio,
        "min_u_tilde": min_u,
        "max_u_tilde": max_u,
        "update_norm_ratios_vs_rank4": update_norm_ratios,
        "gate_conditions": {
            "ratio_leq_4": ratio_leq_4,
            "high_rank_update_norm_sufficient": high_rank_update_norm_sufficient,
            "non_finite_ok": non_finite_ok,
            "insertion_sets_identical": insertion_sets_identical,
        },
        "next_action": next_action,
        "stop_gpu_main_matrix": stop_gpu_main_matrix,
    }


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------


def _safe_fro(t: torch.Tensor) -> Tuple[float, int]:
    """Frobenius norm, counting non-finite entries instead of hiding them.

    Returns
    -------
    fro_val : float
        Frobenius norm of the finite entries, or NaN if all entries are
        non-finite.
    n_non_finite : int
        Number of non-finite entries in the tensor (NaN or Inf).  The caller
        MUST feed this into the gate's non_finite_count.
    """
    fin = torch.isfinite(t)
    n_non_finite = int((~fin).sum().item())
    t_clean = t[fin]
    if t_clean.numel() == 0:
        return float("nan"), n_non_finite
    return t_clean.pow(2).sum().sqrt().item(), n_non_finite


def _module_path(prefix: str, mod_name: str) -> str:
    """Compose a canonical module path from model prefix and named_modules key."""
    return f"{prefix}.{mod_name}" if prefix else mod_name


def _deep_getattr(obj: nn.Module, path: str):
    """Safe dotted-path attribute access."""
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# Main audit function
# ---------------------------------------------------------------------------


def run_lora_scaling_audit(config: dict) -> dict:
    """Run the LoRA scaling audit probe and write the sub-report JSON.

    Parameters
    ----------
    config : dict
        Must contain at least ``output_dir`` (str).  May also contain
        ``run_id`` (str), ``device`` (str, default 'cuda:0'), etc.

    Returns
    -------
    dict
        The full sub-report dict (also written to
        ``<output_dir>/lora_scaling_audit.json``).
    """
    output_dir = Path(config["output_dir"])
    run_id = config.get("run_id", output_dir.name)
    device_str = config.get("device", "cuda:0")
    updates_k = config.get("updates_k", PROBE_UPDATES_K)
    batch_indices = config.get("batch_indices", PROBE_BATCH_INDICES)

    # Infer device availability
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device_used = device_str

    print(f"[lora_scaling] device={device_used}, updates_k={updates_k}, "
          f"batch_indices={batch_indices}")

    # ------------------------------------------------------------------
    # Lazy imports (avoids loading the entire project for pure-function use)
    # ------------------------------------------------------------------
    from diffusion_policy_3d.policy.phase1a_policy_fm_rl import Phase1aPolicyFM
    from diffusion_policy_3d.policy.routea_reinflow_fm_policy import (
        RouteAReinFlowFMPolicy,
        TARGET_PATTERNS_V1,
        compute_routea_actor_losses,
    )
    from diffusion_policy_3d.policy.reinflow_noise_module import ReinFlowNoiseModule
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    # ------------------------------------------------------------------
    # 1. Load the base FM policy
    # ------------------------------------------------------------------
    ckpt_path = config.get(
        "bc_ckpt",
        "/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy/"
        "data/outputs/2026.06.17/15.22.50_v9_2_nobs8_bc/checkpoints/latest.ckpt",
    )
    print(f"[lora_scaling] Loading base policy from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    sd = ckpt["state_dicts"]["model"]  # use model, not ema (matching training init)
    pc = cfg["policy"]

    n_obs = cfg.get("n_obs_steps", 8)
    base_policy = Phase1aPolicyFM(
        shape_meta=cfg["shape_meta"],
        noise_scheduler=DDIMScheduler(
            num_train_timesteps=50, beta_start=0.0001, beta_end=0.02,
            beta_schedule="squaredcos_cap_v2", clip_sample=True,
            set_alpha_to_one=True, steps_offset=0, prediction_type="sample",
        ),
        horizon=cfg["horizon"], n_action_steps=cfg["n_action_steps"],
        n_obs_steps=n_obs,
        obs_as_global_cond=pc.get("obs_as_global_cond", True),
        diffusion_step_embed_dim=pc.get("diffusion_step_embed_dim", 128),
        down_dims=pc.get("down_dims", (256, 512, 1024)),
        kernel_size=pc.get("kernel_size", 5), n_groups=pc.get("n_groups", 8),
        anchor_bank_path=None,
        cfg_lambda=1.5, num_fm_steps=4, sigma_min=0.005, fm_t_scale=50,
        solver="euler", fm_anchor_t_start=0.95, use_pc_color=False,
        pointnet_type="multi_stage_pointnet",
        pointcloud_encoder_cfg=pc.get("pointcloud_encoder_cfg", None),
        point_downsample=pc.get("point_downsample", True),
    )
    missing, unexpected = base_policy.load_state_dict(sd, strict=False)
    # Filter expected anchor-related misses
    anchor_missing = [k for k in missing if "anchor" in k.lower()]
    other_missing = [k for k in missing if "anchor" not in k.lower()]
    if other_missing:
        print(f"  [WARN] Missing keys (non-anchor): {other_missing}")
    if unexpected and len(unexpected) > 5:
        # Some unexpected obs_encoder keys are expected when switching modes
        pass

    base_policy.eval()
    base_policy.requires_grad_(False)  # freeze all base params
    print(f"  Base policy loaded. param count: "
          f"{sum(p.numel() for p in base_policy.parameters())/1e6:.2f}M")

    # ------------------------------------------------------------------
    # 2. Build a fixed probe batch
    #    We extract global_cond from the replay buffer to avoid running
    #    the obs_encoder (saves GPU memory).
    # ------------------------------------------------------------------
    replay_path = config.get(
        "b0_replay",
        "/home/cjj/AUV/program/shiyan/results_routeA_RL_b0b1_rerun_v13_20260710/"
        "b0_collect_rolefix/replay.pt",
    )
    print(f"[lora_scaling] Loading replay buffer from {replay_path}")
    with open(replay_path, "rb") as f:
        replay_data = pickle.load(f)
    transitions = replay_data["transitions"]
    print(f"  Replay contains {len(transitions)} transitions")

    # Build global_cond tensor from fixed indices
    global_cond_list = []
    for idx in batch_indices:
        t = transitions[idx]
        gc = torch.from_numpy(t.obs_feature_t).float()  # (1536,)
        global_cond_list.append(gc)
    fixed_global_cond = torch.stack(global_cond_list, dim=0)  # (B, 1536)

    # Build a dummy teacher_action (needed if BC loss is used)
    H_action = base_policy.n_action_steps
    Da = base_policy.action_dim
    dummy_teacher_action = torch.zeros(len(batch_indices), H_action, Da)

    print(f"  Fixed batch: global_cond {list(fixed_global_cond.shape)}, "
          f"teacher_action {list(dummy_teacher_action.shape)}")

    # ------------------------------------------------------------------
    # 3. Per-rank probe
    # ------------------------------------------------------------------
    per_rank_results: Dict[int, dict] = {}
    grand_non_finite_count = 0
    grand_insertion_sets: Dict[int, Set[str]] = {}
    per_rank_update_norm: Dict[int, float] = {}
    per_rank_u_tilde: Dict[int, float] = {}
    H_action = base_policy.n_action_steps
    Da = base_policy.action_dim

    for rank_idx, rank in enumerate(PROBE_RANKS):
        print(f"\n{'='*60}")
        print(f"[lora_scaling] Rank {rank} probe (rank {rank_idx+1}/{len(PROBE_RANKS)})")
        print(f"{'='*60}")

        # --- Seed identically for each rank ---
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # --- Deep copy base on CPU (clean hook isolation between ranks) ---
        base_copy = copy.deepcopy(base_policy)
        base_copy.eval()
        base_copy.requires_grad_(False)

        # --- Fresh noise module ---
        noise_module = ReinFlowNoiseModule(
            num_fm_steps=4, sigma_min=1e-4, sigma_max=0.10, init_sigma=0.01,
        )

        # --- Create actor with LoRA at explicit rank ---
        actor = RouteAReinFlowFMPolicy(
            base_policy=base_copy,
            teacher_policy=base_copy,  # won't be used (compute_teacher=False)
            noise_module=noise_module,
            adapter_config={
                "lora_rank": rank,
                "lora_alpha": 8,
                "target_patterns": TARGET_PATTERNS_V1,
            },
            freeze_base=True,
            train_anchor_head=False,
            return_path=False,
        )

        actor.to(device_used)
        actor.train()

        # Record alpha and scaling
        alpha = actor.adapter_config["lora_alpha"]
        scaling = alpha / max(rank, 1)

        print(f"  LoRA: rank={rank}, alpha={alpha}, scaling={scaling:.4f}, "
              f"n_adapters={len(actor._lora_adaptors)}")

        # --- Record insertion set ---
        report = actor.get_adapter_report()
        module_paths = set()
        layers_info = []
        for layer in report["inserted_layers"]:
            mod_path = _module_path("model", layer["module_name"])
            module_paths.add(mod_path)
            layers_info.append(layer)

        # Record layer-level init metrics
        pre_layers = []
        for adaptor, layer_dict in zip(actor._lora_adaptors, layers_info):
            A = adaptor.lora_A.data
            B = adaptor.lora_B.data
            A_fro, nf = _safe_fro(A)
            grand_non_finite_count += nf
            B_fro, nf = _safe_fro(B)
            grand_non_finite_count += nf

            # Effective ΔW = scaling * (A @ B)
            eff_dW = scaling * (A @ B)
            dW_fro, nf = _safe_fro(eff_dW)
            grand_non_finite_count += nf

            # Base weight W0
            mod_name = layer_dict["module_name"]
            target_module = _deep_getattr(base_copy.model, mod_name)
            W0 = target_module.weight.data
            W0_fro, nf = _safe_fro(W0)
            grand_non_finite_count += nf

            n_trainable = layer_dict["trainable_params"]
            u_r = dW_fro / max(W0_fro, 1e-12) if W0_fro > 0 else 0.0

            pre_layers.append({
                "module_path": f"model.{mod_name}",
                "rank": rank,
                "alpha": alpha,
                "scaling": scaling,
                "A_shape": list(A.shape),
                "B_shape": list(B.shape),
                "trainable_count": n_trainable,
                "W0_fro": W0_fro,
                "A_fro": A_fro,
                "B_fro": B_fro,
                "dW_fro": dW_fro,
                "u_r": u_r,
            })

        grand_insertion_sets[rank] = module_paths

        # --- Build optimizer: torch.optim.Adam (NOT AdamW) to match training ---
        #    train_a2_10k_capability.py:4864 uses Adam; AdamW weight_decay would
        #    bias high-rank adapters (more params → more decay).
        optim = torch.optim.Adam(
            actor.get_trainable_param_groups(lr=ACTOR_LR),
            **OPTIMIZER_KWARGS,
        )

        # Capture initial state for update norm measurement
        init_params = {}
        for name, p in actor.named_parameters():
            if p.requires_grad:
                init_params[name] = p.data.clone().cpu()

        # --- Pre-compute the fixed target trajectory via deterministic FM Euler ---
        #    We run ONE deterministic denoising pass on the frozen base model
        #    (LoRA zero-init, so equivalent to the original policy) to obtain a
        #    realistic full-horizon target trajectory.  This mirrors BC fine-tuning
        #    where the frozen reference produces the regression target.
        model = base_copy.model
        B = fixed_global_cond.shape[0]
        T = base_copy.horizon       # 16
        Da = base_copy.action_dim   # 11
        n_steps = base_copy.num_fm_steps   # 4
        fm_t_scale = base_copy.fm_t_scale  # 50
        sigma_min = base_copy.sigma_min    # 0.005
        device = torch.device(device_used)

        with torch.no_grad():
            x_t = torch.randn(B, T, Da, device=device)
            t_end = 1.0 - sigma_min
            ts = torch.linspace(0.0, t_end, n_steps + 1, device=device)
            for i in range(n_steps):
                t_i = ts[i]
                dt_i = (ts[i + 1] - t_i).item()
                t_input = (t_i * fm_t_scale).float().expand(B)
                v = model(sample=x_t, timestep=t_input,
                          local_cond=None, global_cond=fixed_global_cond.to(device))
                x_t = x_t + v * dt_i
            trajectory = x_t[..., :Da]  # (B, T, Da) normalized target

        # --- Run K updates with REAL FM velocity-prediction loss ---
        #    Mirrors Phase1aPolicyFM.compute_loss() ~L594:
        #      noise ~ N(0,I), t ~ U[sigma_min, 1]
        #      x_t = (1-t)*noise + t*trajectory
        #      target_v = trajectory - noise
        #      loss = MSE(model(x_t, t_input, global_cond), target_v)
        non_finite = 0
        step_update_norms = []
        t_start = time.time()

        for step in range(updates_k):
            optim.zero_grad()

            # Sample noise and timestep
            noise = torch.randn_like(trajectory)
            t = torch.rand(B, device=device) * (1.0 - sigma_min) + sigma_min

            # Interpolate: x_t = (1-t)*noise + t*trajectory
            x_t = (1 - t.view(B, 1, 1)) * noise + t.view(B, 1, 1) * trajectory

            # Velocity target = trajectory - noise  (k_stab=0, the default)
            target_v = trajectory - noise

            # Scaled timestep
            t_input = (t * fm_t_scale).float()

            # Model prediction (LoRA adapters fire via forward hooks)
            pred_v = model(
                sample=x_t, timestep=t_input,
                local_cond=None, global_cond=fixed_global_cond.to(device),
            )

            # MSE on velocity field (exact match to compute_loss)
            loss = nn.functional.mse_loss(pred_v, target_v)

            # Check for non-finite values
            if not torch.isfinite(loss).all() or not torch.isfinite(pred_v).all() or not torch.isfinite(target_v).all():
                non_finite += 1
                grand_non_finite_count += 1

            loss.backward()

            # Gradient clipping (same as training script)
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), max_norm=1.0
            )

            # Check for non-finite grads
            for p in actor.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    non_finite += 1
                    grand_non_finite_count += 1
                    break

            optim.step()

            # Track cumulative update norm (L2 of total parameter change so far)
            step_norm = 0.0
            with torch.no_grad():
                for name, p in actor.named_parameters():
                    if p.requires_grad:
                        delta = (p.data.cpu() - init_params[name]).float()
                        step_norm += delta.pow(2).sum().item()
            step_update_norms.append(math.sqrt(step_norm))

        t_elapsed = time.time() - t_start
        actor_update_norm = sum(step_update_norms)

        print(f"  {updates_k} updates in {t_elapsed:.1f}s, "
              f"update_norm={actor_update_norm:.4f}, "
              f"non_finite={non_finite}")
        print(f"  Step norms: min={min(step_update_norms):.4f}, "
              f"max={max(step_update_norms):.4f}, "
              f"mean={np.mean(step_update_norms):.4f}")

        per_rank_update_norm[rank] = actor_update_norm

        # --- Record post-training per-layer metrics ---
        post_layers = []
        for adaptor, layer_dict in zip(actor._lora_adaptors, layers_info):
            A = adaptor.lora_A.data
            B = adaptor.lora_B.data
            A_fro, nf = _safe_fro(A)
            grand_non_finite_count += nf
            B_fro, nf = _safe_fro(B)
            grand_non_finite_count += nf

            eff_dW = scaling * (A @ B)
            dW_fro, nf = _safe_fro(eff_dW)
            grand_non_finite_count += nf

            mod_name = layer_dict["module_name"]
            target_module = _deep_getattr(base_copy.model, mod_name)
            W0 = target_module.weight.data
            W0_fro, nf = _safe_fro(W0)
            grand_non_finite_count += nf

            u_r = dW_fro / max(W0_fro, 1e-12) if W0_fro > 0 else 0.0

            post_layers.append({
                "module_path": f"model.{mod_name}",
                "rank": rank,
                "alpha": alpha,
                "scaling": scaling,
                "A_shape": list(A.shape),
                "B_shape": list(B.shape),
                "trainable_count": layer_dict["trainable_params"],
                "W0_fro": W0_fro,
                "A_fro": A_fro,
                "B_fro": B_fro,
                "dW_fro": dW_fro,
                "u_r": u_r,
            })

        # Compute u_tilde_r = median of u_r across identical module paths
        u_r_values = [layer["u_r"] for layer in post_layers]
        u_tilde_r = float(np.median(u_r_values)) if u_r_values else 0.0
        per_rank_u_tilde[rank] = u_tilde_r

        per_rank_results[rank] = {
            "rank": rank,
            "alpha": alpha,
            "scaling": scaling,
            "layers_init": pre_layers,
            "layers_post": post_layers,
            "u_tilde_r": u_tilde_r,
            "actor_update_norm": actor_update_norm,
            "non_finite_count": non_finite,
            "step_update_norms": step_update_norms,
        }

        # --- Clean up for next rank ---
        del actor, base_copy, noise_module, optim
        torch.cuda.empty_cache()
        print(f"  Cleanup complete. GPU cache emptied.")

    # ------------------------------------------------------------------
    # 4. Aggregate and evaluate gate
    # ------------------------------------------------------------------
    gate_result = evaluate_scaling_gate(
        per_rank_u_tilde=per_rank_u_tilde,
        per_rank_update_norm=per_rank_update_norm,
        non_finite_count=grand_non_finite_count,
        insertion_sets=grand_insertion_sets,
    )

    # ------------------------------------------------------------------
    # 5. Build the output JSON
    # ------------------------------------------------------------------
    output = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        GATE_PASS_KEY: gate_result[GATE_PASS_KEY],
        "device_used": device_used,
        "lr_used": ACTOR_LR,
        "updates_K": updates_k,
        "batch_indices": list(batch_indices),
        "per_rank": {},
        "aggregate": {
            "ratio": gate_result["ratio"],
            "min_u_tilde": gate_result["min_u_tilde"],
            "max_u_tilde": gate_result["max_u_tilde"],
            "update_norm_ratios_vs_rank4": gate_result["update_norm_ratios_vs_rank4"],
        },
        "gate_breakdown": {
            "ratio_leq_4": {
                "pass": gate_result["gate_conditions"]["ratio_leq_4"],
                "detail": f"max(u_tilde)/min(u_tilde) = {gate_result['ratio']:.6f} <= 4.0"
            },
            "high_rank_update_norm_sufficient": {
                "pass": gate_result["gate_conditions"]["high_rank_update_norm_sufficient"],
                "detail": f"all high-rank update norms >= {MIN_UPDATE_NORM_FRACTION*100:.0f}% of rank4"
            },
            "non_finite_ok": {
                "pass": gate_result["gate_conditions"]["non_finite_ok"],
                "detail": f"non_finite_count = {grand_non_finite_count}"
            },
            "insertion_sets_identical": {
                "pass": gate_result["gate_conditions"]["insertion_sets_identical"],
                "detail": f"all ranks have identical adapter module paths"
            },
        },
        "next_action": gate_result["next_action"],
        "stop_gpu_main_matrix": gate_result["stop_gpu_main_matrix"],
        "notes": [
            f"Probe ran {updates_k} Adam updates at lr={ACTOR_LR} on "
            f"fixed batch indices {list(batch_indices)} from B0 replay.",
            f"Optimizer: torch.optim.Adam (no weight_decay, matching train_a2_10k_capability.py:4864).",
            f"Per-rank seeding identical (torch.manual_seed(42)) for comparable A init.",
            f"Loss: REAL FM velocity-prediction (OT-CFM) matching Phase1aPolicyFM.compute_loss(). "
            f"target trajectory from deterministic Euler denoising of frozen base policy.",
            f"Gradient clipping at max_norm=1.0 applied.",
            f"non_finite count includes contributions from _safe_fro (tensor entries) + training loop.",
        ],
    }

    # Per-rank detail
    for rank in PROBE_RANKS:
        r = per_rank_results[rank]
        output["per_rank"][str(rank)] = {
            "rank": r["rank"],
            "alpha": r["alpha"],
            "scaling": r["scaling"],
            "layers": r["layers_post"],
            "u_tilde_r": r["u_tilde_r"],
            "actor_update_norm": r["actor_update_norm"],
            "non_finite_count": r["non_finite_count"],
        }

    # ------------------------------------------------------------------
    # 6. Write output
    # ------------------------------------------------------------------
    output_path = output_dir / "lora_scaling_audit.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[lora_scaling] Written to {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"LoRA Scaling Audit Summary")
    print(f"{'='*60}")
    print(f"  Gate: {'PASS' if gate_result[GATE_PASS_KEY] else 'FAIL'}")
    print(f"  Ratio (max/min u_tilde): {gate_result['ratio']:.4f}  "
          f"(threshold: <= {RATIO_MAX})")
    print(f"  u_tilde per rank: {per_rank_u_tilde}")
    print(f"  Update norm ratios vs rank4: {gate_result['update_norm_ratios_vs_rank4']}")
    if not gate_result[GATE_PASS_KEY]:
        print(f"  Next action: {gate_result['next_action']}")
    print(f"{'='*60}")

    return output
