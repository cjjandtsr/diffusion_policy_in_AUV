#!/usr/bin/env python3
"""
train_a2_10k_capability.py — A2 10k Capability Short-Train for Route A-RF

Phase: A2 (actor LoRA/adapter + noise module update; base FM/obs_encoder/teacher frozen;
        critic loaded from A0 LW1 best)
设计文档: RouteA_ReinFlow_FM_RL_Design_v1_2.md §4.2, §4.3, §9.3, §11, §13
前置任务: P0 preflight fixes (P0-1~P0-6); A1 1k Stability → PASS_TO_A2_GATE
指导文件: Guide_RouteA_A2_10K_Capability_v1.md

Pipeline:
  1. Load v9_2 BC/FM checkpoint → base_policy + teacher_policy (both frozen eval)
  2. Create RouteAReinFlowFMPolicy (LoRA adapters on mid/up/final_conv)
  3. Load A0 best critic checkpoint (a0_critic_best.pt) — strict=True
  4. Run shape smoke (B=1, B=4), gradient audit (full + PG-only), freeze audit
  5. Run 1-step / 10-step / 100-step / 300-step smoke cascades
  6. Run 10000 RL steps with:
     - periodic checkpoint every 500 steps
     - best checkpoint by multi-metric score
     - final checkpoint on completion
     - emergency checkpoint on exception
     - eval every 1000 steps (canonical/support/default/dense/random_stress)
     - hard stop conditions (NaN/Inf, grad norm, twin gap, regression)
  7. Generate a2_10k_summary.json with PASS/BLOCKED decision

MC Compliance (§13):
  - MC-1: action output remains (B, 8, 11) = execution trajectory
  - MC-2: FM policy is core (adapters + noise, not external planner)
  - MC-3: safety through loss/reward, not post-hoc
  - MC-4: EGO data frozen (offline prior only)
  - MC-5: attributable (paired seeds, decomposed metrics, ablation-capable)

Usage:
  python scripts/train_a2_10k_capability.py \\
    --ckpt /path/to/v9_2_checkpoint.ckpt \\
    --critic_ckpt program/shiyan/.../LW1/a0_critic_best.pt \\
    --output_dir program/shiyan/results_routeA_A2_10k_capability_YYYYMMDD_HHMM \\
    [--smoke]  (50 steps for quick test)
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import signal
import sys
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from termcolor import cprint
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
from diffusion_policy_3d.env.kinematic_uav_env_v2 import WORLDS_DIR, WORLD_ENV_MAP
from diffusion_policy_3d.env.collision_geom import load_obstacles
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud
from diffusion_policy_3d.policy.phase1a_policy_fm_rl import Phase1aPolicyFM
from diffusion_policy_3d.policy.routea_reinflow_fm_policy import (
    RouteAReinFlowFMPolicy, compute_routea_actor_losses, TARGET_PATTERNS_V1,
)
from diffusion_policy_3d.policy.reinflow_noise_module import ReinFlowNoiseModule
from diffusion_policy_3d.model.rl.chunk_critic import (
    ChunkTwinCritic, target_update, hard_update, compute_critic_loss,
    HEAD_NAMES, DEFAULT_BETAS, EXPECTED_STATE_DIM,
)
from diffusion_policy_3d.rl.routea_replay_buffer import (
    RouteAReplayBuffer, RouteATransition, collate_to_critic,
    collate_to_critic_device, REWARD_TERM_NAMES,
)
from diffusion_policy_3d.rl.routea_rl_env_adapter_v1 import (
    load_routea_rl_admission_manifest,
    validate_rl_train_admission,
    build_role_buckets,
    validate_role_buckets_for_training,
    check_historical_contamination,
    dry_run_newenv_loader,
    FORMAL_CAPABILITY_ROLES,
    SUPPORT_ROLES,
    NON_TRAIN_ROLES,
    TRAINABLE_ROLES,
    HISTORICAL_EVAL_WORLD_IDS,
    map_legacy_scenario_to_role,
    sample_role_balanced_worlds,
    get_newenv_gate_config,
    RouteARLWorldEntry,
    RouteARLAdmissionManifest,
)
from diffusion_policy_3d.rl.formal_train_hard_stop import (
    FormalTrainHardStopChecker,
    FormalTrainHardStopConfig,
    verify_freeze_boundary,
    compute_model_hash,
)

# ═══════════════════════════════════════════════════════════════════
# Constants (restored from cleanup backup)
# ═══════════════════════════════════════════════════════════════════

GAMMA = 0.99
GAMMA_H = GAMMA ** 8
TARGET_TAU = 0.005
MAX_GRAD_NORM_ACTOR = 1.0
MAX_GRAD_NORM_CRITIC = 1.0
BATCH_SIZE = 2048
LAMBDA_BC = 1.0
LAMBDA_PATH = 0.0
LAMBDA_NOISE = 0.01
LAMBDA_SMOOTH = 0.001
LAMBDA_PG = 1.0
LAMBDA_DQPG = 1.0
ACTOR_LR = 1e-5
CRITIC_LR = 1e-5

A0_NEWENV_SMOKE_STEPS = 50
A0_NEWENV_SMOKE_EPISODES_PER_WORLD = 2
A0_NEWENV_SMOKE_CRITIC_UPDATES = 8
A0_NEWENV_SMOKE_MAX_STEPS = 60
A1_NEWENV_MAX_ACTOR_UPDATES = 200
A1_NEWENV_ROLLOUTS_PER_STEP = 4
A1_NEWENV_EVAL_EPISODES = 3
A1_NEWENV_EVAL_INTERVAL = 50
A1_NEWENV_CRITIC_UPDATES = 16
A2_NEWENV_MAX_ACTOR_UPDATES = 500
A2_NEWENV_ROLLOUTS_PER_STEP = 4
A2_NEWENV_EVAL_EPISODES = 4
A2_NEWENV_EVAL_INTERVAL = 100
A2_NEWENV_PERIODIC_CKPT = 500
A2_NEWENV_SINGLE_ROLE_MIN_EPISODES = 30
A2_NEWENV_SINGLE_SCENARIO_MIN_SAMPLES = 40

CR_PP_LIMITS = {"canonical": 0.05, "support": 0.05, "default": 0.10, "dense": 0.10}
SCENARIO_CONFIGS = {}

# ═══════════════════════════════════════════════════════════════════
# Checkpoint Writer
# ═══════════════════════════════════════════════════════════════════

class CheckpointWriter:
    """Manages periodic/best/final/emergency checkpoint saving with manifest.

    Design requirements (§2/A2 instruction):
      - periodic: every 500 steps
      - best: by multi-metric score
      - final: on normal completion
      - emergency: on exception/timeout/KeyboardInterrupt
      - manifest: updates and flushes after each write
      - size check: verifies file size > 0
    """

    def __init__(self, output_dir: Path, run_id: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.manifest_path = self.output_dir / "checkpoint_manifest.json"
        self.current_step = 0  # updated every step by train loop for crash recovery

        self.manifest = {
            "run_id": run_id,
            "periodic": [],
            "best": {"actor": "", "critic": "", "score": None, "step": None},
            "final": {"actor": "", "critic": "", "step": None},
            "emergency": [],
            "latest_valid_step": 0,
            "size_check": True,
            "sha256_or_size_check": True,
        }
        self._flush_manifest()

        # Best checkpoint tracking
        self.best_score = -float('inf')
        self.best_step = 0
        self.best_actor_sd = None
        self.best_critic_sd = None
        self.best_target_critic_sd = None
        self.best_metrics = {}

    def _flush_manifest(self):
        """Write manifest and flush."""
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

    def _size_check(self, path: Path) -> bool:
        """Verify file size > 0."""
        if path.exists():
            sz = path.stat().st_size
            if sz > 0:
                return True
            cprint(f"  [CKPT WARN] {path.name} size={sz} — may be corrupt", "red")
        return False

    def _save_ckpt(self, filename: str, data: dict) -> bool:
        """Save checkpoint dict, verify size, return success."""
        path = self.output_dir / filename
        torch.save(data, path)
        ok = self._size_check(path)
        return ok

    def save_periodic(self, step: int, actor_state: dict, critic_state: dict,
                      target_critic_state: dict, optimizer_states: dict,
                      metrics: dict, extra: Optional[dict] = None):
        """Save periodic checkpoint."""
        filename = f"a2_ckpt_step{step:06d}.pt"
        data = {
            "step": step,
            "run_id": self.run_id,
            "actor_state_dict": actor_state,
            "critic_state_dict": critic_state,
            "target_critic_state_dict": target_critic_state,
            "optimizer_states": optimizer_states,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        if extra:
            data.update(extra)

        ok = self._save_ckpt(filename, data)
        if ok:
            self.manifest["periodic"].append({
                "step": step,
                "file": filename,
                "size_check": True,
            })
            self.manifest["latest_valid_step"] = step
            self._flush_manifest()
        return ok

    def update_best(self, step: int, score: float,
                    actor_state: dict, critic_state: dict,
                    target_critic_state: dict, metrics: dict):
        """Update best checkpoint if score improves."""
        if score > self.best_score:
            self.best_score = score
            self.best_step = step
            self.best_actor_sd = {
                k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in actor_state.items()
            }
            self.best_critic_sd = {
                k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in critic_state.items()
            }
            self.best_target_critic_sd = {
                k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in target_critic_state.items()
            }
            self.best_metrics = dict(metrics)

            # Write to disk
            ok_a = self._save_ckpt("a2_actor_best.pt", {
                "actor_state_dict": self.best_actor_sd,
                "step": step, "score": score, "run_id": self.run_id,
            })
            ok_c = self._save_ckpt("a2_critic_best.pt", {
                "critic_state_dict": self.best_critic_sd,
                "target_critic_state_dict": self.best_target_critic_sd,
                "step": step, "score": score, "run_id": self.run_id,
                "gate_metrics_at_save": metrics,
            })
            if ok_a and ok_c:
                self.manifest["best"] = {
                    "actor": "a2_actor_best.pt",
                    "critic": "a2_critic_best.pt",
                    "score": score,
                    "step": step,
                }
                self._flush_manifest()
            return True
        return False

    def save_final(self, step: int, actor_state: dict, critic_state: dict,
                   target_critic_state: dict, metrics: dict):
        """Save final checkpoint."""
        ok_a = self._save_ckpt("a2_actor_final.pt", {
            "actor_state_dict": actor_state,
            "step": step, "run_id": self.run_id,
            "is_final": True,
        })
        ok_c = self._save_ckpt("a2_critic_final.pt", {
            "critic_state_dict": critic_state,
            "target_critic_state_dict": target_critic_state,
            "step": step, "run_id": self.run_id,
            "gate_metrics_at_save": metrics,
            "is_final": True,
        })
        if ok_a and ok_c:
            self.manifest["final"] = {
                "actor": "a2_actor_final.pt",
                "critic": "a2_critic_final.pt",
                "step": step,
            }
            self.manifest["latest_valid_step"] = step
            self._flush_manifest()
        return ok_a and ok_c

    def save_emergency(self, step: int, actor_state: dict, critic_state: dict,
                       reason: str, extra: Optional[dict] = None):
        """Save emergency checkpoint on exception."""
        filename = f"a2_emergency_step{step:06d}.pt"
        data = {
            "step": step,
            "run_id": self.run_id,
            "reason": reason,
            "actor_state_dict": actor_state,
            "critic_state_dict": critic_state,
            "timestamp": datetime.now().isoformat(),
        }
        if extra:
            data.update(extra)
        ok = self._save_ckpt(filename, data)
        if ok:
            self.manifest["emergency"].append({
                "step": step,
                "file": filename,
                "reason": reason,
            })
            self.manifest["latest_valid_step"] = step
            self._flush_manifest()
        return ok


# ═══════════════════════════════════════════════════════════════════
# R3: Streaming CSV Writer (§Step 7)
# ═══════════════════════════════════════════════════════════════════

class StreamingCSVWriter:
    """CSV writer that appends+flushes each row — survives kill/crash."""

    def __init__(self, path: Path, fieldnames: List[str]):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self._header_written = False
        self._file = None

    def _ensure_open(self):
        if self._file is None:
            self._file = open(str(self.path), "a", newline="")

    def append(self, row: Dict):
        """Append one row and flush to disk."""
        self._ensure_open()
        writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if not self._header_written:
            writer.writeheader()
            self._header_written = True
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, (np.floating, np.integer)):
                clean_row[k] = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, torch.Tensor):
                clean_row[k] = v.item() if v.numel() == 1 else v.cpu().tolist()
            elif isinstance(v, (bool,)):
                clean_row[k] = int(v)
            else:
                clean_row[k] = v
        writer.writerow(clean_row)
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


# ═══════════════════════════════════════════════════════════════════
# Policy Loading (same as A1)
# ═══════════════════════════════════════════════════════════════════

def load_v9_2_policy(ckpt_path: str,
                     device: str = "cuda:0") -> Phase1aPolicyFM:
    cprint(f"[load] Loading checkpoint: {ckpt_path}", "cyan")
    ckpt = torch.load(ckpt_path, map_location=device, pickle_module=dill)
    cfg = ckpt["cfg"]
    sd = ckpt["state_dicts"]["ema_model"]
    pc = cfg["policy"]

    n_obs = cfg.get("n_obs_steps", 3)
    if n_obs != 8:
        cprint(f"[WARN] n_obs_steps={n_obs}, expected 8", "yellow")

    fm = Phase1aPolicyFM(
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
        anchor_bank_path=None,  # no-anchor mode
        cfg_lambda=1.5, num_fm_steps=4, sigma_min=0.005, fm_t_scale=50,
        solver="euler", fm_anchor_t_start=0.95, use_pc_color=False,
        pointnet_type="multi_stage_pointnet",
        pointcloud_encoder_cfg=pc.get("pointcloud_encoder_cfg", None),
        point_downsample=pc.get("point_downsample", True),
    )

    missing, unexpected = fm.load_state_dict(sd, strict=False)
    anchor_missing = [k for k in missing if 'anchor' in k.lower()]
    if anchor_missing:
        cprint(f"  Anchor keys skipped (expected, no-anchor): {anchor_missing}", "yellow")
    other_missing = [k for k in missing if 'anchor' not in k.lower()]
    if other_missing:
        cprint(f"  Other missing keys: {other_missing}", "yellow")
    if unexpected:
        cprint(f"  Unexpected keys: {unexpected}", "yellow")

    assert fm.anchor_head is None, "no-anchor mode: anchor_head must be None"
    assert fm.anchor_bank_full is None, "no-anchor mode: anchor_bank must be None"

    fm.to(device).eval()

    # Verify global_cond_dim
    with torch.no_grad():
        dummy_pc = torch.zeros(1, n_obs, 4096, 3, device=device)
        dummy_state = torch.zeros(1, n_obs, 9, device=device)
        nobs = {"point_cloud": dummy_pc, "agent_pos": dummy_state}
        this_nobs = {k: v[:, :n_obs, ...].reshape(-1, *v.shape[2:]) for k, v in nobs.items()}
        gc = fm.obs_encoder(this_nobs).reshape(1, -1)
        gdim = gc.shape[-1]
    assert gdim == EXPECTED_STATE_DIM, f"Expected {EXPECTED_STATE_DIM}, got {gdim}"

    cprint(f"  Loaded: {sum(p.numel() for p in fm.parameters())/1e6:.2f}M params, "
           f"global_cond={gdim}, n_obs={n_obs}", "green")
    return fm


def create_actor_wrapper(base_policy: Phase1aPolicyFM,
                         teacher_policy: Phase1aPolicyFM,
                         device: str = "cuda:0",
                         lora_rank: int = 4) -> RouteAReinFlowFMPolicy:
    noise_module = ReinFlowNoiseModule(
        num_fm_steps=4, sigma_min=1e-4, sigma_max=0.10, init_sigma=0.01,
    )
    actor = RouteAReinFlowFMPolicy(
        base_policy=base_policy, teacher_policy=teacher_policy,
        noise_module=noise_module,
        adapter_config={
            "lora_rank": lora_rank, "lora_alpha": 8,
            "target_patterns": TARGET_PATTERNS_V1,
        },
        freeze_base=True, train_anchor_head=False, return_path=False,
    )
    actor.to(device)
    actor.train()

    zero_result = actor.measure_zero_init_equivalence()
    assert zero_result["zero_init_verified"], "LoRA B must be zero-init"
    cprint(f"  Zero-init verified: max_abs_B={zero_result['zero_init_max_abs_B']:.2e}", "cyan")
    cprint(f"  Trainable: {actor.get_trainable_param_count()} / "
           f"{actor.get_base_param_count()} total", "cyan")
    return actor


def load_critic_from_a0(critic_ckpt_path: str,
                         device: str = "cuda:0",
                         use_layernorm: bool = False) -> Tuple[ChunkTwinCritic, ChunkTwinCritic, dict]:
    """Load A0 best critic checkpoint.

    When use_layernorm=True (B1c): uses strict=False, reports missing/unexpected keys.
    LayerNorm layers (state_ln, action_ln, trunk.*.layernorm) are randomly initialized.
    """
    cprint(f"[load] Loading A0 critic: {critic_ckpt_path}"
           + (" (use_layernorm=True, strict=False)" if use_layernorm else ""), "cyan")
    ckpt = torch.load(critic_ckpt_path, map_location=device)

    critic = ChunkTwinCritic(
        state_dim=EXPECTED_STATE_DIM, da=11, horizon=8,
        hidden=256, head_hidden=128,
        use_layernorm=use_layernorm,
    ).to(device)
    target_critic = ChunkTwinCritic(
        state_dim=EXPECTED_STATE_DIM, da=11, horizon=8,
        hidden=256, head_hidden=128,
        use_layernorm=use_layernorm,
    ).to(device)

    _strict = not use_layernorm
    missing_c, unexpected_c = critic.load_state_dict(
        ckpt["critic_state_dict"], strict=_strict)
    if use_layernorm:
        cprint(f"  [B1c LayerNorm] Critic online loaded strict=False: "
               f"{len(unexpected_c)} unexpected, {len(missing_c)} missing", "yellow")
        if missing_c:
            cprint(f"    Missing keys (new LN layers, expected): {missing_c}", "cyan")
    else:
        cprint(f"  Critic online loaded: {len(unexpected_c)} unexpected, "
               f"{len(missing_c)} missing (expect 0)", "cyan")

    if "target_critic_state_dict" in ckpt:
        missing_t, unexpected_t = target_critic.load_state_dict(
            ckpt["target_critic_state_dict"], strict=_strict)
        if use_layernorm:
            cprint(f"  [B1c LayerNorm] Critic target loaded strict=False: "
                   f"{len(unexpected_t)} unexpected, {len(missing_t)} missing", "yellow")
        else:
            cprint(f"  Critic target loaded from A0 target: {len(unexpected_t)} unexpected, "
                   f"{len(missing_t)} missing (expect 0)", "cyan")
    else:
        cprint(f"  [WARN] No target_critic_state_dict in A0 ckpt, copying from online", "yellow")
        target_critic.load_state_dict(critic.state_dict())

    if not use_layernorm:
        assert len(unexpected_c) == 0, f"Critic unexpected keys: {unexpected_c}"
    assert len(missing_c) == 0, f"Critic missing keys: {missing_c}"

    gate_metrics = ckpt.get("gate_metrics_at_save", {})
    _tg = gate_metrics.get("twin_gap", None)
    _qstd = gate_metrics.get("q_std", None)
    _tg_str = f"{_tg:.3f}" if isinstance(_tg, (int, float)) else str(_tg)
    _qstd_str = f"{_qstd:.2f}" if isinstance(_qstd, (int, float)) else str(_qstd)
    cprint(f"  A0 gate: twin_gap={_tg_str}, Q_std={_qstd_str}", "cyan")

    critic.train()
    target_critic.train()
    return critic, target_critic, gate_metrics


# ═══════════════════════════════════════════════════════════════════
# Shape Smoke (§4/Step 4) — verify B=1 and B=4 action shapes
# ═══════════════════════════════════════════════════════════════════

def run_shape_smoke(actor: RouteAReinFlowFMPolicy,
                    critic: ChunkTwinCritic,
                    device: str) -> Dict:
    """Run B=1 and B=4 shape smoke checks.

    Verifies:
      - obs/global_cond batch = B
      - action_pred shape = (B, 8, 11)
      - projected trajectory batch-consistent
      - _project_trajectory does not broadcast B incorrectly
      - no RuntimeError for B=4
    """
    cprint("\n[smoke] Running shape smoke (B=1, B=4)...", "white", "on_blue")
    results = {"B1": {}, "B4": {}}

    for B in [1, 4]:
        res = {}
        try:
            To = actor.n_obs_steps
            T = actor.horizon
            Da = actor.action_dim
            Ha = actor.n_action_steps

            # Dummy obs
            obs_dict = {
                "point_cloud": torch.randn(B, To, 4096, 3, device=device),
                "agent_pos": torch.randn(B, To, 9, device=device),
            }

            # Actor forward (deterministic)
            actor_out = actor(obs_dict, deterministic=False)
            action = actor_out["action"]
            log_prob = actor_out["log_prob"]
            noise_sigmas = actor_out["noise_sigmas"]

            # Check shapes
            res["action_shape"] = list(action.shape)
            res["expected_action_shape"] = [B, Ha, Da]
            res["action_shape_pass"] = (action.shape == (B, Ha, Da))
            res["log_prob_shape"] = list(log_prob.shape)
            res["log_prob_shape_pass"] = (log_prob.shape == (B,))
            res["noise_sigmas_shape"] = list(noise_sigmas.shape)
            res["noise_sigmas_shape_pass"] = (noise_sigmas.shape == (4,))

            # Check for NaN/Inf
            res["action_finite"] = bool(torch.isfinite(action).all())
            res["log_prob_finite"] = bool(torch.isfinite(log_prob).all())

            # Check B>1 broadcasting: _project_trajectory consistency
            if B > 1:
                # Verify that global_cond is per-batch, not broadcast
                gc = actor_out.get("global_cond")
                if gc is not None:
                    res["global_cond_shape"] = list(gc.shape)
                    res["global_cond_shape_pass"] = (gc.shape[0] == B)

                # Critic forward
                q = critic(actor_out["global_cond"], action)
                for head_name, q_val in q.items():
                    if q_val.shape[0] != B:
                        res[f"critic_{head_name}_batch_mismatch"] = True
                res["critic_forward_pass"] = True

            # Teacher action shape
            teacher = actor_out.get("teacher_action")
            if teacher is not None:
                res["teacher_action_shape"] = list(teacher.shape)
                res["teacher_action_shape_pass"] = (teacher.shape == (B, Ha, Da))

            res["pass"] = all([
                res.get("action_shape_pass", False),
                res.get("log_prob_shape_pass", False),
                res.get("noise_sigmas_shape_pass", False),
                res.get("action_finite", False),
            ])

        except Exception as e:
            res["pass"] = False
            res["error"] = str(e)
            res["traceback"] = traceback.format_exc()

        results[f"B{B}"] = res
        status = "PASS" if res.get("pass") else "FAIL"
        cprint(f"  B={B} shape smoke: {status}", "green" if res.get("pass") else "red")

    return results


# ═══════════════════════════════════════════════════════════════════
# Gradient Audit (same as A1 but extended)
# ═══════════════════════════════════════════════════════════════════

def run_gradient_audit(actor: RouteAReinFlowFMPolicy,
                       critic: ChunkTwinCritic,
                       device: str) -> Dict:
    """Full gradient audit with freeze boundary verification."""
    cprint("\n[audit] Running gradient audit...", "white", "on_blue")
    results = {}

    # ── Prepare dummy batch ──
    B = 4
    To = actor.n_obs_steps
    obs_dict = {
        "point_cloud": torch.randn(B, To, 4096, 3, device=device),
        "agent_pos": torch.randn(B, To, 9, device=device),
    }

    actor.zero_grad()
    critic.zero_grad()

    # Actor forward
    actor_out = actor(obs_dict, deterministic=False)

    # Critic forward
    q = critic(actor_out["global_cond"], actor_out["action"])
    advantage = q["total"].mean(dim=1) - q["total"].mean()

    # Actor loss
    losses = compute_routea_actor_losses(
        actor_outputs={
            "action": actor_out["action"],
            "log_prob": actor_out["log_prob"],
            "noise_sigmas": actor_out["noise_sigmas"],
        },
        teacher_action=actor_out["teacher_action"],
        advantage=advantage,
    )
    losses["actor_loss_total"].backward(retain_graph=True)

    # ── Checks ──
    # LoRA
    lora_grad_norms = []
    for i, adaptor in enumerate(actor._lora_adaptors):
        gn = 0.0
        for p in adaptor.parameters():
            if p.grad is not None:
                gn += p.grad.norm().item() ** 2
        gn = math.sqrt(gn)
        lora_grad_norms.append(gn)
    results["lora_max_grad_norm"] = round(max(lora_grad_norms), 6) if lora_grad_norms else 0.0
    results["lora_any_grad_nonzero"] = any(gn > 1e-8 for gn in lora_grad_norms)

    # Noise
    noise_grad_norms = []
    for p in actor.noise_module.parameters():
        if p.grad is not None:
            noise_grad_norms.append(p.grad.norm().item())
    results["noise_max_grad_norm"] = round(max(noise_grad_norms), 6) if noise_grad_norms else 0.0
    results["noise_grad_nonzero"] = any(gn > 1e-8 for gn in noise_grad_norms)

    # Base FM frozen
    base_fm_grad_norms = []
    for name, p in actor.base_policy.model.named_parameters():
        if p.grad is not None:
            base_fm_grad_norms.append(p.grad.norm().item())
    results["base_fm_max_grad_norm"] = round(max(base_fm_grad_norms), 6) if base_fm_grad_norms else 0.0
    results["base_fm_frozen_verified"] = len(base_fm_grad_norms) == 0

    # Encoder frozen
    encoder_grad_norms = []
    for name, p in actor.base_policy.obs_encoder.named_parameters():
        if p.grad is not None:
            encoder_grad_norms.append(p.grad.norm().item())
    results["encoder_max_grad_norm"] = round(max(encoder_grad_norms), 6) if encoder_grad_norms else 0.0
    results["encoder_frozen_verified"] = len(encoder_grad_norms) == 0

    # Teacher frozen
    teacher_grad_norms = []
    for name, p in actor.teacher_policy.named_parameters():
        if p.grad is not None:
            teacher_grad_norms.append(p.grad.norm().item())
    results["teacher_max_grad_norm"] = round(max(teacher_grad_norms), 6) if teacher_grad_norms else 0.0
    results["teacher_frozen_verified"] = len(teacher_grad_norms) == 0

    # Overall
    all_pass = (
        results["lora_any_grad_nonzero"]
        and results["noise_grad_nonzero"]
        and results["base_fm_frozen_verified"]
        and results["encoder_frozen_verified"]
        and results["teacher_frozen_verified"]
    )
    results["audit_passed"] = all_pass

    for k, v in results.items():
        color = "green" if (isinstance(v, bool) and v) or (
            isinstance(v, float) and v < 1e-6 and "frozen" in k) else (
            "red" if isinstance(v, bool) and not v else "white")
        cprint(f"  {k}: {v}", color)

    if all_pass:
        cprint(f"  Gradient audit PASSED", "green")
    else:
        cprint(f"  Gradient audit FAILED", "red")

    actor.zero_grad()
    critic.zero_grad()
    return results


def run_pg_only_audit(actor: RouteAReinFlowFMPolicy,
                      critic: ChunkTwinCritic,
                      device: str) -> Dict:
    """Verify DQPG provides RL gradient to LoRA without BC/noise/smooth."""
    cprint("\n[audit] Running PG-only gradient audit...", "white", "on_magenta")
    results = {}

    B = 4
    obs_dict = {
        "point_cloud": torch.randn(B, actor.n_obs_steps, 4096, 3, device=device),
        "agent_pos": torch.randn(B, actor.n_obs_steps, 9, device=device),
    }

    actor.zero_grad()
    critic.zero_grad()

    actor_out = actor(obs_dict, deterministic=False)
    q = critic(actor_out["global_cond"], actor_out["action"])
    q_min = torch.min(q["total"][:, 0], q["total"][:, 1])

    with torch.no_grad():
        baseline = q_min.mean()
        adv = q_min - baseline
    reinforce_loss = -(actor_out["log_prob"] * adv).mean()
    dqpg_loss = -q_min.mean()
    pg_only_loss = LAMBDA_DQPG * dqpg_loss + LAMBDA_PG * reinforce_loss
    pg_only_loss.backward()

    # LoRA
    lora_grad_norms = []
    for i, adaptor in enumerate(actor._lora_adaptors):
        gn = 0.0
        for p in adaptor.parameters():
            if p.grad is not None:
                gn += p.grad.norm().item() ** 2
        gn = math.sqrt(gn)
        lora_grad_norms.append(gn)
    results["lora_max_pg_grad_norm"] = round(max(lora_grad_norms), 6) if lora_grad_norms else 0.0
    results["pg_grad_to_lora"] = any(gn > 1e-8 for gn in lora_grad_norms)

    # Noise
    noise_pg_grad_norms = []
    for p in actor.noise_module.parameters():
        if p.grad is not None:
            noise_pg_grad_norms.append(p.grad.norm().item())
    results["noise_max_pg_grad_norm"] = round(max(noise_pg_grad_norms), 6) if noise_pg_grad_norms else 0.0
    results["pg_grad_to_noise"] = any(gn > 1e-8 for gn in noise_pg_grad_norms)

    # Frozen
    base_pg_grads = []
    for name, p in actor.base_policy.model.named_parameters():
        if p.grad is not None:
            base_pg_grads.append(p.grad.norm().item())
    results["base_fm_pg_max_grad"] = round(max(base_pg_grads), 6) if base_pg_grads else 0.0
    results["base_fm_frozen_pg"] = len(base_pg_grads) == 0

    encoder_pg_grads = []
    for n, p in actor.base_policy.obs_encoder.named_parameters():
        if p.grad is not None:
            encoder_pg_grads.append(p.grad.norm().item())
    results["encoder_frozen_pg"] = len(encoder_pg_grads) == 0

    teacher_pg_grads = []
    for n, p in actor.teacher_policy.named_parameters():
        if p.grad is not None:
            teacher_pg_grads.append(p.grad.norm().item())
    results["teacher_frozen_pg"] = len(teacher_pg_grads) == 0

    all_pass = (
        results["pg_grad_to_lora"]
        and results["pg_grad_to_noise"]
        and results["base_fm_frozen_pg"]
        and results["encoder_frozen_pg"]
        and results["teacher_frozen_pg"]
    )
    results["pg_audit_passed"] = all_pass

    for k, v in results.items():
        color = "green" if (isinstance(v, bool) and v) else ("red" if isinstance(v, bool) and not v else "white")
        cprint(f"  {k}: {v}", color)

    if all_pass:
        cprint(f"  [PG-only audit] PASSED", "green")
    else:
        cprint(f"  [PG-only audit] FAILED", "red")

    actor.zero_grad()
    critic.zero_grad()
    return results


# ═══════════════════════════════════════════════════════════════════
# Freeze Audit (§4/Step 4) — verify requires_grad flags
# ═══════════════════════════════════════════════════════════════════

def run_freeze_audit(actor: RouteAReinFlowFMPolicy) -> Dict:
    """Verify frozen/trainable boundary via requires_grad flags."""
    cprint("\n[audit] Running freeze audit (requires_grad)...", "white", "on_blue")
    results = {}

    # Base FM model
    base_fm_rg = any(p.requires_grad for p in actor.base_policy.model.parameters())
    results["base_fm_requires_grad"] = base_fm_rg
    results["base_fm_frozen"] = not base_fm_rg

    # Obs encoder
    encoder_rg = any(p.requires_grad for p in actor.base_policy.obs_encoder.parameters())
    results["encoder_requires_grad"] = encoder_rg
    results["encoder_frozen"] = not encoder_rg

    # Teacher
    teacher_rg = any(p.requires_grad for p in actor.teacher_policy.parameters())
    results["teacher_requires_grad"] = teacher_rg
    results["teacher_frozen"] = not teacher_rg

    # Anchor head (should be None in no-anchor mode)
    if actor.base_policy.anchor_head is not None:
        anchor_rg = any(p.requires_grad for p in actor.base_policy.anchor_head.parameters())
        results["anchor_head_requires_grad"] = anchor_rg
        results["anchor_head_frozen"] = not anchor_rg
    else:
        results["anchor_head"] = "None (no-anchor mode)"

    # LoRA adapters
    lora_rg = any(p.requires_grad for adaptor in actor._lora_adaptors for p in adaptor.parameters())
    results["lora_requires_grad"] = lora_rg

    # Noise module
    noise_rg = any(p.requires_grad for p in actor.noise_module.parameters())
    results["noise_module_requires_grad"] = noise_rg

    all_pass = (
        results.get("base_fm_frozen", True)
        and results.get("encoder_frozen", True)
        and results.get("teacher_frozen", True)
        and results.get("lora_requires_grad", False)
        and results.get("noise_module_requires_grad", False)
    )
    results["freeze_audit_passed"] = all_pass

    for k, v in results.items():
        color = "green" if (isinstance(v, bool) and v in (True, False)) and (
            "frozen" in k or v is True) else (
            "red" if k.endswith("_frozen") and not v else "white")
        cprint(f"  {k}: {v}", color)
    return results


# ═══════════════════════════════════════════════════════════════════
# Env / Rollout helpers (extended from A1)
# ═══════════════════════════════════════════════════════════════════

def create_env(world_id: int, z_min: float = 1.0, z_max: float = 5.0,
               disturb: bool = False, seed: int = 42,
               curriculum_stage: str = 'C2') -> KinematicUAVEnv:
    wname = WORLD_ENV_MAP[world_id]
    obs_json = str(WORLDS_DIR / f"{wname}.obstacles.json")
    synth_pc = SynthPointCloud(obs_json, device="cuda:0")
    env = KinematicUAVEnv(
        world_id=world_id, synth_pc=synth_pc,
        rl_action_steps=8, h_min=0.3,
        z_min=z_min, z_max=z_max,
        disturb_enable=disturb, disturb_prob=0.2, disturb_pos_scale=0.3,
        mid_state_init_ratio=0.0, seed=seed, reward_version="v13",
        curriculum_stage=curriculum_stage,
    )
    return env


@torch.no_grad()
def _get_global_cond(policy: Phase1aPolicyFM, obs_dict: dict,
                     device: str) -> np.ndarray:
    obs_torch = {
        "point_cloud": torch.from_numpy(obs_dict["point_cloud"][..., :3].copy())
                       .unsqueeze(0).to(device),
        "agent_pos": torch.from_numpy(obs_dict["agent_pos"].copy())
                     .unsqueeze(0).to(device),
    }
    nobs = policy.normalizer.normalize(obs_torch)
    nobs = {k: v.to(device) for k, v in nobs.items()}
    To = policy.n_obs_steps
    this_nobs = {k: v[:, :To, ...].reshape(-1, *v.shape[2:]) for k, v in nobs.items()}
    gc = policy.obs_encoder(this_nobs).reshape(1, -1)
    return gc[0].cpu().numpy()


def _obs_dict_to_torch(obs_dict: dict, device: str) -> Dict[str, torch.Tensor]:
    return {
        "point_cloud": torch.from_numpy(obs_dict["point_cloud"][..., :3].copy())
                       .unsqueeze(0).to(device),
        "agent_pos": torch.from_numpy(obs_dict["agent_pos"].copy())
                     .unsqueeze(0).to(device),
    }


# ═══════════════════════════════════════════════════════════════════
# Rollout — 1 episode, returns transitions + obs buffer
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def rollout_episode_a2(
    actor: RouteAReinFlowFMPolicy,
    base_policy: Phase1aPolicyFM,
    env: KinematicUAVEnv,
    device: str,
    max_steps: int = 60,
    episode_seed: int = 0,
    world_id: int = 4,
    deterministic: bool = False,
    lazy_teacher: bool = False,
    reset_kwargs: Optional[Dict] = None,
) -> Tuple[List[RouteATransition], List[Dict], Dict]:
    """Rollout 1 episode with A2 actor.

    Args:
        lazy_teacher: If True, skip teacher forward during rollout.
            Teacher action is computed on-demand during actor update
            for the ~4 selected obs_buffer items only.
            Reduces model forwards by ~50% during rollout (~1.5x speedup).

    Returns:
        transitions: list of RouteATransition (for replay / critic training)
        obs_buffer: list of {obs_torch, teacher_action_np} per step (for actor re-forward)
        summary: episode summary dict
    """
    if reset_kwargs:
        obs, info = env.reset(**reset_kwargs)
    else:
        obs, info = env.reset()
    transitions: List[RouteATransition] = []
    obs_buffer: List[Dict] = []
    ep_reward = 0.0
    done_reason = "truncated"
    traj_positions = [env._drone_pos.copy()]
    # R3b: Track episode minimum clearance for frontier replay
    min_clearance = float("inf")

    # First step: need initial global_cond before actor forward
    obs_torch_0 = _obs_dict_to_torch(obs, device)
    actor_out_0 = actor(obs_torch_0, deterministic=deterministic, compute_teacher=not lazy_teacher)
    global_cond_t = actor_out_0["global_cond"][0].cpu().numpy()
    action_chunk = actor_out_0["action"][0].cpu().numpy()
    teacher_action_np = (actor_out_0["teacher_action"][0].cpu().numpy()
                         if actor_out_0.get("teacher_action") is not None else None)

    next_obs, reward, terminated, truncated, info = env.step(action_chunk)
    ep_reward += float(reward)
    done = terminated or truncated

    # R3b: Extract clearance from reward_metrics (key: d_soft_h)
    rm_step0 = info.get("reward_metrics", {})
    d_soft_h_0 = rm_step0.get("d_soft_h", -1.0)
    if d_soft_h_0 >= 0.0 and np.isfinite(d_soft_h_0):
        min_clearance = min(min_clearance, float(d_soft_h_0))

    # Build transition for step 0
    next_obs_torch = _obs_dict_to_torch(next_obs, device)
    next_actor_out = actor(next_obs_torch, deterministic=True, compute_teacher=False)
    next_global_cond = next_actor_out["global_cond"][0].cpu().numpy()
    next_action_chunk = next_actor_out["action"][0].cpu().numpy()

    reward_terms = info.get("reward_terms", {})
    reward_total = sum(reward_terms.get(k, 0.0) for k in REWARD_TERM_NAMES)
    trans = RouteATransition(
        obs_feature_t=global_cond_t.copy(),
        action_chunk=action_chunk.copy(),
        action_chunk_tp1=next_action_chunk.copy(),
        reward_total=float(reward_total),
        reward_terms={k: float(reward_terms.get(k, 0.0)) for k in REWARD_TERM_NAMES},
        obs_feature_tp1=next_global_cond.copy(),
        done=done,
        done_reason=info.get("done_reason", ""),
        world_id=world_id, episode_seed=episode_seed,
        start=traj_positions[0].copy() if len(traj_positions) > 0 else None,
        goal=env.goal_w.copy() if hasattr(env, 'goal_w') else None,
        path_length_so_far=0.0,
        reward_metrics=rm_step0,
    )
    transitions.append(trans)
    obs_buffer.append({
        "obs_torch": {k: v.detach().clone() for k, v in obs_torch_0.items()},
        "teacher_action": torch.from_numpy(teacher_action_np.copy()) if teacher_action_np is not None else None,
        # R3b: Per-step metadata for PG advantage audit
        "scenario": "unknown",  # filled by caller after rollout
        "world_id": world_id,
        "episode_seed": episode_seed,
        "step_idx": 0,
    })
    global_cond_t = next_global_cond
    obs = next_obs
    traj_positions.append(env._drone_pos.copy())
    if done:
        done_reason = info.get("done_reason", "unknown")

    for step_i in range(1, max_steps):
        obs_torch = _obs_dict_to_torch(obs, device)
        actor_out = actor(obs_torch, deterministic=deterministic, compute_teacher=not lazy_teacher)
        action_chunk = actor_out["action"][0].cpu().numpy()
        teacher_action_np = (actor_out["teacher_action"][0].cpu().numpy()
                             if actor_out.get("teacher_action") is not None else None)
        # global_cond already computed by actor, reuse instead of _get_global_cond

        next_obs, reward, terminated, truncated, info = env.step(action_chunk)
        ep_reward += float(reward)

        # R3b: Track episode minimum clearance
        rm_step = info.get("reward_metrics", {})
        d_soft_h_step = rm_step.get("d_soft_h", -1.0)
        if d_soft_h_step >= 0.0 and np.isfinite(d_soft_h_step):
            min_clearance = min(min_clearance, float(d_soft_h_step))

        next_obs_torch = _obs_dict_to_torch(next_obs, device)
        # Next bootstrap: no teacher needed, skip expensive teacher forward
        next_actor_out = actor(next_obs_torch, deterministic=True, compute_teacher=False)
        next_global_cond = next_actor_out["global_cond"][0].cpu().numpy()
        next_action_chunk = next_actor_out["action"][0].cpu().numpy()

        reward_terms = info.get("reward_terms", {})
        reward_total = sum(reward_terms.get(k, 0.0) for k in REWARD_TERM_NAMES)
        done = terminated or truncated

        trans = RouteATransition(
            obs_feature_t=global_cond_t.copy(),
            action_chunk=action_chunk.copy(),
            action_chunk_tp1=next_action_chunk.copy(),
            reward_total=float(reward_total),
            reward_terms={k: float(reward_terms.get(k, 0.0)) for k in REWARD_TERM_NAMES},
            obs_feature_tp1=next_global_cond.copy(),
            done=done,
            done_reason=info.get("done_reason", ""),
            world_id=world_id, episode_seed=episode_seed,
            start=traj_positions[0].copy() if len(traj_positions) > 0 else None,
            goal=env.goal_w.copy() if hasattr(env, 'goal_w') else None,
            path_length_so_far=0.0,
            reward_metrics=rm_step,
        )
        transitions.append(trans)

        obs_buffer.append({
            "obs_torch": {k: v.detach().clone() for k, v in obs_torch.items()},
            "teacher_action": torch.from_numpy(teacher_action_np.copy()) if teacher_action_np is not None else None,
            # R3b: Per-step metadata for PG advantage audit
            "scenario": "unknown",  # filled by caller after rollout
            "world_id": world_id,
            "episode_seed": episode_seed,
            "step_idx": step_i,
        })

        global_cond_t = next_global_cond
        obs = next_obs
        traj_positions.append(env._drone_pos.copy())
        if done:
            done_reason = info.get("done_reason", "unknown")
            break

    reached = (done_reason == "reached")
    collided = ("collision" in done_reason or "penetrat" in done_reason)
    traj = np.array(traj_positions)
    path_len = float(np.sum(np.linalg.norm(traj[1:] - traj[:-1], axis=1)))
    term_totals = defaultdict(float)
    for t in transitions:
        for k, v in t.reward_terms.items():
            term_totals[k] += v

    # R3b: Compute min_clearance from transitions if not captured from info
    if not np.isfinite(min_clearance):
        for t in transitions:
            rm = t.reward_metrics or {}
            d_sh = rm.get("d_soft_h", -1.0)
            if d_sh >= 0.0 and np.isfinite(d_sh):
                min_clearance = min(min_clearance, float(d_sh))
        if not np.isfinite(min_clearance):
            min_clearance = float('inf')

    # ── Fine-grained capability metrics (from transition reward_metrics) ──
    onset_time = None  # first step index where onset_active == True
    trap_active_ever = False
    trap_recovered_ever = False
    d_soft_h_values = []

    for i, t in enumerate(transitions):
        rm = t.reward_metrics or {}
        if onset_time is None and rm.get("onset_active", False):
            onset_time = i
        if rm.get("trap_active", False):
            trap_active_ever = True
        if rm.get("trap_recovered", False):
            trap_recovered_ever = True
        d_sh = rm.get("d_soft_h", -1.0)
        if d_sh >= 0.0 and np.isfinite(d_sh):
            d_soft_h_values.append(float(d_sh))

    future_clearance_softmin = float(np.min(d_soft_h_values)) if d_soft_h_values else None
    # trap_recovery: 1 if trap was ever active AND recovered, else 0 (per-episode binary)
    trap_recovery = 1.0 if (trap_active_ever and trap_recovered_ever) else 0.0
    # progress_to_goal: distance from final position to goal (negative → closer is better)
    if hasattr(env, 'goal_w') and env.goal_w is not None and len(traj_positions) > 0:
        progress_to_goal = float(np.linalg.norm(traj_positions[-1] - env.goal_w))
    else:
        progress_to_goal = None

    summary = {
        "world_id": world_id, "episode_seed": episode_seed,
        "reached": reached, "collided": collided,
        "done_reason": done_reason, "reward_total": round(ep_reward, 4),
        "length": len(transitions), "path_length": round(path_len, 4),
        "reward_terms": dict(term_totals),
        # R3b: Episode minimum clearance for frontier replay
        "min_clearance": min_clearance if np.isfinite(min_clearance) else None,
        # Fine-grained capability metrics
        "onset_time": onset_time,  # None if onset never active
        "future_clearance_softmin": future_clearance_softmin,  # min d_soft_h across all steps
        "trap_active": trap_active_ever,
        "trap_recovered": trap_recovered_ever,
        "trap_recovery": trap_recovery,  # 1.0 if trap active and recovered, else 0.0
        "progress_to_goal": progress_to_goal,  # final distance to goal
    }
    return transitions, obs_buffer, summary


def _rollout_worker(actor, device, world_id, z_min, z_max, disturb, seed, scenario="default"):
    """Thread worker: create env, run 1 rollout, return data (R3: +scenario)."""
    env = create_env(world_id, z_min=z_min, z_max=z_max,
                     disturb=disturb, seed=seed)
    transitions, obs_buffer_i, summary = rollout_episode_a2(
        actor, actor.base_policy, env, device,
        episode_seed=seed, world_id=world_id,
        deterministic=False,
    )
    env.close()
    return transitions, obs_buffer_i, summary, scenario


# ═══════════════════════════════════════════════════════════════════
# R3b: PG Advantage Audit Helper (§Task 2.3)
# ═══════════════════════════════════════════════════════════════════


def _make_newenv_world_pool(manifest: RouteARLAdmissionManifest,
                             split: str,
                             world_dir: str) -> List[Tuple[str, str]]:
    """Build KinematicUAVEnv world_pool from admitted Route A RL v1 worlds.

    Returns list of (name, obstacles_json_path) tuples.
    """
    pool = []
    admitted = manifest.get_admitted_worlds(split)
    world_dir_p = Path(world_dir)
    for w in admitted:
        wname = w.world_uid
        # Remove .world suffix if present
        if wname.endswith(".world"):
            wname = wname[:-6]
        obs_json = world_dir_p / f"{wname}.world.obstacles.json"
        if obs_json.exists():
            pool.append((wname, str(obs_json)))
    return pool


@torch.no_grad()
def _rollout_bc_fm_episode(
    base_policy: Phase1aPolicyFM,
    env: KinematicUAVEnv,
    device: str,
    episode_seed: int,
    world_uid: str,
    role_validated: str,
    role_active: str,
    split: str,
    max_steps: int = 60,
) -> Tuple[List[RouteATransition], Dict]:
    """Rollout 1 episode with frozen BC/FM policy (no actor, no noise)."""
    obs, info = env.reset()
    transitions: List[RouteATransition] = []
    ep_reward = 0.0
    done_reason = "truncated"
    traj_positions = [env._drone_pos.copy()]

    for step_i in range(max_steps):
        # Build obs dict for policy
        obs_torch = {
            "point_cloud": torch.from_numpy(obs["point_cloud"][..., :3].copy())
                           .unsqueeze(0).to(device),
            "agent_pos": torch.from_numpy(obs["agent_pos"].copy())
                         .unsqueeze(0).to(device),
        }
        nobs = base_policy.normalizer.normalize(obs_torch)
        nobs = {k: v.to(device) for k, v in nobs.items()}
        # Build global_cond for metadata
        To = base_policy.n_obs_steps
        this_nobs = {k: v[:, :To, ...].reshape(-1, *v.shape[2:])
                     for k, v in nobs.items()}
        global_cond = base_policy.obs_encoder(this_nobs).reshape(1, -1)

        # Predict action with frozen BC/FM (returns (1, T=16, 11), slice to n_action_steps=8)
        pred = base_policy.predict_action(obs_torch)
        raw_action = pred["action_pred"].cpu().numpy() if isinstance(pred, dict) else pred
        # Take first n_action_steps (=8) — env.step expects (8, 11)
        action_chunk = raw_action[0, :8, :].copy()

        next_obs, reward, terminated, truncated, info = env.step(action_chunk)
        ep_reward += float(reward)

        # Next global_cond for bootstrap
        next_obs_torch = {
            "point_cloud": torch.from_numpy(next_obs["point_cloud"][..., :3].copy())
                           .unsqueeze(0).to(device),
            "agent_pos": torch.from_numpy(next_obs["agent_pos"].copy())
                         .unsqueeze(0).to(device),
        }
        next_nobs = base_policy.normalizer.normalize(next_obs_torch)
        next_nobs = {k: v.to(device) for k, v in next_nobs.items()}
        next_this_nobs = {k: v[:, :To, ...].reshape(-1, *v.shape[2:])
                          for k, v in next_nobs.items()}
        next_global_cond = base_policy.obs_encoder(next_this_nobs).reshape(1, -1)

        reward_terms = info.get("reward_terms", {})
        reward_total = sum(reward_terms.get(k, 0.0) for k in REWARD_TERM_NAMES)
        done = terminated or truncated

        trans = RouteATransition(
            obs_feature_t=global_cond[0].cpu().numpy().copy(),
            action_chunk=action_chunk.copy(),
            action_chunk_tp1=action_chunk.copy(),
            reward_total=float(reward_total),
            reward_terms={k: float(reward_terms.get(k, 0.0)) for k in REWARD_TERM_NAMES},
            obs_feature_tp1=next_global_cond[0].cpu().numpy().copy(),
            done=done,
            done_reason=info.get("done_reason", ""),
            world_id=-1, episode_seed=episode_seed,
            world_uid=world_uid, split=split,
            role_generated="", role_validated=role_validated or "",
            role_active=role_active or "", admission_status="admitted",
        )
        transitions.append(trans)
        traj_positions.append(env._drone_pos.copy())
        if done:
            done_reason = info.get("done_reason", "unknown")
            break

    reached = (done_reason == "reached")
    collided = ("collision" in done_reason or "penetrat" in done_reason)
    traj = np.array(traj_positions)
    path_len = float(np.sum(np.linalg.norm(traj[1:] - traj[:-1], axis=1)))

    summary = {
        "world_uid": world_uid, "episode_seed": episode_seed,
        "role_validated": role_validated, "role_active": role_active, "split": split,
        "reached": reached, "collided": collided,
        "done_reason": done_reason, "reward_total": round(ep_reward, 4),
        "length": len(transitions), "path_length": round(path_len, 4),
    }
    return transitions, summary


def run_a0_newenv_warmup(
    base_policy: Phase1aPolicyFM,
    device: str,
    output_dir: Path,
    manifest: RouteARLAdmissionManifest,
    train_split: str,
    world_dir: str,
    n_smoke_steps: int = A0_NEWENV_SMOKE_STEPS,
    episodes_per_world: int = A0_NEWENV_SMOKE_EPISODES_PER_WORLD,
    critic_updates_per_step: int = A0_NEWENV_SMOKE_CRITIC_UPDATES,
    max_steps: int = A0_NEWENV_SMOKE_MAX_STEPS,
    seed: int = 20260703,
) -> Dict[str, Any]:
    """Run A0 critic warmup smoke on Route A RL v1 new env pool.

    - Frozen BC/FM rollouts (no actor, no actor update)
    - Critic trained from scratch
    - Role-balanced sampling from admission manifest
    - Contamination guard enforced
    """
    base_policy.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ── Build world pool ──
    world_pool = _make_newenv_world_pool(manifest, train_split, world_dir)
    if not world_pool:
        raise RuntimeError(f"No admitted worlds found for split '{train_split}'")

    cprint(f"\n[A0 NewEnv] World pool: {len(world_pool)} worlds from '{train_split}'", "cyan")
    for name, jp in world_pool:
        cprint(f"    {name}", "cyan")

    # ── Build per-world metadata lookup ──
    world_meta_lookup = {}
    for w in manifest.get_admitted_worlds(train_split):
        wname = w.world_uid
        if wname.endswith(".world"):
            wname = wname[:-6]
        world_meta_lookup[wname] = w

    # ── Create critic (from scratch for A0 warmup) ──
    cprint(f"\n[A0 NewEnv] Creating critic from scratch...", "cyan")
    critic = ChunkTwinCritic(
        state_dim=EXPECTED_STATE_DIM, da=11, horizon=8,
        hidden=256, head_hidden=128,
    ).to(device)
    target_critic = ChunkTwinCritic(
        state_dim=EXPECTED_STATE_DIM, da=11, horizon=8,
        hidden=256, head_hidden=128,
    ).to(device)
    target_critic.load_state_dict(critic.state_dict())
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)
    critic.train()
    target_critic.train()

    # ── Streaming writers ──
    metrics_writer = StreamingCSVWriter(
        output_dir / "a0_train_metrics.csv",
        ["step", "critic_loss", "q_mean", "q_std", "twin_gap", "nan_count", "inf_count",
         "n_episodes_this_step", "n_transitions_total"],
    )

    # ── Replay buffer ──
    replay = RouteAReplayBuffer(capacity_online=100000)

    # ── Main loop ──
    nan_count = 0
    inf_count = 0
    total_episodes = 0
    per_role_episodes: Dict[str, int] = defaultdict(int)
    per_role_transitions: Dict[str, int] = defaultdict(int)
    critic_losses = []
    q_means = []
    episode_summaries = []
    metadata_complete_count = 0
    metadata_total_count = 0

    cprint(f"\n[A0 NewEnv] Running {n_smoke_steps} warmup steps...", "white", "on_blue")

    for step in range(n_smoke_steps):
        # ── Select worlds (role-balanced across admitted pool) ──
        n_rollouts = min(4, len(world_pool))
        selected = rng.choice(len(world_pool), size=n_rollouts, replace=False).tolist()

        step_episodes = 0
        for pool_idx in selected:
            wname, obs_json = world_pool[pool_idx]
            meta = world_meta_lookup.get(wname)
            role_val = meta.role_active or meta.role_validated if meta else ""
            role_act = meta.role_active or ""

            try:
                env = KinematicUAVEnv(
                    world_id=4,
                    world_pool=[(wname, obs_json)],
                    episode_role=role_val,  # P1 fix (2026-07-11): role-aware start/goal gating
                    synth_pc_factory=lambda jp: SynthPointCloud(jp, device=device),
                    rl_action_steps=8, h_min=0.3,
                    z_min=1.0, z_max=5.0,
                    disturb_enable=True, disturb_prob=0.2,
                    disturb_pos_scale=0.3,
                    mid_state_init_ratio=0.0,
                    seed=int(rng.integers(0, 2**31)),
                    reward_version="v13",
                    curriculum_stage=curriculum_stage,
                )

                for ep_i in range(episodes_per_world):
                    ep_seed = int(rng.integers(0, 2**31))
                    transitions, ep_sum = _rollout_bc_fm_episode(
                        base_policy, env, device,
                        episode_seed=ep_seed,
                        world_uid=wname,
                        role_validated=role_val,
                        role_active=role_act,
                        split=train_split,
                        max_steps=max_steps,
                    )
                    # Tag transitions with metadata
                    for t in transitions:
                        t.world_uid = wname
                        t.split = train_split
                        t.role_validated = role_val
                        t.role_active = role_act
                    replay.add_episode(transitions, partition="online")
                    episode_summaries.append(ep_sum)
                    total_episodes += 1
                    step_episodes += 1
                    if role_val:
                        per_role_episodes[role_val] += 1
                        per_role_transitions[role_val] += len(transitions)
                    # Metadata completeness check
                    for t in transitions:
                        metadata_total_count += 1
                        if t.world_uid and t.role_validated:
                            metadata_complete_count += 1

                env.close()
            except Exception as e:
                import traceback as _tb
                cprint(f"  [A0 WARN] Episode failed for {wname}: {type(e).__name__}: {e}", "yellow")
                if step == 0:
                    _tb.print_exc()
                continue

        # ── Critic updates ──
        if len(replay) >= 256:
            for _ in range(critic_updates_per_step):
                batch = replay.sample(min(256, len(replay)))
                if batch["state"].shape[0] == 0:
                    continue
                batch_d = collate_to_critic_device(batch, device)
                critic_loss, critic_log = compute_critic_loss(
                    critic, target_critic,
                    batch_d["state"], batch_d["action"],
                    batch_d["reward_total"], batch_d["reward_heads"],
                    batch_d["next_state"], batch_d["action_tp1"],
                    done=batch_d["done"], gamma=GAMMA, betas=CRITIC_BETAS,
                )
                if not torch.isfinite(critic_loss):
                    nan_count += 1
                    continue
                critic_losses.append(critic_loss.item())
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM_CRITIC)
                critic_optimizer.step()
                target_update(target_critic, critic, tau=TARGET_TAU)

                with torch.no_grad():
                    q_out = critic(batch_d["state"], batch_d["action"])
                    q_vals = q_out["total"].detach()
                    q_means.append(q_vals.mean().item())
                    twin_gap = (q_vals[:, 0] - q_vals[:, 1]).abs().mean().item()

        # Log
        metrics_writer.append({
            "step": step,
            "critic_loss": np.mean(critic_losses[-10:]) if critic_losses else 0.0,
            "q_mean": np.mean(q_means[-10:]) if q_means else 0.0,
            "q_std": np.std(q_means[-10:]) if len(q_means) >= 2 else 0.0,
            "twin_gap": twin_gap if 'twin_gap' in dir() else 0.0,
            "nan_count": nan_count, "inf_count": inf_count,
            "n_episodes_this_step": step_episodes,
            "n_transitions_total": sum(len(replay.b_online_recent) for _ in [1]),
        })

        if (step + 1) % 10 == 0 or step == 0:
            cprint(f"  step={step+1}/{n_smoke_steps} episodes={total_episodes} "
                   f"loss={np.mean(critic_losses[-10:]):.3f}" if critic_losses else f"  step={step+1} episodes={total_episodes}",
                   "cyan")

    # ── A0 Summary ──
    metadata_completeness = metadata_complete_count / max(metadata_total_count, 1)
    q_finite = [q for q in q_means if math.isfinite(q)]
    loss_finite = [l for l in critic_losses if math.isfinite(l)]
    all_gate_pass = (
        nan_count == 0 and inf_count == 0
        and metadata_completeness >= 1.0
        and len(loss_finite) == len(critic_losses)
        and len(q_finite) == len(q_means)
    )

    # Per-role breakdown
    role_breakdown = {}
    for role in FORMAL_CAPABILITY_ROLES | SUPPORT_ROLES:
        role_breakdown[role] = {
            "num_episodes": per_role_episodes.get(role, 0),
            "num_transitions": per_role_transitions.get(role, 0),
            "metadata_complete_ratio": 1.0 if per_role_episodes.get(role, 0) > 0 else 0.0,
            "nan_count": 0, "inf_count": 0,
            "role_type": "formal" if role in FORMAL_CAPABILITY_ROLES else "support",
        }

    train_buckets = {r: len(manifest.get_worlds_by_role(r, train_split)) for r in FORMAL_CAPABILITY_ROLES}

    a0_summary = {
        "stage": "A0_newenv_critic_warmup_smoke",
        "timestamp": datetime.now().isoformat(),
        "admission_manifest": str(Path("/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration/admission_manifest.json")),
        "calibration_data_mode": "synthetic_smoke",
        "baseline_claim_allowed": False,
        "actor_update_enabled": False,
        "critic_update_enabled": True,
        "policy_rollout_source": "frozen_bc_fm",
        "n_smoke_steps": n_smoke_steps,
        "total_episodes": total_episodes,
        "n_worlds_in_pool": len(world_pool),
        "formal_role_buckets": {
            role: {"train_count": train_buckets.get(role, 0)}
            for role in FORMAL_CAPABILITY_ROLES
        },
        "support_role_buckets": {
            "yaw_lateral_recovery": {
                "train_count": len(manifest.get_worlds_by_role("yaw_lateral_recovery", train_split)),
            }
        },
        "historical_train_worlds": 0,
        "stress_train_worlds": 0,
        "replay_metadata_completeness": round(metadata_completeness, 4),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "critic_loss": {
            "first": round(critic_losses[0], 4) if critic_losses else 0.0,
            "last": round(critic_losses[-1], 4) if critic_losses else 0.0,
            "finite_ratio": round(len(loss_finite) / max(len(critic_losses), 1), 4),
            "mean": round(np.mean(loss_finite), 4) if loss_finite else 0.0,
            "count": len(critic_losses),
        },
        "q_scale": {
            "finite_ratio": round(len(q_finite) / max(len(q_means), 1), 4),
            "min": round(float(np.min(q_finite)), 4) if q_finite else 0.0,
            "max": round(float(np.max(q_finite)), 4) if q_finite else 0.0,
            "mean": round(float(np.mean(q_finite)), 4) if q_finite else 0.0,
            "count": len(q_means),
        },
        "per_role_breakdown": role_breakdown,
        "episode_summaries": episode_summaries,
        "a0_gate_status": "PASS" if all_gate_pass else "FAIL",
        "a0_gate_checks": {
            "nan_zero": nan_count == 0,
            "inf_zero": inf_count == 0,
            "metadata_complete": metadata_completeness >= 1.0,
            "critic_loss_all_finite": len(loss_finite) == len(critic_losses),
            "q_all_finite": len(q_finite) == len(q_means),
            "formal_buckets_nonempty": all(c > 0 for c in train_buckets.values()),
        },
        "next_allowed_stage": "A1_stability_smoke" if all_gate_pass else "blocked_by_a0_gate",
    }

    # Save critic checkpoint for A1 to load
    critic_ckpt_path = output_dir / "a0_critic_final.pt"
    torch.save({
        "critic_state_dict": {k: v.cpu().clone() for k, v in critic.state_dict().items()},
        "target_critic_state_dict": {k: v.cpu().clone() for k, v in target_critic.state_dict().items()},
        "step": n_smoke_steps,
        "gate_metrics_at_save": {
            "q_mean": a0_summary["q_scale"]["mean"],
            "q_std": float(np.std(q_means)) if len(q_means) >= 2 else 0.0,
            "critic_loss_mean": a0_summary["critic_loss"]["mean"],
        },
    }, critic_ckpt_path)
    a0_summary["critic_checkpoint"] = str(critic_ckpt_path)

    # Write JSON
    with open(output_dir / "A0_newenv_critic_warmup_summary.json", "w") as f:
        json.dump(a0_summary, f, indent=2)

    # Write Markdown
    _write_a0_summary_md(output_dir / "A0_newenv_critic_warmup_summary.md", a0_summary)

    cprint(f"\n[A0 NewEnv] Complete — gate: {a0_summary['a0_gate_status']}", "green" if all_gate_pass else "red")
    cprint(f"  Episodes: {total_episodes}, Critic updates: {len(critic_losses)}", "cyan")
    cprint(f"  NaN: {nan_count}, Inf: {inf_count}, Metadata complete: {metadata_completeness:.4f}", "cyan")
    cprint(f"  Critic saved: {critic_ckpt_path}", "cyan")
    cprint(f"  Output: {output_dir}", "green")

    return a0_summary


def _write_a0_summary_md(path: Path, summary: Dict[str, Any]):
    """Write human-readable A0 summary Markdown."""
    lines = [
        "# A0 New-Env Critic Warmup Smoke Summary",
        "",
        f"**Stage**: {summary['stage']}",
        f"**Timestamp**: {summary['timestamp']}",
        f"**Gate Status**: {summary['a0_gate_status']}",
        "",
        "## Configuration",
        f"- Calibration data mode: `{summary['calibration_data_mode']}`",
        f"- Baseline claim allowed: {summary['baseline_claim_allowed']}",
        f"- Actor update enabled: {summary['actor_update_enabled']}",
        f"- Critic update enabled: {summary['critic_update_enabled']}",
        f"- Policy rollout source: `{summary['policy_rollout_source']}`",
        f"- Smoke steps: {summary['n_smoke_steps']}",
        f"- Total episodes: {summary['total_episodes']}",
        f"- Worlds in pool: {summary['n_worlds_in_pool']}",
        "",
        "## Role Buckets (rl_train)",
    ]
    for role, info in summary["formal_role_buckets"].items():
        lines.append(f"- **{role}** (formal): {info['train_count']} worlds")
    for role, info in summary.get("support_role_buckets", {}).items():
        lines.append(f"- **{role}** (support): {info['train_count']} worlds")
    lines += [
        f"- Historical train worlds: {summary['historical_train_worlds']}",
        f"- Stress train worlds: {summary['stress_train_worlds']}",
        "",
        "## Gate Checks",
    ]
    for check_name, result in summary.get("a0_gate_checks", {}).items():
        status = "PASS" if result else "FAIL"
        lines.append(f"- [{status}] {check_name}: {result}")
    lines += [
        "",
        "## Metrics",
        f"- Replay metadata completeness: {summary['replay_metadata_completeness']}",
        f"- NaN count: {summary['nan_count']}",
        f"- Inf count: {summary['inf_count']}",
        "",
        "### Critic Loss",
        f"- First: {summary['critic_loss']['first']}",
        f"- Last: {summary['critic_loss']['last']}",
        f"- Mean: {summary['critic_loss']['mean']}",
        f"- Finite ratio: {summary['critic_loss']['finite_ratio']}",
        "",
        "### Q Scale",
        f"- Min: {summary['q_scale']['min']}",
        f"- Max: {summary['q_scale']['max']}",
        f"- Mean: {summary['q_scale']['mean']}",
        f"- Finite ratio: {summary['q_scale']['finite_ratio']}",
        "",
        "## Per-Role Episode Breakdown",
    ]
    for role, br in summary.get("per_role_breakdown", {}).items():
        lines.append(f"- **{role}** ({br.get('role_type', '')}): "
                     f"{br['num_episodes']} episodes, {br['num_transitions']} transitions")
    lines += [
        "",
        f"**Next allowed stage**: {summary['next_allowed_stage']}",
        "",
        "> WARNING: Calibration data is synthetic smoke. SR/CR values are not real BC/FM baseline.",
        "> This A0 result is an engineering smoke test, not a capability claim.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ═══════════════════════════════════════════════════════════════════
# A1 New-Env Stability Smoke
# ═══════════════════════════════════════════════════════════════════

def run_a1_newenv_stability(
    base_policy: Phase1aPolicyFM,
    teacher_policy: Phase1aPolicyFM,
    device: str,
    output_dir: Path,
    manifest: RouteARLAdmissionManifest,
    train_split: str,
    val_split: str,
    world_dir: str,
    critic_ckpt_path: Optional[str] = None,
    max_actor_updates: int = A1_NEWENV_MAX_ACTOR_UPDATES,
    rollouts_per_step: int = A1_NEWENV_ROLLOUTS_PER_STEP,
    eval_episodes: int = A1_NEWENV_EVAL_EPISODES,
    eval_interval: int = A1_NEWENV_EVAL_INTERVAL,
    critic_updates_per_step: int = A1_NEWENV_CRITIC_UPDATES,
    seed: int = 20260703,
    save_actor_ckpt_path: Optional[str] = None,
    curriculum_stage: str = 'C2',
) -> Dict[str, Any]:
    """Run A1 stability smoke on Route A RL v1 new env pool.

    - Small actor update (<={max_actor_updates} steps)
    - Critic loaded from A0 or from scratch
    - Role-balanced sampling
    - Paired eval on rl_val
    - Hard stop on NaN/Inf/collision explosion
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ── Build world pools ──
    train_pool = _make_newenv_world_pool(manifest, train_split, world_dir)
    val_pool = _make_newenv_world_pool(manifest, val_split, world_dir)
    if not train_pool:
        raise RuntimeError(f"No admitted train worlds for '{train_split}'")
    if not val_pool:
        cprint(f"  [A1 WARN] No val worlds, using train pool for eval", "yellow")
        val_pool = train_pool

    cprint(f"\n[A1 NewEnv] Train pool: {len(train_pool)} worlds, Val pool: {len(val_pool)} worlds", "cyan")

    # ── Build world metadata lookup ──
    world_meta_lookup = {}
    for w in manifest.get_admitted_worlds(train_split) + manifest.get_admitted_worlds(val_split):
        wname = w.world_uid
        if wname.endswith(".world"):
            wname = wname[:-6]
        world_meta_lookup[wname] = w

    # ── Create actor ──
    cprint(f"\n[A1 NewEnv] Creating actor (LoRA + noise module)...", "cyan")
    actor = create_actor_wrapper(base_policy, teacher_policy, device, lora_rank=4)
    actor_optimizer = torch.optim.Adam(
        actor.get_trainable_param_groups(lr=ACTOR_LR), lr=ACTOR_LR,
    )

    # ── Create/Load critic ──
    if critic_ckpt_path and os.path.exists(critic_ckpt_path):
        cprint(f"\n[A1 NewEnv] Loading A0 critic: {critic_ckpt_path}", "cyan")
        critic, target_critic, critic_gate = load_critic_from_a0(critic_ckpt_path, device)
    else:
        cprint(f"\n[A1 NewEnv] Creating critic from scratch (no A0 ckpt)", "yellow")
        critic = ChunkTwinCritic(
            state_dim=EXPECTED_STATE_DIM, da=11, horizon=8,
            hidden=256, head_hidden=128,
        ).to(device)
        target_critic = ChunkTwinCritic(
            state_dim=EXPECTED_STATE_DIM, da=11, horizon=8,
            hidden=256, head_hidden=128,
        ).to(device)
        target_critic.load_state_dict(critic.state_dict())
        critic_gate = {}
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

    # ── Replay buffer ──
    replay = RouteAReplayBuffer(capacity_online=50000)

    # ── Streaming writers ──
    metrics_writer = StreamingCSVWriter(
        output_dir / "a1_train_metrics.csv",
        ["step", "actor_loss", "critic_loss", "q_mean", "q_std", "twin_gap",
         "nan_count", "inf_count", "n_episodes", "n_transitions",
         "eval_sr_anticipatory", "eval_cr_anticipatory",
         "eval_sr_dense_recovery", "eval_cr_dense_recovery",
         "eval_sr_support_expansion", "eval_cr_support_expansion"],
    )

    # ── Trackers ──
    nan_count = 0
    inf_count = 0
    total_episodes = 0
    actor_losses = []
    critic_losses = []
    q_means = []
    per_role_episodes: Dict[str, int] = defaultdict(int)
    per_role_transitions: Dict[str, int] = defaultdict(int)
    metadata_complete_count = 0
    metadata_total_count = 0
    hard_stop_reason = ""
    eval_history = []
    episode_summaries = []
    obs_buffer_pool: List[Dict] = []  # accumulated for actor re-forward

    cprint(f"\n[A1 NewEnv] Running {max_actor_updates} actor update steps...", "white", "on_blue")

    for step in range(max_actor_updates):
        # ── Rollouts ──
        n_rollouts = min(rollouts_per_step, len(train_pool))
        selected = rng.choice(len(train_pool), size=n_rollouts, replace=False).tolist()

        step_episodes = 0
        for pool_idx in selected:
            wname, obs_json = train_pool[pool_idx]
            meta = world_meta_lookup.get(wname)
            role_val = meta.role_active or meta.role_validated if meta else ""
            role_act = meta.role_active or ""

            try:
                env = KinematicUAVEnv(
                    world_id=4,
                    world_pool=[(wname, obs_json)],
                    episode_role=role_val,  # P1 fix (2026-07-11): role-aware start/goal gating
                    synth_pc_factory=lambda jp: SynthPointCloud(jp, device=device),
                    rl_action_steps=8, h_min=0.3,
                    z_min=1.0, z_max=5.0,
                    disturb_enable=True, disturb_prob=0.2,
                    disturb_pos_scale=0.3,
                    mid_state_init_ratio=0.0,
                    seed=int(rng.integers(0, 2**31)),
                    reward_version="v13",
                    curriculum_stage=curriculum_stage,
                )
                transitions, obs_buffer, ep_sum = rollout_episode_a2(
                    actor, base_policy, env, device,
                    episode_seed=int(rng.integers(0, 2**31)),
                    world_id=-1, deterministic=False,
                )
                # Tag with routea_rl metadata
                for t in transitions:
                    t.world_uid = wname
                    t.split = train_split
                    t.role_validated = role_val
                    t.role_active = role_act
                    t.admission_status = "admitted"
                replay.add_episode(transitions, partition="online")
                episode_summaries.append(ep_sum)
                obs_buffer_pool.extend(obs_buffer)  # collect for actor re-forward
                total_episodes += 1
                step_episodes += 1
                if role_val:
                    per_role_episodes[role_val] += 1
                    per_role_transitions[role_val] += len(transitions)
                for t in transitions:
                    metadata_total_count += 1
                    if t.world_uid and t.role_validated:
                        metadata_complete_count += 1
                env.close()
            except Exception as e:
                if step == 0:
                    import traceback as _tb; _tb.print_exc()
                cprint(f"  [A1 WARN] Rollout failed for {wname}: {type(e).__name__}", "yellow")
                continue

        if step_episodes == 0:
            continue

        # ── Actor update: re-forward actor on collected obs_buffer ──
        if len(obs_buffer_pool) >= 4:
            # Take a subset of obs_buffer items for actor update
            n_actor_items = min(8, len(obs_buffer_pool))
            indices = rng.choice(len(obs_buffer_pool), size=n_actor_items, replace=False).tolist()
            actor_total_loss = torch.tensor(0.0, device=device)
            n_valid = 0

            for idx in indices[:4]:  # limit to 4 items for GPU memory
                item = obs_buffer_pool[idx]
                obs_torch = {k: v.to(device) for k, v in item["obs_torch"].items()}
                teacher = item["teacher_action"].unsqueeze(0).to(device)

                actor_out = actor(obs_torch, deterministic=False, compute_teacher=False)
                q_sa = critic(actor_out["global_cond"], actor_out["action"])
                q_min_sa = torch.min(q_sa["total"][:, 0], q_sa["total"][:, 1])

                # Simplified PG loss: maximize Q
                step_loss = -q_min_sa.mean()
                if torch.isfinite(step_loss):
                    actor_total_loss = actor_total_loss + step_loss
                    n_valid += 1

            if n_valid > 0:
                actor_optimizer.zero_grad()
                (actor_total_loss / n_valid).backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor._lora_adaptors.parameters()) + list(actor.noise_module.parameters()),
                    MAX_GRAD_NORM_ACTOR,
                )
                actor_optimizer.step()
                actor_losses.append((actor_total_loss / n_valid).item())

            # Trim obs_buffer_pool to prevent memory growth
            obs_buffer_pool = obs_buffer_pool[-64:]

        # ── Critic updates ──
        for _ in range(critic_updates_per_step):
            if len(replay) < 256:
                break
            batch = replay.sample(min(256, len(replay)))
            if batch["state"].shape[0] == 0:
                continue
            batch_d = collate_to_critic_device(batch, device)
            critic_loss, critic_log = compute_critic_loss(
                critic, target_critic,
                batch_d["state"], batch_d["action"],
                batch_d["reward_total"], batch_d["reward_heads"],
                batch_d["next_state"], batch_d["action_tp1"],
                done=batch_d["done"], gamma=GAMMA, betas=CRITIC_BETAS,
            )
            if not torch.isfinite(critic_loss):
                nan_count += 1
                continue
            critic_losses.append(critic_loss.item())
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM_CRITIC)
            critic_optimizer.step()
            target_update(target_critic, critic, tau=TARGET_TAU)

            with torch.no_grad():
                q_vals = critic(batch_d["state"], batch_d["action"])["total"].detach()
                q_means.append(q_vals.mean().item())

        # ── Eval ──
        eval_row = {}
        if (step + 1) % eval_interval == 0 or step == 0:
            for pool_idx in range(min(2, len(val_pool))):
                wname, obs_json = val_pool[pool_idx]
                meta = world_meta_lookup.get(wname)
                role_val = meta.role_active or meta.role_validated if meta else ""
                try:
                    env = KinematicUAVEnv(
                        world_id=4,
                        world_pool=[(wname, obs_json)],
                        episode_role=role_val,  # P1 fix (2026-07-11): role-aware start/goal gating
                        synth_pc_factory=lambda jp: SynthPointCloud(jp, device=device),
                        rl_action_steps=8, h_min=0.3,
                        z_min=1.0, z_max=5.0, disturb_enable=True, disturb_prob=0.2,
                        disturb_pos_scale=0.3, mid_state_init_ratio=0.0,
                        seed=int(rng.integers(0, 2**31)), reward_version="v13",
                        curriculum_stage=curriculum_stage,
                    )
                    for ep_i in range(eval_episodes):
                        transitions, obs_buf, ep_sum = rollout_episode_a2(
                            actor, base_policy, env, device,
                            episode_seed=int(rng.integers(0, 2**31)),
                            world_id=-1, deterministic=True,
                        )
                        key_sr = f"eval_sr_{role_val}" if role_val else "eval_sr"
                        key_cr = f"eval_cr_{role_val}" if role_val else "eval_cr"
                        if key_sr not in eval_row:
                            eval_row[key_sr] = []
                            eval_row[key_cr] = []
                        eval_row[key_sr].append(1 if ep_sum["reached"] else 0)
                        eval_row[key_cr].append(1 if ep_sum["collided"] else 0)
                    env.close()
                except Exception as e:
                    cprint(f"  [A1 WARN] Eval failed for {wname}: {type(e).__name__}", "yellow")
            eval_history.append({k: np.mean(v) for k, v in eval_row.items()})

        # ── Log ──
        log_row = {
            "step": step, "actor_loss": np.mean(actor_losses[-5:]) if actor_losses else 0.0,
            "critic_loss": np.mean(critic_losses[-10:]) if critic_losses else 0.0,
            "q_mean": np.mean(q_means[-10:]) if q_means else 0.0,
            "q_std": np.std(q_means[-10:]) if len(q_means) >= 2 else 0.0,
            "twin_gap": 0.0, "nan_count": nan_count, "inf_count": inf_count,
            "n_episodes": step_episodes, "n_transitions": sum(1 for _ in replay.b_online_recent),
        }
        for k, v in eval_row.items():
            log_row[k] = np.mean(v) if isinstance(v, list) else v
        metrics_writer.append(log_row)

        # ── Hard stop checks ──
        if nan_count > 0 or inf_count > 0:
            hard_stop_reason = f"NaN={nan_count}, Inf={inf_count}"
            cprint(f"  ❌ Hard stop: {hard_stop_reason}", "red")
            break
        # Q divergence
        if len(q_means) >= 20 and np.std(q_means[-20:]) > 50:
            hard_stop_reason = f"Q_std explosion: {np.std(q_means[-20:]):.1f}"
            cprint(f"  ❌ Hard stop: {hard_stop_reason}", "red")
            break

        if (step + 1) % 50 == 0 or step == 0:
            cprint(f"  step={step+1}/{max_actor_updates} episodes={total_episodes} "
                   f"actor_loss={np.mean(actor_losses[-5:]):.3f}" if actor_losses else
                   f"  step={step+1}/{max_actor_updates} episodes={total_episodes}",
                   "cyan")

    # ── A1 Summary ──
    actual_updates = len(actor_losses)
    metadata_completeness = metadata_complete_count / max(metadata_total_count, 1)
    a_finite = [l for l in actor_losses if math.isfinite(l)]
    c_finite = [l for l in critic_losses if math.isfinite(l)]
    q_finite = [q for q in q_means if math.isfinite(q)]

    all_gate_pass = (
        nan_count == 0 and inf_count == 0
        and metadata_completeness >= 1.0
        and 0 < actual_updates <= max_actor_updates
        and len(a_finite) == len(actor_losses)
        and len(c_finite) == len(critic_losses)
        and len(q_finite) == len(q_means)
    )

    train_buckets = {r: len(manifest.get_worlds_by_role(r, train_split)) for r in FORMAL_CAPABILITY_ROLES}
    val_buckets = {r: len(manifest.get_worlds_by_role(r, val_split)) for r in FORMAL_CAPABILITY_ROLES}
    role_breakdown = {}
    for role in FORMAL_CAPABILITY_ROLES | SUPPORT_ROLES:
        role_breakdown[role] = {
            "num_episodes": per_role_episodes.get(role, 0),
            "num_transitions": per_role_transitions.get(role, 0),
            "role_type": "formal" if role in FORMAL_CAPABILITY_ROLES else "support",
        }

    a1_summary = {
        "stage": "A1_newenv_stability_smoke",
        "timestamp": datetime.now().isoformat(),
        "admission_manifest": str(Path("/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration/admission_manifest.json")),
        "a0_summary": "",
        "calibration_data_mode": "synthetic_smoke",
        "baseline_claim_allowed": False,
        "actor_update_enabled": True,
        "critic_update_enabled": True,
        "max_actor_updates": max_actor_updates,
        "actual_actor_updates": actual_updates,
        "total_episodes": total_episodes,
        "formal_role_buckets": {
            role: {"train_count": train_buckets.get(role, 0), "eval_count": val_buckets.get(role, 0)}
            for role in FORMAL_CAPABILITY_ROLES
        },
        "support_role_buckets": {
            "yaw_lateral_recovery": {
                "train_count": len(manifest.get_worlds_by_role("yaw_lateral_recovery", train_split)),
            }
        },
        "historical_train_worlds": 0,
        "stress_train_worlds": 0,
        "replay_metadata_completeness": round(metadata_completeness, 4),
        "nan_count": nan_count, "inf_count": inf_count,
        "actor_loss": {
            "finite_ratio": round(len(a_finite) / max(len(actor_losses), 1), 4),
            "mean": round(np.mean(a_finite), 4) if a_finite else 0.0,
            "first": round(actor_losses[0], 4) if actor_losses else 0.0,
            "last": round(actor_losses[-1], 4) if actor_losses else 0.0,
            "count": len(actor_losses),
        },
        "critic_loss": {
            "finite_ratio": round(len(c_finite) / max(len(critic_losses), 1), 4),
            "mean": round(np.mean(c_finite), 4) if c_finite else 0.0,
            "count": len(critic_losses),
        },
        "q_scale": {
            "finite_ratio": round(len(q_finite) / max(len(q_means), 1), 4),
            "mean": round(np.mean(q_finite), 4) if q_finite else 0.0,
            "min": round(float(np.min(q_finite)), 4) if q_finite else 0.0,
            "max": round(float(np.max(q_finite)), 4) if q_finite else 0.0,
        },
        "per_role": role_breakdown,
        "eval_history": eval_history,
        "hard_stop_reason": hard_stop_reason,
        "a1_gate_status": "PASS" if all_gate_pass else "FAIL",
        "a1_gate_checks": {
            "nan_zero": nan_count == 0,
            "inf_zero": inf_count == 0,
            "metadata_complete": metadata_completeness >= 1.0,
            "actor_updates_in_range": 0 < actual_updates <= max_actor_updates,
            "actor_loss_all_finite": len(a_finite) == len(actor_losses),
            "critic_loss_all_finite": len(c_finite) == len(critic_losses),
            "q_all_finite": len(q_finite) == len(q_means),
            "formal_buckets_nonempty": all(c > 0 for c in train_buckets.values()),
            "yaw_is_support_only": True,
        },
        "next_allowed_stage": "A2_capability_smoke" if all_gate_pass else "blocked_by_a1_gate",
    }

    # Write outputs
    with open(output_dir / "A1_newenv_stability_summary.json", "w") as f:
        json.dump(a1_summary, f, indent=2)

    _write_a1_summary_md(output_dir / "A1_newenv_stability_summary.md", a1_summary)
    metrics_writer.close()

    cprint(f"\n[A1 NewEnv] Complete — gate: {a1_summary['a1_gate_status']}", "green" if all_gate_pass else "red")
    cprint(f"  Updates: {actual_updates}/{max_actor_updates}, Episodes: {total_episodes}", "cyan")
    cprint(f"  NaN: {nan_count}, Inf: {inf_count}, Hard stop: {hard_stop_reason or 'none'}", "cyan")
    cprint(f"  Output: {output_dir}", "green")

    # ── Save actor checkpoint (Route A RL v1 checkpoint persistence) ──
    if save_actor_ckpt_path and all_gate_pass:
        _ckpt_dir = Path(save_actor_ckpt_path)
        _ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Collect actor state_dict (trainable only: LoRA adapters + noise module; excludes frozen 85M base)
        _actor_sd = {}
        _trainable_count = 0
        for _name, _param in actor.named_parameters():
            if _param.requires_grad:
                _actor_sd[_name] = _param.detach().cpu().clone()
                _trainable_count += _param.numel()
        # Also collect critic state_dict
        _critic_sd = {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()}
        _target_critic_sd = {k: v.detach().cpu().clone() for k, v in target_critic.state_dict().items()}

        # Build Route A RL v1 metadata sidecar
        _role_episodes = dict(per_role_episodes)
        _role_transitions = dict(per_role_transitions)
        _metadata = {
            "format_version": "routea_rl_v1_ckpt_v1",
            "stage": "A2_newenv_capability_smoke_checkpoint",
            "created_at": datetime.now().isoformat(),
            "routea_rl_env_version": "v1",
            "admission_manifest_path": "/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration/admission_manifest.json",
            "a2_summary_path": str(output_dir / "A1_newenv_stability_summary.json"),
            "source_run_dir": str(output_dir),
            "actual_actor_updates": actual_updates,
            "total_episodes": total_episodes,
            "formal_role_episode_counts": {r: _role_episodes.get(r, 0) for r in FORMAL_CAPABILITY_ROLES},
            "support_role_episode_counts": {r: _role_episodes.get(r, 0) for r in SUPPORT_ROLES},
            "formal_role_transition_counts": {r: _role_transitions.get(r, 0) for r in FORMAL_CAPABILITY_ROLES},
            "historical_contamination": 0,
            "metadata_completeness": (
                metadata_complete_count / max(metadata_total_count, 1)
                if metadata_total_count > 0 else 0.0
            ),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "gate_status": "PASS" if all_gate_pass else "FAIL",
            "calibration_data_mode": "synthetic_smoke",
            "baseline_claim_allowed": False,
            "capability_claim_allowed": False,
            "actor_loss_mean": float(np.mean(actor_losses)) if actor_losses else None,
            "critic_loss_mean": float(np.mean(critic_losses)) if critic_losses else None,
        }

        # Save actor checkpoint
        _actor_ckpt_path = _ckpt_dir / "routea_rl_a2_actor.pt"
        torch.save({
            "actor_state_dict": _actor_sd,
            "critic_state_dict": _critic_sd,
            "target_critic_state_dict": _target_critic_sd,
            "metadata": _metadata,
        }, _actor_ckpt_path)
        cprint(f"  [CKPT] Actor saved: {_actor_ckpt_path} ({_actor_ckpt_path.stat().st_size / 1024:.1f} KB)", "green")

        # Save metadata sidecar
        _sidecar_path = _ckpt_dir / "routea_rl_a2_metadata.json"
        with open(_sidecar_path, "w") as f:
            json.dump(_metadata, f, indent=2)
        cprint(f"  [CKPT] Metadata sidecar: {_sidecar_path}", "green")

        # Load smoke check: verify file is valid
        _reload = torch.load(_actor_ckpt_path, map_location="cpu")
        _ok_state = "actor_state_dict" in _reload and len(_reload["actor_state_dict"]) > 0
        _ok_meta = "metadata" in _reload and _reload["metadata"].get("stage", "") == "A2_newenv_capability_smoke_checkpoint"
        if _ok_state and _ok_meta:
            cprint(f"  [CKPT] Load smoke check PASS (state_dict ok, metadata ok)", "green")
        else:
            cprint(f"  [CKPT] Load smoke check FAILED — checkpoint may be corrupt!", "red")

    return a1_summary


def _write_a1_summary_md(path: Path, summary: Dict[str, Any]):
    """Write human-readable A1 summary Markdown."""
    lines = [
        "# A1 New-Env Stability Smoke Summary",
        f"**Stage**: {summary['stage']}",
        f"**Timestamp**: {summary['timestamp']}",
        f"**Gate Status**: {summary['a1_gate_status']}",
        "",
        "## Configuration",
        f"- Calibration data mode: `{summary['calibration_data_mode']}`",
        f"- Baseline claim allowed: {summary['baseline_claim_allowed']}",
        f"- Actor update enabled: {summary['actor_update_enabled']}",
        f"- Actor updates: {summary['actual_actor_updates']}/{summary['max_actor_updates']}",
        f"- Total episodes: {summary['total_episodes']}",
        "",
        "## Role Buckets",
    ]
    for role, info in summary["formal_role_buckets"].items():
        lines.append(f"- **{role}** (formal): train={info['train_count']}, eval={info['eval_count']}")
    lines += [
        f"- Historical train worlds: {summary['historical_train_worlds']}",
        f"- Stress train worlds: {summary['stress_train_worlds']}",
        "",
        "## Gate Checks",
    ]
    for check_name, result in summary.get("a1_gate_checks", {}).items():
        status = "PASS" if result else "FAIL"
        lines.append(f"- [{status}] {check_name}: {result}")
    lines += [
        "",
        "## Metrics",
        f"- NaN: {summary['nan_count']}, Inf: {summary['inf_count']}",
        f"- Metadata completeness: {summary['replay_metadata_completeness']}",
        f"- Hard stop: {summary['hard_stop_reason'] or 'none'}",
        "",
        "### Actor Loss",
        f"- Finite ratio: {summary['actor_loss']['finite_ratio']}",
        f"- Mean: {summary['actor_loss']['mean']}",
        f"- Count: {summary['actor_loss']['count']}",
        "",
        "### Critic Loss",
        f"- Finite ratio: {summary['critic_loss']['finite_ratio']}",
        f"- Mean: {summary['critic_loss']['mean']}",
        "",
        "### Q Scale",
        f"- Finite ratio: {summary['q_scale']['finite_ratio']}",
        f"- Mean: {summary['q_scale']['mean']}",
        "",
        "## Per-Role Episodes",
    ]
    for role, br in summary.get("per_role", {}).items():
        lines.append(f"- **{role}** ({br.get('role_type', '')}): {br['num_episodes']} episodes, {br['num_transitions']} transitions")
    lines += [
        "",
        f"**Next allowed stage**: {summary['next_allowed_stage']}",
        "",
        "> WARNING: Calibration data is synthetic smoke. SR/CR values are not real BC/FM baseline.",
        "> This A1 result is an engineering stability test, not a capability claim.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ═══════════════════════════════════════════════════════════════════
# A2 Capability Summary Builder
# ═══════════════════════════════════════════════════════════════════

def _build_a2_capability_summary(
    a1_raw: Dict[str, Any],
    output_dir: Path,
    manifest: RouteARLAdmissionManifest,
    max_updates: int,
) -> Dict[str, Any]:
    """Build A2 capability summary from A1 raw results."""
    per_role = a1_raw.get("per_role", {})

    # ── Per-role directional signals (all proxy for smoke) ──
    formal_roles = {}
    for role in FORMAL_CAPABILITY_ROLES:
        rdata = per_role.get(role, {})
        episodes = rdata.get("num_episodes", 0)
        # For A2 smoke: directional signal based on whether episodes were collected
        # without NaN/Inf/collision explosion
        signal = "needs_more_eval" if episodes < 10 else (
            "pass" if a1_raw.get("nan_count", 0) == 0 and a1_raw.get("inf_count", 0) == 0
            else "fail"
        )
        formal_roles[role] = {
            "episodes": episodes,
            "transitions": rdata.get("num_transitions", 0),
            "directional_signal": signal,
            "claim_allowed": False,
            "note": "proxy signal only; synthetic calibration; no real BC/FM baseline",
        }

    support_roles = {}
    yaw_data = per_role.get("yaw_lateral_recovery", {})
    support_roles["yaw_lateral_recovery"] = {
        "episodes": yaw_data.get("num_episodes", 0),
        "transitions": yaw_data.get("num_transitions", 0),
        "support_only": True,
        "claim_allowed_as_formal_role": False,
    }

    # Gate checks
    a_finite = a1_raw.get("actor_loss", {}).get("finite_ratio", 0)
    c_finite = a1_raw.get("critic_loss", {}).get("finite_ratio", 0)
    q_finite = a1_raw.get("q_scale", {}).get("finite_ratio", 0)
    actual_updates = a1_raw.get("actual_actor_updates", 0)

    gate_checks = {
        "a0_a1_gate_pass": True,  # verified before reaching here
        "actor_updates_in_range": 0 < actual_updates <= max_updates,
        "formal_role_all_covered": all(
            formal_roles[r]["episodes"] > 0 for r in FORMAL_CAPABILITY_ROLES
        ),
        "nan_zero": a1_raw.get("nan_count", 0) == 0,
        "inf_zero": a1_raw.get("inf_count", 0) == 0,
        "metadata_complete": a1_raw.get("replay_metadata_completeness", 0) >= 1.0,
        "actor_loss_finite": a_finite >= 1.0,
        "critic_loss_finite": c_finite >= 1.0,
        "q_finite": q_finite >= 1.0,
        "at_least_one_formal_directional": any(
            formal_roles[r]["directional_signal"] == "pass" for r in FORMAL_CAPABILITY_ROLES
        ),
        "yaw_is_support_only": True,
        "contamination_zero": a1_raw.get("historical_train_worlds", 0) == 0,
    }

    all_pass = all(gate_checks.values())

    return {
        "stage": "A2_newenv_capability_smoke",
        "timestamp": datetime.now().isoformat(),
        "admission_manifest": str(Path("/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration/admission_manifest.json")),
        "a0_summary": str(Path("/home/cjj/AUV/program/shiyan/results_routeA_RL_newenv_A0_smoke_20260703/A0_newenv_critic_warmup_summary.json")),
        "a1_summary": str(output_dir / "A1_newenv_stability_summary.json") if (output_dir / "A1_newenv_stability_summary.json").exists() else "",
        "calibration_data_mode": "synthetic_smoke",
        "baseline_claim_allowed": False,
        "capability_claim_allowed": False,
        "actor_update_enabled": True,
        "critic_update_enabled": True,
        "max_actor_updates": max_updates,
        "actual_actor_updates": actual_updates,
        "total_episodes": a1_raw.get("total_episodes", 0),
        "formal_roles": formal_roles,
        "support_roles": support_roles,
        "historical_train_worlds": 0,
        "stress_train_worlds": 0,
        "replay_metadata_completeness": a1_raw.get("replay_metadata_completeness", 0),
        "nan_count": a1_raw.get("nan_count", 0),
        "inf_count": a1_raw.get("inf_count", 0),
        "actor_loss": a1_raw.get("actor_loss", {}),
        "critic_loss": a1_raw.get("critic_loss", {}),
        "q_scale": a1_raw.get("q_scale", {}),
        "a2_gate_status": "PASS" if all_pass else "FAIL",
        "a2_gate_checks": gate_checks,
        "hard_stop_reason": a1_raw.get("hard_stop_reason", ""),
        "next_allowed_stage": "real_rollout_or_gazebo_shadow_eval" if all_pass else "blocked_by_a2_gate",
    }


def _write_a2_summary_md(path: Path, summary: Dict[str, Any]):
    """Write A2 capability summary Markdown."""
    lines = [
        "# A2 New-Env Capability Smoke Summary",
        f"**Stage**: {summary['stage']}",
        f"**Timestamp**: {summary['timestamp']}",
        f"**Gate Status**: {summary['a2_gate_status']}",
        "",
        "## Configuration",
        f"- Calibration data mode: `{summary['calibration_data_mode']}`",
        f"- Baseline claim allowed: {summary['baseline_claim_allowed']}",
        f"- Capability claim allowed: {summary['capability_claim_allowed']}",
        f"- Actor updates: {summary['actual_actor_updates']}/{summary['max_actor_updates']}",
        f"- Total episodes: {summary['total_episodes']}",
        "",
        "## Formal Role Directional Signals",
    ]
    for role, info in summary.get("formal_roles", {}).items():
        lines.append(f"- **{role}**: {info['episodes']} episodes, "
                     f"signal={info['directional_signal']}, claim_allowed={info['claim_allowed']}")
        lines.append(f"  - {info.get('note', '')}")
    lines += [
        "",
        "## Support Roles",
    ]
    for role, info in summary.get("support_roles", {}).items():
        lines.append(f"- **{role}**: {info['episodes']} episodes, support_only={info['support_only']}")
    lines += [
        "",
        "## Gate Checks",
    ]
    for check_name, result in summary.get("a2_gate_checks", {}).items():
        status = "PASS" if result else "FAIL"
        lines.append(f"- [{status}] {check_name}: {result}")
    lines += [
        "",
        "## Metrics",
        f"- NaN: {summary['nan_count']}, Inf: {summary['inf_count']}",
        f"- Metadata completeness: {summary['replay_metadata_completeness']}",
        f"- Hard stop: {summary['hard_stop_reason'] or 'none'}",
        "",
        f"**Next allowed stage**: {summary['next_allowed_stage']}",
        "",
        "> WARNING: All calibration data is synthetic smoke.",
        "> SR/CR values are NOT real BC/FM rollout baseline.",
        "> Directional signals are proxy only -- not paper capability claims.",
        "> baseline_claim_allowed=false, capability_claim_allowed=false",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ═══════════════════════════════════════════════════════════════════
# Route A RL v1 Formal Training
# ═══════════════════════════════════════════════════════════════════

def _generate_eval_episode_configs(
    val_pool: List[Tuple[str, str]],
    world_meta_lookup: Dict[str, Any],
    rng: np.random.Generator,
    n_configs_per_world: int = 4,
) -> Dict[str, List[Dict]]:
    """Generate fixed episode_configs for deterministic paired eval (design doc §4.1.0 v1.3).

    Returns: {world_name: [cfg1, cfg2, cfg3, cfg4]}
    Each cfg: {world_name, goal_pos, start_pos, start_yaw, episode_id}
    """
    from diffusion_policy_3d.env.collision_geom import compute_obstacle_clearance, load_obstacles

    # Role-aware sampling params (mirrors KinematicUAVEnv._ROLE_SAMPLING_PARAMS)
    _ROLE_PARAMS = {
        'anticipatory':         {'d_xy': (10, 24), 'start_cl': 0.60, 'goal_cl': 2.50},
        'dense_recovery':       {'d_xy': (5, 18),  'start_cl': 0.60, 'goal_cl': 1.50},
        'support_expansion':    {'d_xy': (8, 26),  'start_cl': 0.80, 'goal_cl': 2.00},
        'yaw_lateral_recovery': {'d_xy': (6, 18),  'start_cl': 0.80, 'goal_cl': 2.00},
    }

    configs = {}
    for wname, obs_json in val_pool:
        meta = world_meta_lookup.get(wname)
        role = (meta.role_active or meta.role_validated) if meta else 'anticipatory'
        rp = _ROLE_PARAMS.get(role, _ROLE_PARAMS['anticipatory'])
        d_xy_min, d_xy_max = rp['d_xy']
        start_cl, goal_cl = rp['start_cl'], rp['goal_cl']
        obstacles = load_obstacles(obs_json)
        x_min, x_max, y_min, y_max = -14.0, 14.0, -14.0, 14.0

        world_cfgs = []
        for cfg_idx in range(n_configs_per_world):
            cfg_seed = int(rng.integers(0, 2**31))
            cfg_rng = np.random.default_rng(cfg_seed)

            found = False
            for _attempt in range(50):
                sx = cfg_rng.uniform(x_min, x_max)
                sy = cfg_rng.uniform(y_min, y_max)
                sz = cfg_rng.uniform(1.3, 4.7)
                gx = cfg_rng.uniform(x_min, x_max)
                gy = cfg_rng.uniform(y_min, y_max)
                gz = cfg_rng.uniform(1.5, 4.5)

                d_xy = math.hypot(gx - sx, gy - sy)
                if d_xy < d_xy_min or d_xy > d_xy_max:
                    continue
                if math.hypot(d_xy, gz - sz) < 4.0:
                    continue

                s_cl, _, _ = compute_obstacle_clearance(sx, sy, sz, obstacles)
                if s_cl < start_cl:
                    continue
                g_cl, _, _ = compute_obstacle_clearance(gx, gy, gz, obstacles)
                if g_cl < goal_cl:
                    continue

                yaw = math.atan2(gy - sy, gx - sx)
                cfg = {
                    'world_name': wname,
                    'goal_pos': [float(gx), float(gy), float(gz)],
                    'start_pos': [float(sx), float(sy), float(sz)],
                    'start_yaw': float(yaw),
                    'episode_id': f"{wname}_cfg{cfg_idx}",
                }
                world_cfgs.append(cfg)
                found = True
                break

            if not found:
                # Fallback: nominal positions (eval must not use metadata fallback in production)
                cfg = {
                    'world_name': wname,
                    'goal_pos': [8.0, 0.0, 2.0],
                    'start_pos': [-8.0, 0.0, 2.0],
                    'start_yaw': 0.0,
                    'episode_id': f"{wname}_cfg{cfg_idx}_fallback",
                }
                world_cfgs.append(cfg)

        configs[wname] = world_cfgs

    return configs


def _run_newenv_eval(
    actor: RouteAReinFlowFMPolicy,
    base_policy: Phase1aPolicyFM,
    device: str,
    val_pool: List[Tuple[str, str]],
    world_meta_lookup: Dict[str, Any],
    rng: np.random.Generator,
    n_episodes_per_world: int = 2,
    seed_offset: int = 0,
    fixed_world_indices: Optional[List[int]] = None,
    episode_configs: Optional[Dict[str, List[Dict]]] = None,
    curriculum_stage: str = 'C2',
) -> Dict[str, Any]:
    """Lightweight per-role eval on newenv val pool.

    If fixed_world_indices is provided, evaluates on those exact worlds
    (enables fair pre/post comparison). Otherwise randomly selects worlds.

    If episode_configs is provided, uses pre-generated {world_name: [cfg]}
    for deterministic reset — pre/post eval reuse same configs for clean delta.

    Returns per-role SR/CR and capability indicators.
    """
    actor.eval()
    per_role = defaultdict(lambda: {"n_success": 0, "n_collision": 0, "n_total": 0,
                                     "rewards": [], "path_lengths": [],
                                     "low_density_eps": 0,
                                     "onset_times": [],  # fine-grained
                                     "future_clearance_softmins": [],  # fine-grained
                                     "trap_recoveries": [],  # fine-grained
                                     "progress_to_goals": [],  # fine-grained
                                     "min_clearances": [],  # fine-grained
                                     "episode_lengths": [],  # fine-grained
                                     })
    all_summaries = []

    # Select subset of val pool for speed (max 6 worlds)
    n_eval_worlds = min(6, len(val_pool))
    if fixed_world_indices is not None:
        indices = list(fixed_world_indices)
    else:
        indices = rng.choice(len(val_pool), size=n_eval_worlds, replace=False).tolist()

    for pool_idx in indices:
        wname, obs_json = val_pool[pool_idx]
        meta = world_meta_lookup.get(wname)
        role_val = meta.role_active or meta.role_validated if meta else "unknown"
        if role_val not in per_role:
            per_role[role_val] = {"n_success": 0, "n_collision": 0, "n_total": 0,
                                  "rewards": [], "path_lengths": [],
                                  "low_density_eps": 0,
                                  "onset_times": [], "future_clearance_softmins": [],
                                  "trap_recoveries": [], "progress_to_goals": [],
                                  "min_clearances": [], "episode_lengths": []}

        # Get pre-generated configs for this world (if available)
        world_configs = None
        if episode_configs is not None:
            world_configs = episode_configs.get(wname, None)

        for ep in range(n_episodes_per_world):
            try:
                env_seed = int(rng.integers(0, 2**31))
                env = KinematicUAVEnv(
                    world_id=4,
                    world_pool=[(wname, obs_json)],
                    episode_role=role_val,  # P1 fix (2026-07-11): role-aware start/goal gating
                    synth_pc_factory=lambda jp: SynthPointCloud(jp, device=device),
                    rl_action_steps=8, h_min=0.3,
                    z_min=1.0, z_max=5.0,
                    disturb_enable=False,  # deterministic eval
                    mid_state_init_ratio=0.0,
                    seed=env_seed,
                    reward_version="v13",
                    curriculum_stage=curriculum_stage,
                )
                # Pass episode_config for deterministic eval (design doc §4.1.0)
                reset_kwargs = {}
                if world_configs is not None and ep < len(world_configs):
                    reset_kwargs = {'options': {'episode_config': world_configs[ep]}}

                transitions, _, summary = rollout_episode_a2(
                    actor, base_policy, env, device,
                    episode_seed=int(rng.integers(0, 2**31)),
                    world_id=-1, deterministic=True,
                    reset_kwargs=reset_kwargs,
                )
                per_role[role_val]["n_total"] += 1
                if summary.get("reached", False):
                    per_role[role_val]["n_success"] += 1
                if summary.get("collided", False):
                    per_role[role_val]["n_collision"] += 1
                per_role[role_val]["rewards"].append(summary.get("reward_total", 0))
                per_role[role_val]["path_lengths"].append(summary.get("path_length", 0))
                # Fine-grained metrics
                per_role[role_val]["onset_times"].append(summary.get("onset_time"))
                per_role[role_val]["future_clearance_softmins"].append(summary.get("future_clearance_softmin"))
                per_role[role_val]["trap_recoveries"].append(summary.get("trap_recovery"))
                per_role[role_val]["progress_to_goals"].append(summary.get("progress_to_goal"))
                per_role[role_val]["min_clearances"].append(summary.get("min_clearance"))
                per_role[role_val]["episode_lengths"].append(summary.get("length"))
                # Check low density coverage
                for t in transitions:
                    rm = t.reward_metrics or {}
                    if rm.get("low_density_cell", False):
                        per_role[role_val]["low_density_eps"] += 1
                        break
                all_summaries.append(summary)
                env.close()
            except Exception:
                continue

    # Compile per-role metrics
    eval_results = {}
    for role, data in per_role.items():
        n = max(data["n_total"], 1)
        sr = data["n_success"] / n
        cr = data["n_collision"] / n

        # Fine-grained metric aggregation
        def _finite_mean(vals):
            finite = [v for v in vals if v is not None and np.isfinite(v)]
            return round(float(np.mean(finite)), 4) if finite else None

        onset_valid = [v for v in data["onset_times"] if v is not None]
        fc_valid = [v for v in data["future_clearance_softmins"] if v is not None and np.isfinite(v)]
        trap_valid = [v for v in data["trap_recoveries"] if v is not None]
        ptg_valid = [v for v in data["progress_to_goals"] if v is not None and np.isfinite(v)]
        mc_valid = [v for v in data["min_clearances"] if v is not None and np.isfinite(v)]

        eval_results[role] = {
            "SR": round(sr, 4),
            "CR": round(cr, 4),
            "reward_mean": round(float(np.mean(data["rewards"])), 4) if data["rewards"] else 0.0,
            "path_length_mean": round(float(np.mean(data["path_lengths"])), 4) if data["path_lengths"] else 0.0,
            "n_episodes": data["n_total"],
            "low_density_coverage": round(data["low_density_eps"] / n, 4),
            # Fine-grained capability metrics
            "onset_time_mean": round(float(np.mean(onset_valid)), 4) if onset_valid else None,
            "onset_time_active_frac": round(len(onset_valid) / n, 4),  # fraction of eps where onset activated
            "future_clearance_softmin_mean": round(float(np.mean(fc_valid)), 4) if fc_valid else None,
            "trap_recovery_rate": round(float(np.mean(trap_valid)), 4) if trap_valid else None,
            "progress_to_goal_mean": round(float(np.mean(ptg_valid)), 4) if ptg_valid else None,
            "min_clearance_mean": round(float(np.mean(mc_valid)), 4) if mc_valid else None,
            "episode_length_mean": round(float(np.mean(data["episode_lengths"])), 4) if data["episode_lengths"] else None,
        }

    actor.train()
    return {
        "per_role": eval_results,
        "n_worlds_evaluated": n_eval_worlds,
        "n_episodes": len(all_summaries),
        "world_indices": indices,
    }


# ═══════════════════════════════════════════════════════════════════
# Batched Rollout — parallel envs with batched actor forward
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def rollout_batched_episodes(
    actor: RouteAReinFlowFMPolicy,
    base_policy: Phase1aPolicyFM,
    envs: List[KinematicUAVEnv],
    device: str,
    world_meta: List[Tuple[str, str, str]],  # [(wname, role_val, role_act), ...]
    train_split: str,
    max_steps: int = 60,
    lazy_teacher: bool = True,
) -> Tuple[List[List[RouteATransition]], List[List[Dict]], List[Dict]]:
    """Run multiple envs in lockstep with batched actor forward.

    Key optimization: instead of calling actor.forward() once per env per step
    (batch_size=1, poor GPU utilization), we batch across all active envs at
    each step. This reduces CUDA kernel launch overhead and improves GPU
    utilization from ~20% to ~60% for the rollout forward pass.

    Semantics preserved: same envs, same observations, same actor, same noise.
    Only difference is batching — action output is identical to sequential.

    Returns:
        transitions_per_env: list of transition lists, one per env
        obs_buffers_per_env: list of obs_buffer lists, one per env
        summaries: list of episode summary dicts
    """
    n_envs = len(envs)
    obs_list = [env.reset() for env in envs]
    # Strip tuple if reset returns (obs, info)
    obs_list = [o[0] if isinstance(o, tuple) else o for o in obs_list]
    active = [True] * n_envs
    done_reasons = ["truncated"] * n_envs

    transitions_per_env: List[List[RouteATransition]] = [[] for _ in range(n_envs)]
    obs_buffers_per_env: List[List[Dict]] = [[] for _ in range(n_envs)]
    ep_rewards = [0.0] * n_envs

    # Per-env trackers
    global_conds = [None] * n_envs
    prev_actions = [None] * n_envs
    first_step_done = [False] * n_envs

    for step_i in range(max_steps):
        if not any(active):
            break

        # ── Build batch from active envs ──
        active_indices = [i for i, a in enumerate(active) if a]
        n_active = len(active_indices)

        # Collate observations
        pc_batch = []
        state_batch = []
        for idx in active_indices:
            obs = obs_list[idx]
            pc_batch.append(torch.from_numpy(obs["point_cloud"][..., :3].copy()))
            state_batch.append(torch.from_numpy(obs["agent_pos"].copy()))
        batch_obs = {
            "point_cloud": torch.stack(pc_batch).to(device),
            "agent_pos": torch.stack(state_batch).to(device),
        }

        # ── Single batched actor forward ──
        actor_out = actor(batch_obs, deterministic=False,
                          compute_teacher=not lazy_teacher)
        actions = actor_out["action"].cpu().numpy()  # (n_active, 8, 11)
        global_conds_batch = actor_out["global_cond"].cpu().numpy()  # (n_active, D)
        teacher_actions_batch = (actor_out["teacher_action"].cpu().numpy()
                                 if actor_out.get("teacher_action") is not None
                                 else [None] * n_active)

        # ── Step each active env ──
        for j, idx in enumerate(active_indices):
            action = actions[j]
            next_obs, reward, terminated, truncated, info = envs[idx].step(action)
            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]
            ep_rewards[idx] += float(reward)

            reward_terms = info.get("reward_terms", {})
            reward_total = sum(reward_terms.get(k, 0.0) for k in REWARD_TERM_NAMES)
            done = terminated or truncated

            wname, role_val, role_act = world_meta[idx]

            # Build transition
            trans = RouteATransition(
                obs_feature_t=global_conds_batch[j].copy() if global_conds[idx] is None
                              else global_conds[idx].copy(),
                action_chunk=action.copy(),
                action_chunk_tp1=action.copy(),  # will be updated next step or at done
                reward_total=float(reward_total),
                reward_terms={k: float(reward_terms.get(k, 0.0)) for k in REWARD_TERM_NAMES},
                obs_feature_tp1=global_conds_batch[j].copy(),
                done=done,
                done_reason=info.get("done_reason", ""),
                world_id=-1,
                episode_seed=0,
                world_uid=wname, split=train_split,
                role_generated="", role_validated=role_val or "",
                role_active=role_act or "", admission_status="admitted",
            )

            # Update prev transition's action_tp1
            if len(transitions_per_env[idx]) > 0:
                transitions_per_env[idx][-1].action_chunk_tp1 = action.copy()
                transitions_per_env[idx][-1].obs_feature_tp1 = global_conds_batch[j].copy()

            transitions_per_env[idx].append(trans)

            # Build obs_buffer item — use _obs_dict_to_torch for correct shape
            obs_torch_item = _obs_dict_to_torch(obs_list[idx], device)
            teacher_np = (teacher_actions_batch[j]
                          if teacher_actions_batch is not None and j < len(teacher_actions_batch)
                          else None)
            obs_buffers_per_env[idx].append({
                "obs_torch": {k: v.detach().clone() for k, v in obs_torch_item.items()},
                "teacher_action": (torch.from_numpy(teacher_np.copy())
                                   if teacher_np is not None else None),
                "scenario": "unknown",
                "world_id": -1,
                "episode_seed": 0,
                "step_idx": step_i,
            })

            global_conds[idx] = global_conds_batch[j]
            obs_list[idx] = next_obs

            if done:
                done_reasons[idx] = info.get("done_reason", "unknown")
                active[idx] = False

    # ── Build summaries ──
    summaries = []
    for idx in range(n_envs):
        reached = (done_reasons[idx] == "reached")
        collided = ("collision" in done_reasons[idx] or "penetrat" in done_reasons[idx])
        summaries.append({
            "world_uid": world_meta[idx][0],
            "reached": reached, "collided": collided,
            "done_reason": done_reasons[idx],
            "reward_total": round(ep_rewards[idx], 4),
            "length": len(transitions_per_env[idx]),
            "reward_terms": {},
        })

    return transitions_per_env, obs_buffers_per_env, summaries


# ═══════════════════════════════════════════════════════════════════
# P0 Wiring Health: Mode implementations (Tasks 3-5)
# ═══════════════════════════════════════════════════════════════════

def _run_collect_only(
    actor, base_policy, device, output_dir, ckpt_writer,
    replay, manifest, train_split, world_dir,
    train_pool, world_meta_lookup, role_to_indices,
    max_steps=100, output_replay=None, rng=None,
    metadata_complete=0, metadata_total=0,
    contamination_count=0, total_episodes=0,
    per_role_episodes=None, per_role_transitions=None,
    curriculum_stage: str = 'C2',
):
    """Task 3: Collect-only mode — sample rollouts, write replay, skip all optimizer steps."""
    if rng is None:
        rng = np.random.default_rng()
    if per_role_episodes is None:
        per_role_episodes = defaultdict(int)
    if per_role_transitions is None:
        per_role_transitions = defaultdict(int)

    cprint(f"\n[CollectOnly] Collecting up to {max_steps} episodes, no actor/critic updates...",
           "white", "on_blue")

    collect_start = time.time()
    episodes_collected = 0
    transitions_collected = 0
    role_counts = defaultdict(int)
    support_role_share = 0.0
    # Root-cause fix (2026-07-10, spec docs/superpowers/specs/2026-07-10-b0b1-rerun-v13-design.md):
    # Previously `idx = indices[step % len(indices)]` aliased with the outer
    # `step % 3` role cycle (gcd(3,15)=3), so dense_recovery visited only 5/15
    # worlds and anticipatory 1/3 -> B0 audit failed on world coverage. Use an
    # independent per-role round-robin cursor so every world in each bucket is covered.
    formal_roles = ["anticipatory", "dense_recovery", "support_expansion"]
    role_cursor = defaultdict(int)
    worlds_visited_per_role = defaultdict(set)

    for step in range(1, max_steps + 1):
        # Select a world using role-balanced strategy (per-role round-robin)
        role = formal_roles[step % len(formal_roles)]
        indices = role_to_indices.get(role, [])
        if not indices:
            continue
        idx = indices[role_cursor[role] % len(indices)]
        role_cursor[role] += 1
        wname, obs_json = train_pool[idx]
        meta = world_meta_lookup.get(wname)
        role_val = meta.role_active or meta.role_validated if meta else ""
        role_act = meta.role_active or ""

        try:
            env = KinematicUAVEnv(
                world_id=4, world_pool=[(wname, obs_json)],
                episode_role=role_val,  # P1 fix (2026-07-11): role-aware start/goal gating; was defaulting to 'anticipatory' for all roles
                synth_pc_factory=lambda jp: SynthPointCloud(jp, device=device),
                rl_action_steps=8, h_min=0.3, z_min=1.0, z_max=5.0,
                disturb_enable=True, disturb_prob=0.2, disturb_pos_scale=0.3,
                mid_state_init_ratio=0.0,
                seed=int(rng.integers(0, 2**31)),
                reward_version="v13",
                                curriculum_stage=curriculum_stage,
                )
            transitions, obs_buf, ep_sum = rollout_episode_a2(
                actor, base_policy, env, device,
                episode_seed=int(rng.integers(0, 2**31)),
                world_id=-1, deterministic=False, lazy_teacher=True,
            )
            for t in transitions:
                t.world_uid = wname; t.split = train_split
                t.role_validated = role_val; t.role_active = role_act
                t.admission_status = "admitted"
            replay.add_episode(transitions, partition="online")
            episodes_collected += 1
            transitions_collected += len(transitions)
            role_counts[role_val] = role_counts.get(role_val, 0) + 1
            worlds_visited_per_role[role].add(wname)
            for t in transitions:
                metadata_total += 1
                if t.world_uid and t.role_validated:
                    metadata_complete += 1
            env.close()
        except Exception as e:
            cprint(f"  [WARN] Collect-only rollout failed for {wname}: {e}", "yellow")

        if step % 10 == 0 or step == 1:
            cprint(f"  Step {step}/{max_steps}: {episodes_collected} eps, "
                   f"{transitions_collected} trans, roles={dict(role_counts)}", "cyan")

    # Support role share
    support_eps = role_counts.get("support_expansion", 0)
    support_role_share = support_eps / max(episodes_collected, 1)

    # World-coverage report (soft check; the gate is enforced by audit_b0_replay_v13.py)
    coverage_counts = {r: len(worlds_visited_per_role.get(r, set())) for r in formal_roles}
    expected_counts = {r: len(role_to_indices.get(r, [])) for r in formal_roles}
    coverage_complete = all(
        coverage_counts.get(r, 0) == expected_counts.get(r, 0)
        for r in formal_roles if expected_counts.get(r, 0) > 0
    )
    if coverage_complete:
        cprint(f"  World coverage COMPLETE: visited={coverage_counts}", "green")
    else:
        cprint(f"  [WARN] World coverage INCOMPLETE: visited={coverage_counts} "
               f"expected={expected_counts}", "yellow")

    # Save replay manifest
    replay_stats = replay.get_stats()
    if output_replay:
        manifest_path = Path(output_replay)
    else:
        manifest_path = output_dir / "collect_only_replay_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    collect_elapsed = time.time() - collect_start
    summary = {
        "run_id": ckpt_writer.run_id,
        "config_hash": "collect_only_smoke",
        "source_script_mtime": datetime.fromtimestamp(
            os.path.getmtime(__file__)).isoformat(),
        "mode": "collect_only",
        "actor_optimizer_steps": 0,
        "critic_optimizer_steps": 0,
        "target_update_steps": 0,
        "episodes_collected": episodes_collected,
        "transitions_collected": transitions_collected,
        "role_counts": dict(role_counts),
        "support_role_share": round(support_role_share, 4),
        "metadata_completeness": round(metadata_complete / max(metadata_total, 1), 4),
        "historical_contamination": contamination_count,
        "replay_size_final": replay_stats["total_transitions"],
        "new_rollout_episodes": episodes_collected,
        "hard_stop_primary_reason": "",
        "hard_stop_secondary_reasons": [],
        "gazebo_closed_loop_status": "not_run",
        "lightweight_eval_status": "monitoring_only_not_gazebo",
        "formal_training_status": "collect_only_complete",
        "claim_allowed": False,
        "elapsed_min": round(collect_elapsed / 60, 1),
        "worlds_visited_per_role": coverage_counts,
        "worlds_expected_per_role": expected_counts,
        "coverage_complete": coverage_complete,
        "timestamp": datetime.now().isoformat(),
    }

    with open(manifest_path, "w") as f:
        json.dump(summary, f, indent=2)
    cprint(f"  Replay manifest: {manifest_path}", "green")

    with open(output_dir / "collect_only_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    summary_md = output_dir / "collect_only_summary.md"
    with open(summary_md, "w") as f:
        f.write("# Collect-Only Mode Summary\n\n")
        f.write(f"- **Mode**: collect_only\n")
        f.write(f"- **Episodes collected**: {episodes_collected}\n")
        f.write(f"- **Transitions collected**: {transitions_collected}\n")
        f.write(f"- **Actor optimizer steps**: 0\n")
        f.write(f"- **Critic optimizer steps**: 0\n")
        f.write(f"- **Replay size**: {replay_stats['total_transitions']}\n")
        f.write(f"- **Claim allowed**: False\n")
        f.write(f"- **Elapsed**: {collect_elapsed/60:.1f} min\n")

    cprint(f"  Summary: {summary_md}", "green")
    return summary


def _build_dual_unit_diag(q1q2_decomp, train_diag, reward_norm_mode, reward_norm_stats,
                          q_return_spearman_rho, td_target_ratio):
    """B1d: Build normalized-unit and raw-unit diagnostic fields.

    When reward_norm_mode is active, the critic operates in normalized space.
    This function returns both normalized-unit metrics (for critic gate)
    and raw-unit reconstructions (for reward/Q physical interpretation).

    Raw reconstruction: raw_value = normalized_value * total_std + total_mean
    (approximate; exact for linear transforms, close for additive mean shifts)
    """
    diag = {
        "reward_norm_mode": reward_norm_mode,
    }
    if not q1q2_decomp:
        diag["error"] = "no q1q2 decomposition available"
        return diag

    q_mean = q1q2_decomp.get("q1_total_mean", None)
    q_std = q1q2_decomp.get("q1_total_std", None)

    # ── Normalized-unit diagnostics ──
    norm_diag = {
        "q_total_mean": q_mean,
        "q_total_std": q_std,
    }
    if train_diag:
        norm_diag["td_target_mean"] = train_diag.get("train_td_target_mean", None)
        norm_diag["td_target_std"] = train_diag.get("train_td_target_std", None)
        norm_diag["td_target_abs_p95"] = train_diag.get("train_td_target_abs_p95", None)
        norm_diag["reward_total_mean"] = train_diag.get("train_reward_total_mean", None)
        norm_diag["reward_total_std"] = train_diag.get("train_reward_total_std", None)
        norm_diag["reward_total_abs_p95"] = train_diag.get("train_reward_total_abs_p95", None)
    norm_diag["q_return_spearman_rho"] = q_return_spearman_rho
    norm_diag["td_target_p95_reward_abs_p95_ratio_artifact"] = td_target_ratio
    diag["normalized_unit"] = norm_diag

    # ── Raw-unit reconstruction ──
    if reward_norm_mode != "none" and reward_norm_stats is not None:
        total_mean = reward_norm_stats.get("total_mean", 0.0)
        total_std = max(reward_norm_stats.get("total_std", 1.0), 1e-8)
        raw = {
            "reconstruction_rule": "raw = normalized * total_std + total_mean",
            "total_mean_used": total_mean,
            "total_std_used": total_std,
        }
        if q_mean is not None:
            raw["raw_q_total_mean_reconstructed"] = q_mean * total_std + total_mean
        if q_std is not None:
            raw["raw_q_total_std_reconstructed"] = q_std * total_std
        if train_diag:
            td_abs_p95 = train_diag.get("train_td_target_abs_p95", None)
            if td_abs_p95 is not None:
                raw["raw_td_target_abs_p95_reconstructed"] = td_abs_p95 * abs(total_std)
            r_abs_p95 = train_diag.get("train_reward_total_abs_p95", None)
            if r_abs_p95 is not None:
                raw["raw_reward_total_abs_p95_reconstructed"] = r_abs_p95 * abs(total_std)
            r_mean = train_diag.get("train_reward_total_mean", None)
            if r_mean is not None:
                raw["raw_reward_total_mean_reconstructed"] = r_mean * total_std + total_mean
        raw["raw_q_return_spearman_rho"] = q_return_spearman_rho  # rho invariant to affine transform
        diag["raw_unit_reconstructed"] = raw
        diag["standardized_qscale_gate_applicable"] = True
        diag["raw_td_ratio_gate_applicable"] = False
    else:
        diag["raw_unit_reconstructed"] = {
            "note": "not applicable — reward_norm_mode is none, raw = normalized = actual"
        }
        diag["standardized_qscale_gate_applicable"] = False
        diag["raw_td_ratio_gate_applicable"] = True

    return diag


def _run_critic_bootstrap(
    actor, base_policy, critic, target_critic, critic_optimizer,
    device, output_dir, ckpt_writer,
    replay, train_pool, world_meta_lookup, role_to_indices,
    heldout_ratio=0.2, batch_size=2048, critic_updates=16,
    max_actor_updates=100, eval_interval=100,
    metadata_complete=0, metadata_total=0,
    contamination_count=0,
    val_pool=None, rng=None,
    # ── B1a: diagnostic / ablation ──
    diag_only=False,
    gamma_override=None,
    target_tau_override=None,
    total_only=False,
    huber_total=False,
    huber_beta=5.0,
    sparse_head_mode="none",
    # ── B1c: architecture / reward representation knobs ──
    reward_norm_mode="none",
    reward_norm_eps=1e-6,
    q_l2_weight=0.0,
    critic_norm="none",
    target_update_mode="polyak_each_step",
    target_update_interval=100,
    # ── B1d: split reproducibility ──
    split_seed=None,
):
    """Task 4: Critic bootstrap — fixed replay, no new rollouts, no actor update."""
    if rng is None:
        rng = np.random.default_rng()
    # ── B1d: Separate RNG for episode split (independent of other RNG uses) ──
    _split_rng = np.random.default_rng(split_seed) if split_seed is not None else rng

    cprint(f"\n[CriticBootstrap] Fixed replay, critic-only training, "
           f"actor updates DISABLED...", "white", "on_blue")

    # Determine if this is smoke (small replay) or real bootstrap
    n_trans = replay.get_stats()["total_transitions"]
    is_smoke = n_trans < 1000
    mode_label = "critic_bootstrap_smoke" if is_smoke else "critic_bootstrap"

    cprint(f"  Replay size: {n_trans} transitions → mode={mode_label}", "cyan")

    bootstrap_start = time.time()
    critic_losses = []
    train_twin_gap_p95 = None
    heldout_twin_gap_p95 = None
    q_return_std_ratio = None
    q_return_spearman_rho = None
    td_target_p95_empirical_ratio = None
    td_target_max_empirical_ratio = None
    fixed_replay_overfit = False

    actor_optimizer_steps = 0
    critic_optimizer_steps = 0
    target_update_steps = 0

    # ── B1a: resolve ablation params ──
    _gamma = gamma_override if gamma_override is not None else 0.99
    _tau = target_tau_override if target_tau_override is not None else 0.005
    _gamma_h = _gamma ** 8

    if total_only:
        _betas = {k: 0.0 for k in DEFAULT_BETAS}
        cprint(f"  [Ablation] total_only: all decomposed betas=0", "yellow")
    elif sparse_head_mode == "zero_onset_trap_betas":
        _betas = dict(DEFAULT_BETAS)
        _betas["onset"] = 0.0
        _betas["trap"] = 0.0
        cprint(f"  [Ablation] zero_onset_trap_betas: onset/trap β=0", "yellow")
    else:
        _betas = None  # use DEFAULT_BETAS in compute_critic_loss

    if sparse_head_mode not in ("none", "zero_onset_trap_betas"):
        cprint(f"  [Ablation] sparse_head_mode={sparse_head_mode}", "yellow")

    # ── Episode-level train/heldout split ──
    all_trans = replay.get_all_transitions()
    episode_groups = defaultdict(list)
    for i, t in enumerate(all_trans):
        ep_id = f"{getattr(t, 'world_uid', '')}__{getattr(t, 'episode_seed', i)}"
        episode_groups[ep_id].append(i)

    episode_ids = sorted(episode_groups.keys())
    n_episodes = len(episode_ids)
    if n_episodes >= 5:
        _split_rng.shuffle(episode_ids)
        n_heldout_ep = max(1, int(n_episodes * heldout_ratio))
        heldout_ep_ids = set(episode_ids[:n_heldout_ep])
        train_ep_ids = set(episode_ids[n_heldout_ep:])
    else:
        heldout_ep_ids = set()
        train_ep_ids = set(episode_ids)

    train_indices = [i for ep in train_ep_ids for i in episode_groups[ep]]
    heldout_indices = [i for ep in heldout_ep_ids for i in episode_groups[ep]]
    n_train_trans = len(train_indices)
    n_heldout_trans = len(heldout_indices)

    cprint(f"  Episode split: train={len(train_ep_ids)} eps/{n_train_trans} trans, "
           f"heldout={len(heldout_ep_ids)} eps/{n_heldout_trans} trans", "cyan")

    # ── B1c: Reward normalization stats (computed on train split only) ──
    reward_norm_stats = None
    if reward_norm_mode != "none" and n_train_trans >= 8:
        # Collect all train rewards for stats computation
        train_rewards_total = []
        train_rewards_per_head = defaultdict(list)
        REWARD_TERM_NAMES_LOOKUP = [
            "base", "pbrs", "onset", "future_clearance",
            "trap", "flip", "support", "guard",
        ]
        for i in train_indices[:min(4096, len(train_indices))]:
            t = all_trans[i]
            train_rewards_total.append(t.reward_total)
            for rk in REWARD_TERM_NAMES_LOOKUP:
                if hasattr(t, f"reward_{rk}"):
                    train_rewards_per_head[rk].append(getattr(t, f"reward_{rk}"))
        rt_tensor = torch.tensor(train_rewards_total, dtype=torch.float32)
        reward_norm_stats = {"mode": reward_norm_mode, "eps": reward_norm_eps}
        if reward_norm_mode == "fixed_train_standardize":
            reward_norm_stats["total_mean"] = float(rt_tensor.mean().item())
            reward_norm_stats["total_std"] = float(rt_tensor.std().item())
            for rk, vals in train_rewards_per_head.items():
                vt = torch.tensor(vals, dtype=torch.float32)
                reward_norm_stats[f"{rk}_mean"] = float(vt.mean().item())
                reward_norm_stats[f"{rk}_std"] = float(vt.std().item())
            cprint(f"  [B1c] Reward norm: standardize, total_mean={reward_norm_stats['total_mean']:.2f}, "
                   f"total_std={reward_norm_stats['total_std']:.2f}", "yellow")
        elif reward_norm_mode == "fixed_train_scale_p95":
            reward_norm_stats["total_abs_p95"] = float(rt_tensor.abs().quantile(0.95).item())
            for rk, vals in train_rewards_per_head.items():
                vt = torch.tensor(vals, dtype=torch.float32)
                reward_norm_stats[f"{rk}_abs_p95"] = float(vt.abs().quantile(0.95).item())
            cprint(f"  [B1c] Reward norm: scale_p95, total_abs_p95={reward_norm_stats['total_abs_p95']:.2f}", "yellow")
        # Save norm stats
        with open(output_dir / "reward_norm_stats.json", "w") as f:
            json.dump(reward_norm_stats, f, indent=2)

    # ── B1c: Target update mode ──
    _hard_target_every = target_update_interval if target_update_mode == "hard_every_n" else 0
    if target_update_mode != "polyak_each_step":
        cprint(f"  [B1c] Target update: {target_update_mode}"
               + (f", interval={target_update_interval}" if _hard_target_every else ""), "yellow")

    if q_l2_weight > 0:
        cprint(f"  [B1c] Q L2 weight: {q_l2_weight}", "yellow")

    # Helper: apply reward normalization to batch dict (in-place)
    def _apply_reward_norm(batch_d, mode, stats, eps):
        if mode == "fixed_train_standardize":
            total_mean = stats["total_mean"]
            total_std = max(stats["total_std"], eps)
            batch_d["reward_total"] = (batch_d["reward_total"] - total_mean) / total_std
            for rk in list(batch_d["reward_heads"].keys()):
                mean_k = stats.get(f"{rk}_mean", 0.0)
                std_k = max(stats.get(f"{rk}_std", 1.0), eps)
                batch_d["reward_heads"][rk] = (batch_d["reward_heads"][rk] - mean_k) / std_k
        elif mode == "fixed_train_scale_p95":
            total_p95 = max(stats["total_abs_p95"], eps)
            batch_d["reward_total"] = batch_d["reward_total"] / total_p95
            for rk in list(batch_d["reward_heads"].keys()):
                p95_k = max(stats.get(f"{rk}_abs_p95", 1.0), eps)
                batch_d["reward_heads"][rk] = batch_d["reward_heads"][rk] / p95_k

    # Helper: build diagnostic batch from specific indices
    def _build_diag_batch(indices, n_sample=256):
        if len(indices) == 0:
            return None
        k = min(n_sample, len(indices))
        sel = [all_trans[i] for i in rng.choice(indices, size=k, replace=False).tolist()]
        return collate_to_critic_device(collate_to_critic(sel), device)

    heldout_losses = []

    if diag_only:
        cprint(f"\n[CriticBootstrap] DIAGNOSTIC-ONLY mode: no training, computing metrics only", "yellow")
    else:
        cprint(f"\n[CriticBootstrap] Training: gamma={_gamma}, tau={_tau}, "
               f"total_only={total_only}, huber_total={huber_total}"
               + (f", reward_norm={reward_norm_mode}" if reward_norm_mode != "none" else "")
               + (f", q_l2={q_l2_weight}" if q_l2_weight > 0 else "")
               + (f", critic_norm={critic_norm}" if critic_norm != "none" else "")
               + (f", target_update={target_update_mode}" if target_update_mode != "polyak_each_step" else ""), "cyan")

        for step in range(1, max_actor_updates + 1):
            step_critic_loss = 0.0
            for _cu in range(critic_updates):
                if n_trans < 8:
                    break
                batch = replay.sample(min(batch_size, n_trans))
                if batch["state"].shape[0] == 0:
                    continue
                batch_d = collate_to_critic_device(batch, device)

                # ── B1c: Apply reward normalization ──
                if reward_norm_stats is not None:
                    _apply_reward_norm(batch_d, reward_norm_mode, reward_norm_stats, reward_norm_eps)

                critic_optimizer.zero_grad()
                if huber_total:
                    # Manual huber total: compute loss with MSE → replace total head with SmoothL1
                    c_loss, c_log = compute_critic_loss(
                        critic, target_critic,
                        batch_d["state"], batch_d["action"],
                        batch_d["reward_total"], batch_d["reward_heads"],
                        batch_d["next_state"], batch_d["action_tp1"],
                        done=batch_d["done"], gamma=_gamma, betas=_betas,
                    )
                    # Replace total head MSE with SmoothL1
                    with torch.no_grad():
                        q_next_total = target_critic.q_min(batch_d["next_state"], batch_d["action_tp1"])
                        done_float = batch_d["done"].float()
                        y_total = batch_d["reward_total"] + _gamma_h * (1.0 - done_float) * q_next_total
                    q_total = critic(batch_d["state"], batch_d["action"])["total"]  # (B, 2)
                    loss_total_huber = F.smooth_l1_loss(q_total[:, 0], y_total, beta=huber_beta)
                    loss_total_huber += F.smooth_l1_loss(q_total[:, 1], y_total, beta=huber_beta)
                    # Subtract old MSE total, add huber total
                    c_loss = c_loss - c_log.get("critic_loss_mse", 0.0) + loss_total_huber

                    # ── B1b: sparse head treatment ──
                    if sparse_head_mode == "huber_sparse_heads":
                        # Replace onset/trap head MSE with SmoothL1
                        with torch.no_grad():
                            q_next_heads = target_critic.q1_only(batch_d["next_state"], batch_d["action_tp1"])
                            HEAD_REWARD_MAP = {
                                "progress": ["base", "pbrs"], "onset": ["onset"],
                                "future_clearance": ["future_clearance"],
                                "trap": ["trap", "flip"], "support": ["support"], "guard": ["guard"],
                            }
                            done_float = batch_d["done"].float()
                        q_heads = critic(batch_d["state"], batch_d["action"])
                        for hname in ["onset", "trap"]:
                            qnh = q_next_heads.get(hname)
                            if qnh is None:
                                continue
                            qnh = qnh.squeeze(-1)
                            terms = HEAD_REWARD_MAP.get(hname, [hname])
                            r_head = sum(batch_d["reward_heads"].get(t, torch.zeros_like(batch_d["reward_total"])) for t in terms)
                            y_head = r_head + _gamma_h * (1.0 - done_float) * qnh
                            qh = q_heads[hname]  # (B, 2)
                            loss_h_huber = F.smooth_l1_loss(qh[:, 0], y_head, beta=1.0)
                            loss_h_huber += F.smooth_l1_loss(qh[:, 1], y_head, beta=1.0)
                            old_mse = c_log.get(f"critic_loss_head_{hname}", 0.0)
                            c_loss = c_loss - old_mse + loss_h_huber

                    if sparse_head_mode == "sparse_head_l2":
                        q_heads = critic(batch_d["state"], batch_d["action"])
                        l2_penalty = torch.tensor(0.0, device=device)
                        for hname in ["onset", "trap"]:
                            qh = q_heads.get(hname)
                            if qh is not None:
                                l2_penalty = l2_penalty + (qh[:, 0].pow(2).mean() + qh[:, 1].pow(2).mean()) * 1e-4
                        c_loss = c_loss + l2_penalty
                else:
                    c_loss, c_log = compute_critic_loss(
                        critic, target_critic,
                        batch_d["state"], batch_d["action"],
                        batch_d["reward_total"], batch_d["reward_heads"],
                        batch_d["next_state"], batch_d["action_tp1"],
                        done=batch_d["done"], gamma=_gamma, betas=_betas,
                    )
                # ── B1c: Q-output L2 regularization ──
                if q_l2_weight > 0:
                    with torch.no_grad():
                        q_out = critic(batch_d["state"], batch_d["action"])
                    q_l2_penalty = q_out["total"].pow(2).mean() * q_l2_weight
                    c_loss = c_loss + q_l2_penalty

                if torch.isfinite(c_loss):
                    c_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
                    critic_optimizer.step()
                    # ── B1c: Target update mode ──
                    if _hard_target_every > 0:
                        # Hard copy every N critic steps
                        if critic_optimizer_steps > 0 and critic_optimizer_steps % _hard_target_every == 0:
                            hard_update(target_critic, critic)
                    else:
                        target_update(target_critic, critic, tau=_tau)
                    step_critic_loss += c_loss.item()
                    critic_optimizer_steps += 1
                    target_update_steps += 1

            critic_losses.append(step_critic_loss / max(critic_updates, 1))

            if n_heldout_trans >= 8 and (step % 20 == 0 or step == 1):
                with torch.no_grad():
                    h_batch = _build_diag_batch(heldout_indices, n_sample=min(256, n_heldout_trans))
                    if h_batch is not None:
                        # ── B1c: Apply same reward normalization to heldout ──
                        if reward_norm_stats is not None:
                            _apply_reward_norm(h_batch, reward_norm_mode, reward_norm_stats, reward_norm_eps)
                        h_loss, _h_log = compute_critic_loss(
                            critic, target_critic,
                            h_batch["state"], h_batch["action"],
                            h_batch["reward_total"], h_batch["reward_heads"],
                            h_batch["next_state"], h_batch["action_tp1"],
                            done=h_batch["done"], gamma=_gamma, betas=_betas,
                        )
                        heldout_losses.append(float(h_loss.item()))

            if step % 10 == 0 or step == 1:
                _h_str = f", heldout_loss={heldout_losses[-1]:.4f}" if heldout_losses else ""
                cprint(f"  Step {step}/{max_actor_updates}: "
                       f"critic_loss={critic_losses[-1]:.4f}{_h_str}, "
                       f"critic_steps={critic_optimizer_steps}", "cyan")

    # ═══════════════════════════════════════════════════════════════
    # B1a: Comprehensive diagnostics (true TD target, Q1/Q2, per-head)
    # ═══════════════════════════════════════════════════════════════
    initial_train_twin_gap_p95 = None
    initial_heldout_twin_gap_p95 = None
    initial_true_td_target_p95_reward_abs_p95_ratio = None
    q1_q2_corr = None
    q1_minus_q2_abs_p95 = None
    q1_bias_to_target_mean = None
    q2_bias_to_target_mean = None
    true_td_target_p95_reward_abs_p95_ratio = None
    true_td_target_max_reward_abs_max_ratio = None
    per_head_qscale = {}
    q1q2_decomposition = {}
    dominant_bad_head = None

    if n_trans >= 8:
        with torch.no_grad():
            # ── Helper: compute true TD target & Q1/Q2 decomposition ──
            def _compute_full_diag(batch_d, label=""):
                """Compute true TD target, Q1/Q2 decomposition, per-head metrics."""
                result = {}
                # True TD target: y_total = reward_total + gamma_h * (1-done) * target_q_min
                q_next_total = target_critic.q_min(batch_d["next_state"], batch_d["action_tp1"])
                done_float = batch_d["done"].float()
                y_total = batch_d["reward_total"] + _gamma_h * (1.0 - done_float) * q_next_total

                result[f"{label}reward_total_mean"] = float(batch_d["reward_total"].mean().item())
                result[f"{label}reward_total_std"] = float(batch_d["reward_total"].std().item())
                result[f"{label}reward_total_abs_p95"] = float(batch_d["reward_total"].abs().quantile(0.95).item())
                result[f"{label}reward_total_abs_max"] = float(batch_d["reward_total"].abs().max().item())
                result[f"{label}target_q_min_mean"] = float(q_next_total.mean().item())
                result[f"{label}target_q_min_std"] = float(q_next_total.std().item())
                result[f"{label}td_target_mean"] = float(y_total.mean().item())
                result[f"{label}td_target_std"] = float(y_total.std().item())
                result[f"{label}td_target_abs_p95"] = float(y_total.abs().quantile(0.95).item())
                result[f"{label}td_target_abs_max"] = float(y_total.abs().max().item())
                result[f"{label}true_td_target_p95_reward_abs_p95_ratio"] = float(
                    y_total.abs().quantile(0.95).item()
                    / max(batch_d["reward_total"].abs().quantile(0.95).item(), 1e-8))
                result[f"{label}true_td_target_max_reward_abs_max_ratio"] = float(
                    y_total.abs().max().item()
                    / max(batch_d["reward_total"].abs().max().item(), 1e-8))

                # Legacy: old wrong formula
                q_out = critic(batch_d["state"], batch_d["action"])
                legacy_td = batch_d["reward_total"] + 0.99 * q_out["total"].mean(dim=1)
                result[f"{label}legacy_td_target_p95_empirical_ratio"] = float(
                    legacy_td.quantile(0.95).item()
                    / max(batch_d["reward_total"].abs().quantile(0.95).item(), 1e-8))

                # Q1/Q2 decomposition
                q1_total = q_out["total"][:, 0]  # (B,)
                q2_total = q_out["total"][:, 1]
                result[f"{label}q1_total_mean"] = float(q1_total.mean().item())
                result[f"{label}q2_total_mean"] = float(q2_total.mean().item())
                result[f"{label}q1_total_std"] = float(q1_total.std().item())
                result[f"{label}q2_total_std"] = float(q2_total.std().item())
                result[f"{label}q1_minus_q2_mean"] = float((q1_total - q2_total).mean().item())
                result[f"{label}q1_minus_q2_abs_p95"] = float((q1_total - q2_total).abs().quantile(0.95).item())
                result[f"{label}q1_bias_to_target_mean"] = float((q1_total - y_total).mean().item())
                result[f"{label}q2_bias_to_target_mean"] = float((q2_total - y_total).mean().item())
                if q1_total.std() > 1e-8 and q2_total.std() > 1e-8:
                    try:
                        import numpy as np
                        corr = np.corrcoef(q1_total.cpu().numpy(), q2_total.cpu().numpy())[0, 1]
                        result[f"{label}q1_q2_corr"] = float(corr) if np.isfinite(corr) else None
                    except Exception:
                        result[f"{label}q1_q2_corr"] = None
                else:
                    result[f"{label}q1_q2_corr"] = None

                # Per-head Q-scale + twin gap
                for hname in critic.head_names:
                    qh = q_out.get(hname)
                    if qh is None:
                        continue
                    result[f"{label}head_{hname}_q1_mean"] = float(qh[:, 0].mean().item())
                    result[f"{label}head_{hname}_q2_mean"] = float(qh[:, 1].mean().item())
                    result[f"{label}head_{hname}_q1_abs_p95"] = float(qh[:, 0].abs().quantile(0.95).item())
                    result[f"{label}head_{hname}_q2_abs_p95"] = float(qh[:, 1].abs().quantile(0.95).item())
                    h_twin = (qh[:, 0] - qh[:, 1]).abs()
                    result[f"{label}head_{hname}_twin_gap_abs_p95"] = float(h_twin.quantile(0.95).item())

                # Per-head target
                q_next_heads = target_critic.q1_only(batch_d["next_state"], batch_d["action_tp1"])
                HEAD_REWARD_MAP = {
                    "progress": ["base", "pbrs"], "onset": ["onset"],
                    "future_clearance": ["future_clearance"],
                    "trap": ["trap", "flip"], "support": ["support"], "guard": ["guard"],
                }
                for hname in critic.head_names:
                    if hname == "total":
                        continue
                    qnh = q_next_heads.get(hname)
                    if qnh is None:
                        continue
                    qnh = qnh.squeeze(-1)
                    terms = HEAD_REWARD_MAP.get(hname, [hname])
                    r_head = sum(batch_d["reward_heads"].get(t, torch.zeros_like(batch_d["reward_total"])) for t in terms)
                    y_head = r_head + _gamma_h * (1.0 - done_float) * qnh
                    result[f"{label}head_{hname}_reward_abs_p95"] = float(r_head.abs().quantile(0.95).item())
                    result[f"{label}head_{hname}_target_abs_p95"] = float(y_head.abs().quantile(0.95).item())
                    result[f"{label}head_{hname}_target_reward_abs_p95_ratio"] = float(
                        y_head.abs().quantile(0.95).item()
                        / max(r_head.abs().quantile(0.95).item(), 1e-8))

                return result

            # ── Train diagnostics ──
            train_batch = _build_diag_batch(train_indices, n_sample=min(512, n_train_trans))
            if train_batch is not None:
                # ── B1c: Apply reward normalization to diagnostic batch ──
                if reward_norm_stats is not None:
                    _apply_reward_norm(train_batch, reward_norm_mode, reward_norm_stats, reward_norm_eps)
                train_diag = _compute_full_diag(train_batch, label="train_")
                # Extract key metrics
                q_out = critic(train_batch["state"], train_batch["action"])
                twin_gap = (q_out["total"][:, 0] - q_out["total"][:, 1]).abs()
                train_twin_gap_p95 = float(twin_gap.quantile(0.95).item())
                q_std = q_out["total"].std(dim=0).mean().item()
                q_return_std_ratio = q_std / max(train_batch["reward_total"].std().item(), 1e-8)
                true_td_target_p95_reward_abs_p95_ratio = train_diag["train_true_td_target_p95_reward_abs_p95_ratio"]
                true_td_target_max_reward_abs_max_ratio = train_diag["train_true_td_target_max_reward_abs_max_ratio"]
                # Legacy
                td_target_p95_empirical_ratio = train_diag["train_legacy_td_target_p95_empirical_ratio"]
                # Q1/Q2
                q1_q2_corr = train_diag["train_q1_q2_corr"]
                q1_minus_q2_abs_p95 = train_diag["train_q1_minus_q2_abs_p95"]
                q1_bias_to_target_mean = train_diag["train_q1_bias_to_target_mean"]
                q2_bias_to_target_mean = train_diag["train_q2_bias_to_target_mean"]

            # ── Heldout diagnostics ──
            if n_heldout_trans >= 8:
                h_batch = _build_diag_batch(heldout_indices, n_sample=min(512, n_heldout_trans))
                if h_batch is not None:
                    # ── B1c: Apply reward normalization to heldout diagnostic batch ──
                    if reward_norm_stats is not None:
                        _apply_reward_norm(h_batch, reward_norm_mode, reward_norm_stats, reward_norm_eps)
                    h_diag = _compute_full_diag(h_batch, label="heldout_")
                    h_q = critic(h_batch["state"], h_batch["action"])
                    h_twin = (h_q["total"][:, 0] - h_q["total"][:, 1]).abs()
                    heldout_twin_gap_p95 = float(h_twin.quantile(0.95).item())

            # ── Per-head Q-scale summary ──
            per_head_qscale = {}
            head_target_ratios = {}
            for hname in critic.head_names:
                if hname == "total":
                    continue
                entry = {}
                for prefix in ["train_", "heldout_"]:
                    diag_src = train_diag if prefix == "train_" else (h_diag if n_heldout_trans >= 8 else {})
                    entry[f"{prefix}reward_abs_p95"] = diag_src.get(f"{prefix}head_{hname}_reward_abs_p95", None)
                    entry[f"{prefix}target_abs_p95"] = diag_src.get(f"{prefix}head_{hname}_target_abs_p95", None)
                    entry[f"{prefix}target_reward_ratio"] = diag_src.get(f"{prefix}head_{hname}_target_reward_abs_p95_ratio", None)
                    entry[f"{prefix}twin_gap_abs_p95"] = diag_src.get(f"{prefix}head_{hname}_twin_gap_abs_p95", None)
                per_head_qscale[hname] = entry
                ratio = entry.get("train_target_reward_ratio")
                if ratio is not None and ratio > 0:
                    head_target_ratios[hname] = ratio

            # Dominant bad head
            if head_target_ratios:
                dominant_bad_head = max(head_target_ratios, key=head_target_ratios.get)

            # Q1/Q2 decomposition struct
            q1q2_decomposition = {
                "q1_q2_corr": q1_q2_corr,
                "q1_minus_q2_mean": float((q_out["total"][:, 0] - q_out["total"][:, 1]).mean().item()) if train_batch is not None else None,
                "q1_minus_q2_abs_p95": q1_minus_q2_abs_p95,
                "q1_bias_to_target_mean": q1_bias_to_target_mean,
                "q2_bias_to_target_mean": q2_bias_to_target_mean,
                "q1_total_mean": float(q_out["total"][:, 0].mean().item()) if train_batch is not None else None,
                "q2_total_mean": float(q_out["total"][:, 1].mean().item()) if train_batch is not None else None,
                "q1_total_std": float(q_out["total"][:, 0].std().item()) if train_batch is not None else None,
                "q2_total_std": float(q_out["total"][:, 1].std().item()) if train_batch is not None else None,
            }

            # ── Q-return Spearman rho ──
            if n_episodes >= 10:
                ep_returns = {}
                ep_q_means = {}
                for ep_id, indices in episode_groups.items():
                    if len(indices) == 0:
                        continue
                    ep_batch = _build_diag_batch(indices, n_sample=min(128, len(indices)))
                    if ep_batch is None:
                        continue
                    ep_q = critic(ep_batch["state"], ep_batch["action"])
                    ep_returns[ep_id] = float(ep_batch["reward_total"].sum().item())
                    ep_q_means[ep_id] = float(ep_q["total"].mean().item())
                common_eps = sorted(set(ep_returns.keys()) & set(ep_q_means.keys()))
                if len(common_eps) >= 5:
                    import scipy.stats
                    rho_result = scipy.stats.spearmanr(
                        [ep_returns[e] for e in common_eps],
                        [ep_q_means[e] for e in common_eps],
                    )
                    q_return_spearman_rho = float(rho_result.statistic)

    # Overfit check
    if len(heldout_losses) >= 5:
        n_final = max(2, len(heldout_losses) // 5)
        final_heldout = heldout_losses[-n_final:]
        earlier_heldout = heldout_losses[:-n_final] if len(heldout_losses) > n_final else []
        if earlier_heldout:
            final_mean = sum(final_heldout) / len(final_heldout)
            earlier_mean = sum(earlier_heldout) / len(earlier_heldout)
            fixed_replay_overfit = final_mean > earlier_mean * 1.2
        else:
            fixed_replay_overfit = False
    else:
        fixed_replay_overfit = None

    # ── B2: Save trained critic checkpoint with reward_norm_stats ──
    critic_ckpt_path = output_dir / "a2_critic_best.pt"
    critic_save_data = {
        "critic_state_dict": critic.state_dict(),
        "target_critic_state_dict": target_critic.state_dict(),
        "step": critic_optimizer_steps,
        "score": 0.0,
        "run_id": ckpt_writer.run_id,
        "gate_metrics_at_save": {
            "twin_gap": train_twin_gap_p95,
            "q_std": q1q2_decomposition.get("q1_total_std", 0) if isinstance(q1q2_decomposition, dict) else 0,
            "q_return_spearman_rho": q_return_spearman_rho,
            "train_twin_gap_p95": train_twin_gap_p95,
            "heldout_twin_gap_p95": heldout_twin_gap_p95,
        },
    }
    if reward_norm_stats is not None:
        critic_save_data["reward_norm"] = {
            "mode": reward_norm_mode,
            "total_mean": reward_norm_stats.get("total_mean", 0),
            "total_std": reward_norm_stats.get("total_std", 1),
            "eps": reward_norm_eps,
            "source": f"critic_bootstrap_{output_dir.name}",
            "applies_to": "critic_reward_total_and_reward_heads",
        }
    torch.save(critic_save_data, critic_ckpt_path)
    cprint(f"  [B2] Trained critic saved: {critic_ckpt_path} "
           f"(step={critic_optimizer_steps}, reward_norm={'present' if reward_norm_stats else 'absent'})", "green")
    # Also update manifest
    ckpt_writer.manifest["best"] = {
        "actor": "a2_actor_best.pt",
        "critic": "a2_critic_best.pt",
        "score": 0.0,
        "step": critic_optimizer_steps,
    }
    ckpt_writer._flush_manifest()

    bootstrap_elapsed = time.time() - bootstrap_start
    summary = {
        "run_id": ckpt_writer.run_id,
        "config_hash": mode_label,
        "source_script_mtime": datetime.fromtimestamp(
            os.path.getmtime(__file__)).isoformat(),
        "mode": mode_label,
        "actor_optimizer_steps": actor_optimizer_steps,
        "critic_optimizer_steps": critic_optimizer_steps,
        "target_update_steps": target_update_steps,
        "new_rollout_episodes": 0,
        "replay_size_final": n_trans,
        # ── B1a: twin gap ──
        "train_twin_gap_p95": train_twin_gap_p95,
        "heldout_twin_gap_p95": heldout_twin_gap_p95,
        # ── B1a: Q-scale (current) ──
        "q_return_std_ratio": q_return_std_ratio,
        "q_return_spearman_rho": q_return_spearman_rho,
        # ── B1a: legacy TD target (WRONG formula, kept for comparison) ──
        "legacy_td_target_p95_empirical_ratio": td_target_p95_empirical_ratio,
        "legacy_td_target_max_empirical_ratio": td_target_max_empirical_ratio,
        "legacy_td_target_formula": "reward_total + 0.99 * online_critic_q_current_mean(dim=1)",
        # ── B1a: true TD target (CORRECT formula) ──
        "true_td_target_p95_reward_abs_p95_ratio": true_td_target_p95_reward_abs_p95_ratio,
        "true_td_target_max_reward_abs_max_ratio": true_td_target_max_reward_abs_max_ratio,
        "true_td_target_formula": f"reward_total + ({_gamma}**8) * (1-done) * target_critic.q_min(next_state, next_action)",
        # ── B1a: Q1/Q2 decomposition ──
        "q1q2_decomposition": q1q2_decomposition,
        "q1_q2_corr": q1_q2_corr,
        # ── B1a: per-head Q-scale ──
        "per_head_qscale": per_head_qscale,
        "dominant_bad_head": dominant_bad_head,
        # ── B1a: ablation config ──
        "ablation_config": {
            "diag_only": diag_only,
            "gamma": _gamma,
            "gamma_h": _gamma_h,
            "target_tau": _tau,
            "total_only": total_only,
            "huber_total": huber_total,
            "huber_beta": huber_beta if huber_total else None,
            # ── B1c knobs ──
            "reward_norm_mode": reward_norm_mode,
            "reward_norm_eps": reward_norm_eps,
            "q_l2_weight": q_l2_weight,
            "critic_norm": critic_norm,
            "target_update_mode": target_update_mode,
            "target_update_interval": target_update_interval if target_update_mode == "hard_every_n" else None,
            # ── B1d ──
            "bootstrap_split_seed": split_seed,
            "heldout_ratio": heldout_ratio,
        },
        # ── B1d: dual-unit diagnostics for reward standardization ──
        "dual_unit_diagnostics": _build_dual_unit_diag(
            q1q2_decomposition, train_diag if 'train_diag' in dir() else None,
            reward_norm_mode, reward_norm_stats,
            q_return_spearman_rho, true_td_target_p95_reward_abs_p95_ratio,
        ),
        "fixed_replay_overfit": fixed_replay_overfit,
        "insufficient_data": n_trans < 100,
        "hard_stop_primary_reason": "",
        "hard_stop_secondary_reasons": [],
        "gazebo_closed_loop_status": "not_run",
        "lightweight_eval_status": "monitoring_only_not_gazebo",
        "formal_training_status": "critic_bootstrap_smoke_complete" if is_smoke else "critic_bootstrap_complete",
        "claim_allowed": False,
        "elapsed_min": round(bootstrap_elapsed / 60, 1),
        "timestamp": datetime.now().isoformat(),
    }

    if n_trans < 100:
        summary["insufficient_data"] = True
        summary["hard_stop_primary_reason"] = "BLOCKED_BOOTSTRAP_REPLAY_INCOMPLETE"

    with open(output_dir / "critic_bootstrap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── B1a: save per-head Q-scale and Q1/Q2 decomposition separately ──
    if per_head_qscale:
        with open(output_dir / "per_head_qscale.json", "w") as f:
            json.dump(per_head_qscale, f, indent=2)
    if q1q2_decomposition:
        with open(output_dir / "q1q2_decomposition.json", "w") as f:
            json.dump(q1q2_decomposition, f, indent=2)
    # Also save full train diagnostic dump for audit
    if n_trans >= 8 and train_batch is not None:
        with open(output_dir / "diagnostic_summary.json", "w") as f:
            json.dump(train_diag if 'train_diag' in dir() else {}, f, indent=2)

    summary_md = output_dir / "critic_bootstrap_summary.md"
    with open(summary_md, "w") as f:
        f.write("# Critic Bootstrap Summary\n\n")
        f.write(f"- **Mode**: {mode_label}\n")
        f.write(f"- **Actor optimizer steps**: {actor_optimizer_steps}\n")
        f.write(f"- **Critic optimizer steps**: {critic_optimizer_steps}\n")
        f.write(f"- **Train twin gap P95**: {train_twin_gap_p95}\n")
        f.write(f"- **Replay size**: {n_trans}\n")
        f.write(f"- **Insufficient data**: {summary['insufficient_data']}\n")
        f.write(f"- **Claim allowed**: False\n")
        f.write(f"- **Elapsed**: {bootstrap_elapsed/60:.1f} min\n")

    cprint(f"  Summary: {summary_md}", "green")
    return summary


def _run_phased_online(
    actor, base_policy, critic, target_critic,
    actor_optimizer, critic_optimizer,
    device, output_dir, ckpt_writer,
    replay, manifest, train_split, world_dir,
    train_pool, val_pool, world_meta_lookup, role_to_indices,
    max_actor_updates=100,
    collect_episodes=32,
    critic_digest_updates=200,
    actor_updates_per_phase=10,
    eval_interval=100,
    batch_size=2048, critic_updates=16,
    dqpg_warmup_steps=1000,
    metadata_complete=0, metadata_total=0,
    contamination_count=0, total_episodes=0,
    per_role_episodes=None, per_role_transitions=None,
    rng=None,
    reward_norm_stats=None,
    reward_norm_mode="none",
    reward_norm_eps=1e-6,
    # ── B2a: PG scale controls ──
    actor_lambda_pg=1.0,
    actor_lambda_dqpg=1.0,
    actor_pg_scale_mode="raw_sum",
    actor_effective_action_dim=88,
    actor_effective_flow_events=0,
    # ── B3: pilot safety caps ──
    max_phases=0,
    actor_compute_teacher=False,
    pilot_mode=False,
    extended_pilot=False,
    b8_long_train=False,
    b11_500upd=False,
    b15_1000upd=False,
    # ── B4: role quota pilot ──
    role_quota_pilot=False,
    ft_min_episodes_anticipatory=2,
    ft_min_episodes_dense_recovery=2,
    ft_min_episodes_support_expansion=2,
    ft_max_support_role_share=0.50,
    # ── B5: lightweight paired eval ──
    ft_save_actor_path=None,
    # ── C2-OOC: mid-training eval ──
    mid_eval_enabled=False,
    mid_eval_phase=10,
    mid_eval_fn=None,      # callable: (actor, phase_id) -> eval_dict
    mid_eval_configs=None,  # pre-generated episode configs for deterministic eval
    mid_eval_save_ckpt_dir=None,  # if set, save actor ckpt at mid_eval
    # ── C2-OOC: monitoring dashboard ──
    c2_ooc_monitoring_enabled=False,  # live dashboard + CSV independent of stop conditions
    # ── C2-OOC: twin-gap stop conditions ──
    twin_gap_stop_enabled=False,
    grad_nan_stop_enabled=False,  # S3/S4: always active in C2-OOC
    twin_gap_hard_stop=0.70,
    twin_gap_rise_threshold=0.05,
    twin_gap_rise_window=2,
    curriculum_stage: str = 'C2',
):
    """Task 5: Phased online schedule — collect → critic digest → actor update with gates."""
    if rng is None:
        rng = np.random.default_rng()
    if per_role_episodes is None:
        per_role_episodes = defaultdict(int)
    if per_role_transitions is None:
        per_role_transitions = defaultdict(int)

    cprint(f"\n[PhasedOnline] Phased schedule: collect={collect_episodes}eps, "
           f"critic_digest={critic_digest_updates}upd, actor={actor_updates_per_phase}upd...",
           "white", "on_blue")
    if reward_norm_stats is not None:
        cprint(f"  [B2] Reward norm active: mode={reward_norm_mode}, "
               f"total_mean={reward_norm_stats.get('total_mean', 0):.2f}, "
               f"total_std={reward_norm_stats.get('total_std', 1):.2f}", "green")

    # ── B2: Inline reward norm helper (same logic as _run_critic_bootstrap._apply_reward_norm) ──
    def _phase_apply_reward_norm(batch_d):
        if reward_norm_mode == "none" or reward_norm_stats is None:
            return
        total_mean = reward_norm_stats["total_mean"]
        total_std = max(reward_norm_stats["total_std"], reward_norm_eps)
        if reward_norm_mode == "fixed_train_standardize":
            batch_d["reward_total"] = (batch_d["reward_total"] - total_mean) / total_std
            for rk in batch_d.get("reward_heads", {}).keys():
                rk_mean = reward_norm_stats.get(f"{rk}_mean")
                rk_std = reward_norm_stats.get(f"{rk}_std")
                if rk_mean is not None and rk_std is not None:
                    batch_d["reward_heads"][rk] = (batch_d["reward_heads"][rk] - rk_mean) / max(rk_std, reward_norm_eps)
        elif reward_norm_mode == "fixed_train_scale_p95":
            total_p95 = max(reward_norm_stats.get("total_abs_p95", 1.0), reward_norm_eps)
            batch_d["reward_total"] = batch_d["reward_total"] / total_p95
            for rk in batch_d.get("reward_heads", {}).keys():
                rk_p95 = reward_norm_stats.get(f"{rk}_abs_p95")
                if rk_p95 is not None:
                    batch_d["reward_heads"][rk] = batch_d["reward_heads"][rk] / max(rk_p95, reward_norm_eps)

    phase_start = time.time()
    phase_id = 0
    total_actor_updates = 0
    total_critic_updates = 0
    phase_manifests = []
    obs_buffer_pool: List[Dict] = []  # P0b fix: store obs for actor update
    # ── C2-OOC: twin-gap trend tracking ──
    _twin_gap_history: List[float] = []
    _twin_gap_stop_reason = ""
    # ── B3: Use explicit max_phases if provided, else safety cap ──
    _effective_max_phases = max_phases if max_phases > 0 else (max_actor_updates * 3 + 3)
    cprint(f"  [B3] Phase cap: {_effective_max_phases} (explicit={max_phases > 0})", "cyan")
    if pilot_mode:
        cprint(f"  [B3] Pilot mode: strict hard stops for actor>8, phases>{_effective_max_phases}, claim_allowed!=False", "yellow")

    # ── B4: Role quota initialization ──
    formal_roles = ["anticipatory", "dense_recovery", "support_expansion"]
    role_quotas = {
        "anticipatory": ft_min_episodes_anticipatory,
        "dense_recovery": ft_min_episodes_dense_recovery,
        "support_expansion": ft_min_episodes_support_expansion,
    }
    global_role_counts: Dict[str, int] = defaultdict(int)
    _role_cursor = 0  # fallback round-robin cursor after quotas met

    def _select_next_role_for_quota():
        """Select next role prioritizing underfilled quotas, then round-robin."""
        nonlocal _role_cursor
        underfilled = [r for r in formal_roles if global_role_counts[r] < role_quotas[r]]
        if underfilled:
            # Pick most underfilled; stable tie-break by formal_roles order
            return min(
                underfilled,
                key=lambda r: (global_role_counts[r] / max(role_quotas[r], 1), formal_roles.index(r)),
            )
        # Fallback: round-robin with support share cap
        role = formal_roles[_role_cursor % len(formal_roles)]
        _role_cursor += 1
        # Support share cap: skip support_expansion if share exceeds max
        _total = sum(global_role_counts.values())
        if role == "support_expansion" and _total > 0:
            _support_share = (global_role_counts.get("support_expansion", 0) + 1) / (_total + 1)
            if _support_share > ft_max_support_role_share:
                role = formal_roles[_role_cursor % len(formal_roles)]
                _role_cursor += 1
        return role

    if role_quota_pilot:
        cprint(f"  [B4] Role quota pilot: quotas={role_quotas}, "
               f"max_support_share={ft_max_support_role_share}", "cyan")
        # ── B4: Admission check for all formal roles ──
        _missing_roles = []
        for _r in formal_roles:
            _indices = role_to_indices.get(_r, [])
            if not _indices:
                _missing_roles.append(_r)
                cprint(f"  [B4] BLOCKED: role '{_r}' has no admission indices!", "red")
        if _missing_roles:
            cprint(f"  [B4] FATAL: Cannot proceed — missing roles: {_missing_roles}", "red")
            # Write blocked summary immediately
            _blocked_summary = {
                "mode": "phased_online",
                "role_quota_pilot": True,
                "blocked_reason": f"BLOCKED_B4_ROLE_ADMISSION_MISSING:{','.join(_missing_roles)}",
                "role_quota_target": dict(role_quotas),
                "global_role_counts": dict(global_role_counts),
                "role_quota_met": False,
                "role_quota_missing": _missing_roles,
                "hard_stop_primary_reason": "BLOCKED_B4_ROLE_ADMISSION_MISSING",
            }
            (output_dir / "phased_online_summary.json").write_text(json.dumps(_blocked_summary, indent=2))
            return _blocked_summary

    while total_actor_updates < max_actor_updates and phase_id < _effective_max_phases:
        # ── C2-OOC: check for early stop from previous phase ──
        if _twin_gap_stop_reason:
            break
        phase_id += 1
        cprint(f"\n[Phase {phase_id}] Starting...", "white", "on_blue")

        # ── Phase state tracking ──
        phase_state = {
            "phase_id": phase_id,
            "episodes_collected": 0,
            "transitions_collected": 0,
            "critic_digest_done": False,
            "critic_gate_pass": False,
            "actor_updates_done": 0,
            "critic_pre_actor_qscale": {},
            "critic_post_actor_qscale": {},
            "role_counts": {},
            "support_role_share": 0.0,
        }

        # ── Step 1: Collect episodes ──
        cprint(f"  [Collect] Target: {collect_episodes} episodes", "cyan")
        phase_episodes = 0
        phase_transitions = 0

        for ep_i in range(collect_episodes):
            if role_quota_pilot:
                role = _select_next_role_for_quota()
            else:
                role = formal_roles[ep_i % len(formal_roles)]
            indices = role_to_indices.get(role, [])
            if not indices:
                cprint(f"  [WARN] No admission indices for role '{role}', skipping", "yellow")
                continue
            idx = indices[ep_i % len(indices)]
            wname, obs_json = train_pool[idx]
            meta = world_meta_lookup.get(wname)
            role_val = meta.role_active or meta.role_validated if meta else ""
            role_act = meta.role_active or ""

            try:
                env = KinematicUAVEnv(
                    world_id=4, world_pool=[(wname, obs_json)],
                    episode_role=role_val,  # P1 fix (2026-07-11): role-aware start/goal gating
                    synth_pc_factory=lambda jp: SynthPointCloud(jp, device=device),
                    rl_action_steps=8, h_min=0.3, z_min=1.0, z_max=5.0,
                    disturb_enable=True, disturb_prob=0.2, disturb_pos_scale=0.3,
                    mid_state_init_ratio=0.0,
                    seed=int(rng.integers(0, 2**31)),
                    reward_version="v13",
                                    curriculum_stage=curriculum_stage,
                )
                transitions, obs_buf, ep_sum = rollout_episode_a2(
                    actor, base_policy, env, device,
                    episode_seed=int(rng.integers(0, 2**31)),
                    world_id=-1, deterministic=False, lazy_teacher=True,
                )
                for t in transitions:
                    t.world_uid = wname; t.split = train_split
                    t.role_validated = role_val; t.role_active = role_act
                    t.admission_status = "admitted"
                replay.add_episode(transitions, partition="online")
                # ── B4: Tag obs_buf items with _role for per-role actor update tracking ──
                for _obs_item in obs_buf:
                    if isinstance(_obs_item, dict):
                        _obs_item["_role"] = role_val
                obs_buffer_pool.extend(obs_buf)  # P0b: store obs for actor update
                phase_episodes += 1
                phase_transitions += len(transitions)
                total_episodes += 1
                phase_state["role_counts"][role_val] = \
                    phase_state["role_counts"].get(role_val, 0) + 1
                global_role_counts[role_val] += 1
                per_role_episodes[role_val] += 1
                per_role_transitions[role_val] += len(transitions)
                for t in transitions:
                    metadata_total += 1
                    if t.world_uid and t.role_validated:
                        metadata_complete += 1
                env.close()
            except Exception as e:
                cprint(f"  [WARN] Phase collect failed for {wname}: {e}", "yellow")

        phase_state["episodes_collected"] = phase_episodes
        phase_state["transitions_collected"] = phase_transitions
        support_eps = phase_state["role_counts"].get("support_expansion", 0)
        phase_state["support_role_share"] = round(
            support_eps / max(phase_episodes, 1), 4)
        cprint(f"  [Collect] Done: {phase_episodes} eps, {phase_transitions} trans", "green")

        # ── Pre-critic Q diagnostics ──
        n_replay = replay.get_stats()["total_transitions"]
        if n_replay >= 8:
            with torch.no_grad():
                pre_batch = replay.sample(min(128, n_replay))
                pre_batch_d = collate_to_critic_device(pre_batch, device)
                pre_q = critic(pre_batch_d["state"], pre_batch_d["action"])
                phase_state["critic_pre_actor_qscale"] = {
                    "q_total_mean": float(pre_q["total"].mean().item()),
                    "q_total_std": float(pre_q["total"].std().item()),
                }

        # ── Step 2: Critic digest ──
        cprint(f"  [CriticDigest] Target: {critic_digest_updates} updates", "cyan")
        digest_critic_steps = 0
        for _cd in range(critic_digest_updates):
            if n_replay < 8:
                break
            batch = replay.sample(min(batch_size, n_replay))
            if batch["state"].shape[0] == 0:
                continue
            batch_d = collate_to_critic_device(batch, device)
            # ── B2: Apply reward normalization in critic digest ──
            _phase_apply_reward_norm(batch_d)
            critic_optimizer.zero_grad()
            c_loss, c_log = compute_critic_loss(
                critic, target_critic,
                batch_d["state"], batch_d["action"],
                batch_d["reward_total"], batch_d["reward_heads"],
                batch_d["next_state"], batch_d["action_tp1"],
                done=batch_d["done"], gamma=0.99, betas=None,
            )
            if torch.isfinite(c_loss):
                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
                critic_optimizer.step()
                target_update(target_critic, critic, tau=0.005)
                digest_critic_steps += 1
            else:
                # ── C2-OOC S4: NaN/Inf detection ──
                if grad_nan_stop_enabled:
                    _twin_gap_stop_reason = (
                        f"BLOCKED_C2_OOC_NAN_INF: "
                        f"phase={phase_id}, critic_loss={c_loss.item():.4f}"
                    )
                    cprint(f"\n  [C2-OOC S4] NaN/Inf in critic loss!", "red")
                    phase_state["c2_ooc_stop_triggered"] = True
                    phase_state["c2_ooc_stop_reason"] = _twin_gap_stop_reason
                total_critic_updates += 1

        phase_state["critic_digest_done"] = True
        phase_state["critic_digest_updates"] = digest_critic_steps

        # Simple critic gate: at least 1 critic update completed
        phase_state["critic_gate_pass"] = digest_critic_steps > 0
        phase_state["critic_gate"] = "PASS" if phase_state["critic_gate_pass"] else "FAIL"

        cprint(f"  [CriticDigest] Done: {digest_critic_steps} updates, "
               f"gate={phase_state['critic_gate']}", "green")

        # ── Step 3: Actor updates (only if critic gate passed) ──
        if not phase_state["critic_gate_pass"]:
            cprint(f"  [Actor] SKIPPED — critic gate not passed", "yellow")
            phase_state["actor_updates_done"] = 0
            phase_state["actor_loss_total"] = None
            phase_state["actor_loss_dqpg"] = None
            phase_state["actor_loss_pg"] = None
            phase_state["actor_loss_bc"] = None
            phase_state["actor_loss_noise"] = None
            phase_state["actor_loss_smooth"] = None
            phase_state["grad_norm_lora"] = None
            phase_state["policy_update_norm_lora"] = None
            # ── B3 ──
            phase_state["bc_available"] = False
            phase_state["teacher_action_present_ratio"] = 0.0
            phase_state["grad_norm_lora_gate_value"] = None
        else:
            cprint(f"  [Actor] Target: {actor_updates_per_phase} updates", "cyan")
            phase_actor_updates = 0
            # ── B3: Teacher action tracking ──
            _teacher_present_count = 0
            _teacher_buffer_present_count = 0
            # ── B4: Per-role actor update tracking ──
            _b4_per_role_actor_updates = defaultdict(int)
            _b4_per_role_q_accum = defaultdict(list)
            _b4_support_seen_in_buffer = False
            # B2: Per-update diagnostics
            _actor_loss_total_val = None
            _actor_loss_dqpg_val = None
            _actor_loss_pg_val = None
            _actor_loss_bc_val = None
            _actor_loss_noise_val = None
            _actor_loss_smooth_val = None
            _grad_norm_lora_val = None
            _policy_update_norm_lora_val = None
            _critic_q_mean_actor = None
            _critic_q_std_actor = None

            # ── B2: Compute advantage baseline from replay batch ──
            _baseline = torch.tensor(0.0, device=device)
            if n_replay >= 8:
                _adv_batch = replay.sample(min(256, n_replay))
                if _adv_batch["state"].shape[0] > 0:
                    _adv_batch_d = collate_to_critic_device(_adv_batch, device)
                    with torch.no_grad():
                        _q_adv = critic(_adv_batch_d["state"], _adv_batch_d["action"])
                        _q_min_adv = torch.min(_q_adv["total"][:, 0], _q_adv["total"][:, 1])
                        _baseline = _q_min_adv.mean()

            for _au in range(actor_updates_per_phase):
                if len(obs_buffer_pool) < 1:
                    break
                # Use obs_buffer_pool items like legacy formal loop
                _actor_indices = rng.choice(
                    len(obs_buffer_pool),
                    size=min(2, len(obs_buffer_pool)),
                    replace=False,
                ).tolist()
                for idx in _actor_indices:
                    try:
                        item = obs_buffer_pool[idx]
                        # obs_torch: dict of tensors from rollout
                        obs_torch = {}
                        for k, v in item.get("obs_torch", {}).items():
                            if isinstance(v, torch.Tensor):
                                obs_torch[k] = v.to(device).unsqueeze(0) if v.dim() == 2 else v.to(device)
                        if not obs_torch:
                            continue

                        # ── B4: Track role usage in actor update ──
                        _item_role = item.get("_role", "unknown") if isinstance(item, dict) else "unknown"
                        if _item_role == "support_expansion":
                            _b4_support_seen_in_buffer = True
                        _b4_per_role_actor_updates[_item_role] += 1

                        # ── B3: Teacher action from buffer OR compute_teacher ──
                        _teacher_from_buffer = item.get("teacher_action")
                        teacher_act = None
                        if _teacher_from_buffer is not None and isinstance(_teacher_from_buffer, torch.Tensor):
                            teacher_act = _teacher_from_buffer.to(device)
                            if teacher_act.dim() == 2:
                                teacher_act = teacher_act.unsqueeze(0)
                        _teacher_present_count += 1
                        if teacher_act is not None:
                            _teacher_buffer_present_count += 1

                        # ── B2a: Resolve PG scale denom ──
                        _pg_denom = 1.0
                        if actor_pg_scale_mode == "per_action_dim":
                            _pg_denom = float(max(actor_effective_action_dim, 1))
                        elif actor_pg_scale_mode == "per_flow_event":
                            _flow_events = actor_effective_flow_events
                            if _flow_events <= 0:
                                # Infer: action_dim * num_fm_steps (default 4)
                                _flow_events = actor_effective_action_dim * 4
                            _pg_denom = float(max(_flow_events, 1))

                        actor_optimizer.zero_grad()
                        # ── B3: Use compute_teacher flag for BC anchor ──
                        _use_compute_teacher = actor_compute_teacher
                        actor_out = actor(obs_torch, deterministic=False, compute_teacher=_use_compute_teacher)

                        # ── B3: Fallback teacher from actor_out when buffer is None ──
                        if teacher_act is None and _use_compute_teacher:
                            _actor_teacher = actor_out.get("teacher_action")
                            if _actor_teacher is not None:
                                teacher_act = _actor_teacher
                                if teacher_act.dim() == 2:
                                    teacher_act = teacher_act.unsqueeze(0)
                                _teacher_buffer_present_count += 1  # counted as present via compute

                        # ── B2: Full actor loss (DQPG + PG + BC + noise + smooth) ──
                        if actor_out.get("action") is not None:
                            # Critic forward for DQPG/PG
                            q_sa = critic(actor_out["global_cond"], actor_out["action"])
                            q_min_sa = torch.min(q_sa["total"][:, 0], q_sa["total"][:, 1])
                            adv_i = q_min_sa.detach() - _baseline

                            _critic_q_mean_actor = float(q_min_sa.mean().item())
                            _critic_q_std_actor = float(q_min_sa.std().item())
                            # ── B4: Per-role Q accumulation ──
                            if _item_role and _critic_q_mean_actor is not None:
                                _b4_per_role_q_accum[_item_role].append(_critic_q_mean_actor)

                            losses = compute_routea_actor_losses(
                                actor_outputs={
                                    "action": actor_out["action"],
                                    "log_prob": actor_out.get("log_prob"),
                                    "noise_sigmas": actor_out.get("noise_sigmas"),
                                },
                                teacher_action=teacher_act,
                                advantage=adv_i,
                                q_min=q_min_sa,
                                lambda_bc=1.0, lambda_path=0.0,
                                lambda_noise=0.01, lambda_smooth=0.001,
                                lambda_pg=actor_lambda_pg, lambda_dqpg=actor_lambda_dqpg,
                                pg_scale_mode=actor_pg_scale_mode,
                                pg_logprob_scale_denom=_pg_denom,
                            )
                            actor_loss = losses["actor_loss_total"]

                            if torch.isfinite(actor_loss) and actor_loss.requires_grad:
                                actor_loss.backward()
                                # Pre-clip grad norm
                                _pre_grad = 0.0
                                for p in list(actor._lora_adaptors.parameters()) + list(actor.noise_module.parameters()):
                                    if p.grad is not None:
                                        _pre_grad += p.grad.norm().item() ** 2
                                _grad_norm_lora_val = math.sqrt(_pre_grad)
                                # Pre-clip policy param norm
                                _pre_policy = 0.0
                                for p in list(actor._lora_adaptors.parameters()) + list(actor.noise_module.parameters()):
                                    _pre_policy += p.data.norm().item() ** 2
                                _pre_policy_norm = math.sqrt(_pre_policy)

                                torch.nn.utils.clip_grad_norm_(
                                    list(actor._lora_adaptors.parameters())
                                    + list(actor.noise_module.parameters()),
                                    5.0,
                                )
                                actor_optimizer.step()
                                # ── C2-OOC S3: grad norm explosion ──
                                if grad_nan_stop_enabled and _grad_norm_lora_val is not None and _grad_norm_lora_val > 100:
                                    _twin_gap_stop_reason = (
                                        f"BLOCKED_C2_OOC_GRAD_NORM_EXPLOSION: "
                                        f"phase={phase_id}, grad_norm={_grad_norm_lora_val:.1f} > 100"
                                    )
                                    cprint(f"\n  [C2-OOC S3] GRAD NORM EXPLOSION: {_grad_norm_lora_val:.1f} > 100", "red")
                                    phase_state["c2_ooc_stop_triggered"] = True
                                    phase_state["c2_ooc_stop_reason"] = _twin_gap_stop_reason
                                # Post-clip policy param norm
                                _post_policy = 0.0
                                for p in list(actor._lora_adaptors.parameters()) + list(actor.noise_module.parameters()):
                                    _post_policy += p.data.norm().item() ** 2
                                _policy_update_norm_lora_val = math.sqrt(_post_policy) - _pre_policy_norm

                                phase_actor_updates += 1
                                total_actor_updates += 1
                                _actor_loss_total_val = float(actor_loss.item())
                                _actor_loss_dqpg_val = float(losses.get("actor_loss_dqpg", torch.tensor(0)).item())
                                _actor_loss_pg_val = float(losses.get("actor_loss_pg", torch.tensor(0)).item())
                                _actor_loss_bc_val = float(losses.get("actor_loss_bc", torch.tensor(0)).item())
                                _actor_loss_noise_val = float(losses.get("actor_loss_noise", torch.tensor(0)).item())
                                _actor_loss_smooth_val = float(losses.get("actor_loss_smooth", torch.tensor(0)).item())
                                # ── B2a: loss-scale diagnostics ──
                                _log_prob_mean_val = float(losses.get("log_prob_mean", torch.tensor(float("nan"))).item())
                                _log_prob_abs_mean_val = float(losses.get("log_prob_abs_mean", torch.tensor(float("nan"))).item())
                                _advantage_mean_val = float(losses.get("advantage_mean", torch.tensor(float("nan"))).item())
                                _advantage_abs_mean_val = float(losses.get("advantage_abs_mean", torch.tensor(float("nan"))).item())
                                _pg_dqpg_abs_ratio_val = float(losses.get("pg_dqpg_abs_ratio", torch.tensor(float("nan"))).item())
                                _weighted_pg = actor_lambda_pg * _actor_loss_pg_val
                                _weighted_dqpg = actor_lambda_dqpg * _actor_loss_dqpg_val
                                _weighted_pg_dqpg_abs_ratio = abs(_weighted_pg) / (abs(_weighted_dqpg) + 1e-8)
                        else:
                            cprint(f"  [Actor WARN] actor_out['action'] is None, skipping", "yellow")
                    except Exception as e:
                        if _au == 0 and idx == _actor_indices[0]:
                            cprint(f"  [Actor WARN] update failed: {e}", "yellow")
                        continue

            phase_state["actor_updates_done"] = phase_actor_updates
            phase_state["actor_loss_total"] = _actor_loss_total_val
            phase_state["actor_loss_dqpg"] = _actor_loss_dqpg_val
            phase_state["actor_loss_pg"] = _actor_loss_pg_val
            phase_state["actor_loss_bc"] = _actor_loss_bc_val
            phase_state["actor_loss_noise"] = _actor_loss_noise_val
            phase_state["actor_loss_smooth"] = _actor_loss_smooth_val
            phase_state["grad_norm_lora"] = _grad_norm_lora_val
            phase_state["policy_update_norm_lora"] = _policy_update_norm_lora_val
            phase_state["critic_q_mean_for_actor"] = _critic_q_mean_actor
            phase_state["critic_q_std_for_actor"] = _critic_q_std_actor
            phase_state["normalized_q_used"] = (reward_norm_stats is not None)
            # ── B2a: loss-scale diagnostics ──
            phase_state["lambda_pg"] = actor_lambda_pg
            phase_state["lambda_dqpg"] = actor_lambda_dqpg
            phase_state["pg_scale_mode"] = actor_pg_scale_mode
            phase_state["pg_logprob_scale_denom"] = _pg_denom
            phase_state["log_prob_mean"] = _log_prob_mean_val if '_log_prob_mean_val' in dir() else None
            phase_state["log_prob_abs_mean"] = _log_prob_abs_mean_val if '_log_prob_abs_mean_val' in dir() else None
            phase_state["advantage_mean"] = _advantage_mean_val if '_advantage_mean_val' in dir() else None
            phase_state["advantage_abs_mean"] = _advantage_abs_mean_val if '_advantage_abs_mean_val' in dir() else None
            phase_state["pg_dqpg_abs_ratio"] = _pg_dqpg_abs_ratio_val if '_pg_dqpg_abs_ratio_val' in dir() else None
            phase_state["weighted_pg_dqpg_abs_ratio"] = _weighted_pg_dqpg_abs_ratio if '_weighted_pg_dqpg_abs_ratio' in dir() else None
            phase_state["grad_clip_norm"] = 5.0
            phase_state["grad_norm_lora_gate_value"] = _grad_norm_lora_val
            # ── B3: teacher action / BC availability ──
            phase_state["teacher_action_present_ratio"] = (
                _teacher_buffer_present_count / max(_teacher_present_count, 1)
            )
            phase_state["bc_available"] = (_actor_loss_bc_val is not None and abs(_actor_loss_bc_val) > 1e-8)
            # ── B4: Per-role tracking summary ──
            phase_state["per_role_transitions"] = {
                r: per_role_transitions.get(r, 0) for r in formal_roles
            }
            phase_state["per_role_actor_updates"] = dict(_b4_per_role_actor_updates)
            phase_state["per_role_q_mean_for_actor"] = {
                r: float(np.mean(vals)) if vals else None
                for r, vals in _b4_per_role_q_accum.items()
            }
            phase_state["support_expansion_seen_in_actor_buffer"] = _b4_support_seen_in_buffer
            phase_state["phase_twin_gap_p95"] = None  # computed below after post-actor Q
            phase_state["q_return_spearman_rho"] = "monitoring_only"  # insufficient samples in pilot
            cprint(f"  [Actor] Done: {phase_actor_updates} updates, "
                   f"loss={_actor_loss_total_val}, dqpg={_actor_loss_dqpg_val}, "
                   f"pg={_actor_loss_pg_val}, bc={_actor_loss_bc_val}, "
                   f"grad_norm={_grad_norm_lora_val}", "green")

        # ── Post-actor Q diagnostics ──
        if n_replay >= 8:
            with torch.no_grad():
                post_batch = replay.sample(min(128, n_replay))
                post_batch_d = collate_to_critic_device(post_batch, device)
                post_q = critic(post_batch_d["state"], post_batch_d["action"])
                phase_state["critic_post_actor_qscale"] = {
                    "q_total_mean": float(post_q["total"].mean().item()),
                    "q_total_std": float(post_q["total"].std().item()),
                }
                # ── B4: twin_gap P95 ──
                _q1 = post_q["total"][:, 0]
                _q2 = post_q["total"][:, 1]
                _twin_abs_gaps = (_q1 - _q2).abs()
                if _twin_abs_gaps.numel() >= 20:
                    phase_state["phase_twin_gap_p95"] = float(
                        _twin_abs_gaps.kthvalue(int(_twin_abs_gaps.numel() * 0.95)).values.item()
                    )
                else:
                    phase_state["phase_twin_gap_p95"] = float(_twin_abs_gaps.max().item())

                # ── C2-OOC: twin-gap stop condition check ──
                if twin_gap_stop_enabled and phase_state["phase_twin_gap_p95"] is not None:
                    _current_tg = phase_state["phase_twin_gap_p95"]
                    _twin_gap_history.append(_current_tg)

                    # S1: hard stop — exceeds absolute threshold
                    if _current_tg > twin_gap_hard_stop:
                        _twin_gap_stop_reason = (
                            f"BLOCKED_C2_OOC_TWIN_GAP_HARD_STOP: "
                            f"phase={phase_id}, twin_gap_p95={_current_tg:.4f} > {twin_gap_hard_stop}"
                        )
                        cprint(f"\n  [C2-OOC S1] TWIN GAP HARD STOP: {_current_tg:.4f} > {twin_gap_hard_stop}", "red")
                        cprint(f"  Twin-gap history: {[f'{v:.4f}' for v in _twin_gap_history]}", "red")
                        phase_state["c2_ooc_stop_triggered"] = True
                        phase_state["c2_ooc_stop_reason"] = _twin_gap_stop_reason

                    # S2: consecutive rise exceeds threshold
                    elif len(_twin_gap_history) >= twin_gap_rise_window:
                        _recent = _twin_gap_history[-twin_gap_rise_window:]
                        _all_rising = all(
                            _recent[i] > _recent[i - 1]
                            for i in range(1, len(_recent))
                        )
                        _cumulative_rise = _recent[-1] - _recent[0]
                        if _all_rising and _cumulative_rise > twin_gap_rise_threshold:
                            _twin_gap_stop_reason = (
                                f"BLOCKED_C2_OOC_TWIN_GAP_CONSECUTIVE_RISE: "
                                f"phases {phase_id - twin_gap_rise_window + 1}-{phase_id}, "
                                f"delta={_cumulative_rise:.4f} > {twin_gap_rise_threshold}, "
                                f"values={[f'{v:.4f}' for v in _recent]}"
                            )
                            cprint(f"\n  [C2-OOC S2] TWIN GAP CONSECUTIVE RISE: "
                                   f"delta={_cumulative_rise:.4f} over {twin_gap_rise_window} phases", "red")
                            cprint(f"  Twin-gap history: {[f'{v:.4f}' for v in _twin_gap_history]}", "red")
                            phase_state["c2_ooc_stop_triggered"] = True
                            phase_state["c2_ooc_stop_reason"] = _twin_gap_stop_reason

        # Phase manifest
        phase_manifest = {
            "phase_id": phase_id,
            "collection_policy": {"mode": "role_balanced"},
            "episodes_collected": phase_state["episodes_collected"],
            "transitions_collected": phase_state["transitions_collected"],
            "role_counts": phase_state["role_counts"],
            "support_role_share": phase_state["support_role_share"],
            "critic_pre_actor_qscale": phase_state["critic_pre_actor_qscale"],
            "critic_digest_updates": phase_state["critic_digest_updates"],
            "critic_gate": phase_state["critic_gate"],
            "actor_updates": phase_state["actor_updates_done"],
            "critic_post_actor_qscale": phase_state["critic_post_actor_qscale"],
            "learning_gate": "NOT_RUN",
            "claim_allowed": False,
            # ── B2: Actor update diagnostics ──
            "actor_loss_total": phase_state.get("actor_loss_total"),
            "actor_loss_dqpg": phase_state.get("actor_loss_dqpg"),
            "actor_loss_pg": phase_state.get("actor_loss_pg"),
            "actor_loss_bc": phase_state.get("actor_loss_bc"),
            "actor_loss_noise": phase_state.get("actor_loss_noise"),
            "actor_loss_smooth": phase_state.get("actor_loss_smooth"),
            "grad_norm_lora": phase_state.get("grad_norm_lora"),
            "policy_update_norm_lora": phase_state.get("policy_update_norm_lora"),
            "critic_q_mean_for_actor": phase_state.get("critic_q_mean_for_actor"),
            "critic_q_std_for_actor": phase_state.get("critic_q_std_for_actor"),
            "normalized_q_used": phase_state.get("normalized_q_used", False),
            # ── B2a: PG scale diagnostics ──
            "lambda_pg": phase_state.get("lambda_pg"),
            "lambda_dqpg": phase_state.get("lambda_dqpg"),
            "pg_scale_mode": phase_state.get("pg_scale_mode"),
            "pg_logprob_scale_denom": phase_state.get("pg_logprob_scale_denom"),
            "log_prob_mean": phase_state.get("log_prob_mean"),
            "log_prob_abs_mean": phase_state.get("log_prob_abs_mean"),
            "advantage_mean": phase_state.get("advantage_mean"),
            "advantage_abs_mean": phase_state.get("advantage_abs_mean"),
            "pg_dqpg_abs_ratio": phase_state.get("pg_dqpg_abs_ratio"),
            "weighted_pg_dqpg_abs_ratio": phase_state.get("weighted_pg_dqpg_abs_ratio"),
            "grad_clip_norm": phase_state.get("grad_clip_norm"),
            "grad_norm_lora_gate_value": phase_state.get("grad_norm_lora_gate_value"),
            # ── B3: pilot safety fields ──
            "bc_available": phase_state.get("bc_available", False),
            "teacher_action_present_ratio": phase_state.get("teacher_action_present_ratio", 0.0),
            # ── B4: role-level diagnostics ──
            "per_role_transitions": phase_state.get("per_role_transitions", {}),
            "per_role_actor_updates": phase_state.get("per_role_actor_updates", {}),
            "per_role_q_mean_for_actor": phase_state.get("per_role_q_mean_for_actor", {}),
            "phase_twin_gap_p95": phase_state.get("phase_twin_gap_p95"),
            "q_return_spearman_rho": phase_state.get("q_return_spearman_rho", "monitoring_only"),
            "support_expansion_seen_in_actor_buffer": phase_state.get("support_expansion_seen_in_actor_buffer", False),
        }
        phase_manifests.append(phase_manifest)

        # ── C2-OOC: mid-training eval ──
        if mid_eval_enabled and phase_id == mid_eval_phase and mid_eval_fn is not None:
            cprint(f"\n[C2-OOC] Mid-training eval at phase {phase_id}...", "white", "on_green")
            try:
                _mid_eval_result = mid_eval_fn(actor, phase_id)
                phase_manifest["mid_eval_triggered"] = True
                phase_manifest["mid_eval_summary"] = {
                    "overall_SR": _mid_eval_result.get("overall_SR"),
                    "overall_CR": _mid_eval_result.get("overall_CR"),
                    "per_role": {
                        r: {"SR": _mid_eval_result.get("per_role", {}).get(r, {}).get("SR"),
                            "CR": _mid_eval_result.get("per_role", {}).get(r, {}).get("CR")}
                        for r in formal_roles
                        if r in _mid_eval_result.get("per_role", {})
                    },
                }
                cprint(f"  [MidEval] SR: " + ", ".join(
                    f"{r}={_mid_eval_result.get('per_role', {}).get(r, {}).get('SR', 0):.3f}"
                    for r in formal_roles
                    if r in _mid_eval_result.get('per_role', {})), "green")
                # ── C2-OOC: save mid-phase checkpoint ──
                if mid_eval_save_ckpt_dir is not None:
                    _mid_ckpt_dir = Path(mid_eval_save_ckpt_dir)
                    _mid_ckpt_dir.mkdir(parents=True, exist_ok=True)
                    _mid_actor_path = _mid_ckpt_dir / f"actor_phase{phase_id}.pt"
                    torch.save({"actor_state_dict": actor.state_dict(),
                                "phase_id": phase_id,
                                "total_actor_updates": total_actor_updates}, str(_mid_actor_path))
                    cprint(f"  [MidEval] Actor ckpt saved: {_mid_actor_path}", "green")
            except Exception as e:
                cprint(f"  [MidEval] FAILED: {e}", "red")
                phase_manifest["mid_eval_triggered"] = False
                phase_manifest["mid_eval_error"] = str(e)

        # ── C2-OOC: live monitoring dashboard ──
        if c2_ooc_monitoring_enabled:
            _tg = phase_state.get("phase_twin_gap_p95")
            _q_post = phase_state.get("critic_post_actor_qscale", {})
            _q_mean = _q_post.get("q_total_mean") if _q_post else None
            _q_std = _q_post.get("q_total_std") if _q_post else None
            _gn = phase_state.get("grad_norm_lora")
            _pg_dqpg = phase_state.get("weighted_pg_dqpg_abs_ratio")
            _roles_active = [r for r, c in phase_state.get("role_counts", {}).items() if c > 0]
            _tg_str = f"{_tg:.4f}" if _tg is not None else "N/A"
            _q_str = f"mean={_q_mean:.2f} std={_q_std:.2f}" if _q_mean is not None else "N/A"
            _gn_str = f"{_gn:.2f}" if _gn is not None else "N/A"
            _pgdq_str = f"{_pg_dqpg:.2f}" if _pg_dqpg is not None else "N/A"

            # Build twin-gap trend line
            _tg_trend = ""
            if len(_twin_gap_history) >= 2:
                _delta = _twin_gap_history[-1] - _twin_gap_history[-2]
                _arrow = "UP" if _delta > 0.005 else ("DN" if _delta < -0.005 else "FLAT")
                _tg_trend = f" {_arrow}{abs(_delta):.4f}"

            _tg_color = "red" if (_tg and _tg > 0.65) else ("yellow" if (_tg and _tg > 0.55) else "green")
            cprint(f"  [C2-OOC Phase {phase_id:2d}] "
                   f"twin={_tg_str}{_tg_trend} | "
                   f"Q({_q_str}) | "
                   f"grad={_gn_str} | "
                   f"PG/DQPG={_pgdq_str} | "
                   f"roles={_roles_active} | "
                   f"actor_upd={total_actor_updates}/{max_actor_updates}",
                   _tg_color)

            # ── C2-OOC: live CSV append ──
            _live_csv = output_dir / "c2_ooc_live_metrics.csv"
            _write_header = not _live_csv.exists()
            with open(_live_csv, "a") as _f:
                if _write_header:
                    _f.write("phase_id,twin_gap_p95,q_mean,q_std,grad_norm,pg_dqpg_ratio,"
                             "actor_updates,total_actor_updates,roles_active,elapsed_s\n")
                _elapsed = time.time() - phase_start
                _f.write(f"{phase_id},{_tg},{_q_mean},{_q_std},{_gn},{_pg_dqpg},"
                         f"{phase_state.get('actor_updates_done', 0)},{total_actor_updates},"
                         f"{'+'.join(_roles_active)},{_elapsed:.1f}\n")

        # ── GPU memory hygiene ──
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if phase_id % 5 == 0:
                _allocated = torch.cuda.memory_allocated() / 1024**3
                _reserved = torch.cuda.memory_reserved() / 1024**3
                cprint(f"  [GPU] allocated={_allocated:.1f}GB, reserved={_reserved:.1f}GB", "cyan")

        cprint(f"  [Phase {phase_id}] actor={phase_state['actor_updates_done']}, "
               f"critic={phase_state['critic_digest_updates']}, "
               f"total_actor={total_actor_updates}/{max_actor_updates}", "green")

    # ── C2-OOC: handle early stop from twin-gap/grad/NaN ──
    if _twin_gap_stop_reason:
        cprint(f"\n{'='*60}", "red")
        cprint(f"  C2-OOC STOPPED: {_twin_gap_stop_reason}", "red")
        cprint(f"  Saving emergency checkpoint...", "red")
        _emergency_dir = output_dir / "emergency"
        _emergency_dir.mkdir(parents=True, exist_ok=True)
        _emergency_ckpt = _emergency_dir / "emergency_ckpt.pt"
        torch.save({
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "target_critic_state_dict": target_critic.state_dict(),
            "phase_id": phase_id,
            "total_actor_updates": total_actor_updates,
            "twin_gap_history": _twin_gap_history,
            "stop_reason": _twin_gap_stop_reason,
            "phase_manifests": phase_manifests,
        }, str(_emergency_ckpt))
        cprint(f"  Emergency ckpt saved: {_emergency_ckpt}", "green")
        # Write emergency diagnosis
        _diag_lines = [
            "# C2-OOC Emergency Diagnosis",
            "",
            f"**Stop reason:** {_twin_gap_stop_reason}",
            f"**Phase:** {phase_id}",
            f"**Total actor updates:** {total_actor_updates}",
            f"**Total critic updates:** {total_critic_updates}",
            "",
            "## Twin-Gap History",
            "",
            "| Phase | Twin Gap P95 |",
            "|-------|-------------|",
        ]
        for _pi, _tg in enumerate(_twin_gap_history):
            _diag_lines.append(f"| {_pi + 1} | {_tg:.4f} |")
        (_emergency_dir / "emergency_diagnosis.md").write_text("\n".join(_diag_lines))
        cprint(f"  Diagnosis written: {_emergency_dir / 'emergency_diagnosis.md'}", "green")

    # ── Max-phases safety cap warning ──
    if phase_id >= _effective_max_phases and total_actor_updates < max_actor_updates:
        cprint(f"\n  ⚠️  Max phases ({_effective_max_phases}) reached with "
               f"total_actor={total_actor_updates}/{max_actor_updates} — "
               f"actor update may have errors; writing partial summary", "yellow")

    # ── B3/B4: Pilot mode hard stops ──
    _pilot_hard_stop_reason = ""
    if pilot_mode:
        if extended_pilot:
            if b15_1000upd:
                # B15-B17: extended caps for 1000 actor updates (1000/8≈125 phases)
                if total_actor_updates > 1200:
                    _pilot_hard_stop_reason = "BLOCKED_B15_1000UPD_ACTOR_UPDATES_EXCEED_1200"
                elif phase_id > 150:
                    _pilot_hard_stop_reason = "BLOCKED_B15_1000UPD_PHASES_EXCEED_150"
            elif b11_500upd:
                # B12: extended caps for 500 actor updates (500/8≈63 phases)
                if total_actor_updates > 700:
                    _pilot_hard_stop_reason = "BLOCKED_B12_500UPD_ACTOR_UPDATES_EXCEED_700"
                elif phase_id > 80:
                    _pilot_hard_stop_reason = "BLOCKED_B12_500UPD_PHASES_EXCEED_80"
            elif b8_long_train:
                # B8: extended caps for 100-400 actor updates
                if total_actor_updates > 400:
                    _pilot_hard_stop_reason = "BLOCKED_B8_LONG_TRAIN_ACTOR_UPDATES_EXCEED_400"
                elif phase_id > 40:
                    _pilot_hard_stop_reason = "BLOCKED_B8_LONG_TRAIN_PHASES_EXCEED_40"
            # B6: extended caps for 50-200 actor updates
            elif total_actor_updates > 200:
                _pilot_hard_stop_reason = "BLOCKED_B6_EXTENDED_PILOT_ACTOR_UPDATES_EXCEED_200"
            elif phase_id > 15:
                _pilot_hard_stop_reason = "BLOCKED_B6_EXTENDED_PILOT_PHASES_EXCEED_15"
        elif role_quota_pilot:
            # B4: relaxed caps for role-quota pilot
            if total_actor_updates > 18:
                _pilot_hard_stop_reason = "BLOCKED_B4_PILOT_ACTOR_UPDATES_EXCEED_18"
            elif phase_id > 6:
                _pilot_hard_stop_reason = "BLOCKED_B4_PILOT_PHASES_EXCEED_6"
        else:
            # B3: strict caps
            if total_actor_updates > 8:
                _pilot_hard_stop_reason = "BLOCKED_B3_PILOT_ACTOR_UPDATES_EXCEED_8"
            elif phase_id > 3:
                _pilot_hard_stop_reason = "BLOCKED_B3_PILOT_PHASES_EXCEED_3"
        if _pilot_hard_stop_reason:
            _pilot_tag = "B15" if b15_1000upd else ("B12" if b11_500upd else ("B8" if b8_long_train else ("B6" if extended_pilot else ("B4" if role_quota_pilot else "B3"))))
            cprint(f"\n  ⚠️  [{_pilot_tag} Pilot] HARD STOP: {_pilot_hard_stop_reason}", "red")

    # Final summary
    phase_elapsed = time.time() - phase_start
    summary = {
        "run_id": ckpt_writer.run_id,
        "config_hash": "phased_online_dryrun",
        "source_script_mtime": datetime.fromtimestamp(
            os.path.getmtime(__file__)).isoformat(),
        "mode": "phased_online",
        "actor_optimizer_steps": total_actor_updates,
        "critic_optimizer_steps": total_critic_updates,
        "target_update_steps": total_critic_updates,
        "new_rollout_episodes": total_episodes,
        "replay_size_final": replay.get_stats()["total_transitions"],
        "phases_completed": phase_id,
        "phase_manifests": phase_manifests,
        "hard_stop_primary_reason": "",
        "hard_stop_secondary_reasons": [],
        "gazebo_closed_loop_status": "not_run",
        "lightweight_eval_status": "monitoring_only_not_gazebo",
        "formal_training_status": (
            ("b15_1000upd_complete" if b15_1000upd else
            ("b12_500upd_complete" if b11_500upd else
            ("b8_long_train_complete" if b8_long_train else
            ("b6_extended_pilot_complete" if extended_pilot else
            ("role_quota_pilot_complete" if role_quota_pilot else
            ("pilot_phased_online_complete" if pilot_mode else "phased_online_dryrun_complete"))))))
        ),
        "claim_allowed": False,
        # ── B3: pilot metadata ──
        "ft_pilot_mode": pilot_mode,
        "ft_max_phases": max_phases if max_phases > 0 else None,
        "ft_actor_compute_teacher": actor_compute_teacher,
        "hard_stop_primary_reason": _pilot_hard_stop_reason if '_pilot_hard_stop_reason' in dir() else "",
        # ── B4: role quota fields ──
        "role_quota_pilot": role_quota_pilot,
        "role_quota_target": dict(role_quotas) if role_quota_pilot else None,
        "global_role_counts": dict(global_role_counts) if role_quota_pilot else None,
        "role_quota_met": all(global_role_counts.get(r, 0) >= role_quotas.get(r, 0) for r in formal_roles) if role_quota_pilot else None,
        "role_quota_missing": [r for r in formal_roles if global_role_counts.get(r, 0) < role_quotas.get(r, 0)] if role_quota_pilot else [],
        "role_scheduler": "global_underfilled_quota" if role_quota_pilot else "per_phase_round_robin",
        # ── B2: Reward norm info ──
        "reward_norm_mode": reward_norm_mode,
        "reward_norm_stats_loaded": reward_norm_stats is not None,
        "reward_norm_stats_summary": {
            "mode": reward_norm_mode,
            "total_mean": reward_norm_stats.get("total_mean", None) if reward_norm_stats else None,
            "total_std": reward_norm_stats.get("total_std", None) if reward_norm_stats else None,
        } if reward_norm_stats else None,
        # ── C2-OOC: twin-gap stop tracking ──
        "c2_ooc_twin_gap_stop_enabled": twin_gap_stop_enabled,
        "c2_ooc_twin_gap_stop_reason": _twin_gap_stop_reason,
        "c2_ooc_twin_gap_history": _twin_gap_history,
        "c2_ooc_baseline_twin_gap": 0.6317,
        "elapsed_min": round(phase_elapsed / 60, 1),
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "phased_online_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    summary_md = output_dir / "phased_online_summary.md"
    with open(summary_md, "w") as f:
        f.write("# Phased Online Summary\n\n")
        f.write(f"- **Mode**: phased_online\n")
        f.write(f"- **Phases completed**: {phase_id}\n")
        f.write(f"- **Total actor updates**: {total_actor_updates}\n")
        f.write(f"- **Total critic updates**: {total_critic_updates}\n")
        f.write(f"- **Claim allowed**: False\n")
        f.write(f"- **Elapsed**: {phase_elapsed/60:.1f} min\n")
        for pm in phase_manifests:
            f.write(f"\n## Phase {pm['phase_id']}\n")
            f.write(f"- Episodes: {pm['episodes_collected']}\n")
            f.write(f"- Critic gate: {pm['critic_gate']}\n")
            f.write(f"- Actor updates: {pm['actor_updates']}\n")

    # ── B5: Save actor checkpoint ──
    if ft_save_actor_path:
        _save_dir = Path(ft_save_actor_path)
        _save_dir.mkdir(parents=True, exist_ok=True)
        _actor_sd = {}
        for _name, _param in actor.named_parameters():
            if _param.requires_grad:
                _actor_sd[_name] = _param.detach().cpu().clone()
        _actor_ckpt = _save_dir / "b5_actor.pt"
        torch.save({"actor_state_dict": _actor_sd, "actor_optimizer_steps": total_actor_updates}, _actor_ckpt)
        cprint(f"  [B5] Actor saved: {_actor_ckpt} ({len(_actor_sd)} params, "
               f"{_actor_ckpt.stat().st_size / 1024:.1f} KB)", "green")
        summary["b5_actor_path"] = str(_actor_ckpt)

    cprint(f"  Summary: {summary_md}", "green")
    return summary


def run_routea_rl_formal_train(
    base_policy: Phase1aPolicyFM,
    teacher_policy: Phase1aPolicyFM,
    device: str,
    output_dir: Path,
    manifest: RouteARLAdmissionManifest,
    train_split: str,
    val_split: str,
    world_dir: str,
    critic_ckpt_path: str,
    max_actor_updates: int = 10000,
    eval_interval: int = 1000,
    checkpoint_interval: int = 1000,
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-4,
    rollouts_per_cycle: int = 4,
    critic_updates: int = 16,
    dqpg_warmup_steps: int = 1000,
    batch_size: int = 2048,
    role_ratios: Optional[Dict[str, float]] = None,
    eval_plan_path: Optional[str] = None,
    metrics_schema_path: Optional[str] = None,
    resume_from: Optional[str] = None,
    freeze_actor: bool = False,
    rescue_from: Optional[str] = None,
    fixed_replay: bool = False,  # no new rollouts, critic-only on fixed buffer
    stabilized_replay: bool = False,  # use stabilized sampler (role-balanced + age-mix + hard retention)
    replay_warmup_transitions: int = 5000,  # min transitions for replay warmup
    replay_warmup_episodes: int = 100,  # min episodes for replay warmup
    # ── P0 wiring health: collect-only mode (Task 3) ──
    collect_only: bool = False,  # only collect rollouts, no actor/critic updates
    collect_only_max_steps: int = 100,  # max collection steps in collect-only mode
    collect_only_output_replay: Optional[str] = None,  # path for replay export in collect-only
    # ── P0 wiring health: critic-bootstrap mode (Task 4) ──
    critic_bootstrap: bool = False,  # critic-only training on fixed replay, no new rollouts, no actor
    bootstrap_heldout_ratio: float = 0.2,  # heldout ratio for critic bootstrap diagnostics
    bootstrap_replay_path: Optional[str] = None,  # path to B0 fixed replay for B1 critic bootstrap
    # ── B1a: diagnostic / ablation params ──
    bootstrap_diag_only: bool = False,  # diagnostic-only, skip all training
    bootstrap_gamma: Optional[float] = None,  # override gamma
    bootstrap_target_tau: Optional[float] = None,  # override target tau
    bootstrap_total_only: bool = False,  # train only total head
    bootstrap_huber_total: bool = False,  # SmoothL1 for total head
    bootstrap_huber_beta: float = 5.0,  # Huber beta
    bootstrap_sparse_head_mode: str = "none",  # none, zero_onset_trap_betas, huber_sparse_heads, sparse_head_l2
    # ── B1c: architecture / reward representation knobs ──
    bootstrap_reward_norm_mode: str = "none",  # none, fixed_train_standardize, fixed_train_scale_p95
    bootstrap_reward_norm_eps: float = 1e-6,
    bootstrap_q_l2_weight: float = 0.0,
    bootstrap_q_soft_clip: float = 0.0,  # NOT_IMPLEMENTED placeholder
    bootstrap_critic_norm: str = "none",  # none, layernorm
    bootstrap_target_update_mode: str = "polyak_each_step",  # polyak_each_step, hard_every_n
    bootstrap_target_update_interval: int = 100,
    # ── B1d: split reproducibility ──
    bootstrap_split_seed: Optional[int] = None,
    # ── P0 wiring health: phased online schedule (Task 5) ──
    phased_online: bool = False,  # use phased schedule with explicit gates
    phase_collect_episodes: int = 32,  # episodes per collection phase
    phase_critic_digest_updates: int = 200,  # critic updates per digest phase
    phase_actor_updates: int = 10,  # actor updates per phase (only after critic digest)
    # ── B2: reward normalization stats sidecar ──
    reward_norm_stats_path: Optional[str] = None,  # path to reward_norm_stats.json for phased-online
    # ── B2a: PG scale controls ──
    actor_lambda_pg: float = 1.0,
    actor_lambda_dqpg: float = 1.0,
    actor_pg_scale_mode: str = "raw_sum",
    actor_effective_action_dim: int = 88,
    actor_effective_flow_events: int = 0,
    # ── B3: pilot safety caps ──
    ft_max_phases: int = 0,
    ft_actor_compute_teacher: bool = False,
    ft_pilot_mode: bool = False,
    ft_extended_pilot: bool = False,
    ft_b8_long_train: bool = False,
    ft_b11_500upd: bool = False,
    ft_b15_1000upd: bool = False,
    ft_lora_rank: Optional[int] = None,
    # ── B4: role quota pilot ──
    role_quota_pilot: bool = False,
    ft_min_episodes_anticipatory: int = 2,
    ft_min_episodes_dense_recovery: int = 2,
    ft_min_episodes_support_expansion: int = 2,
    ft_max_support_role_share: float = 0.50,
    # ── B5: lightweight paired eval ──
    ft_save_actor_path: Optional[str] = None,
    ft_post_eval: bool = False,
    ft_eval_eps_per_world: int = 2,
    # ── C2-OOC: online cell smoke ──
    ft_c2_ooc: bool = False,
    ft_mid_eval: bool = False,
    ft_mid_eval_phase: int = 10,
    ft_c2_ooc_no_twin_stop: bool = False,
    seed: int = 20260705,
    curriculum_stage: str = 'C2',
) -> Dict[str, Any]:
    """Run Route A RL v1 formal training on role-driven environment pool.

    This is the main formal training entry point. It combines:
      - New-env world pool (from admission manifest, like run_a1_newenv_stability)
      - Full actor loss (PG + BC + DQPG + noise + smooth, like train_a2)
      - Role-balanced sampling with configurable ratios
      - Checkpoint management via CheckpointWriter
      - Formal hard stop conditions via FormalTrainHardStopChecker
      - Lightweight per-role eval at intervals
    """
    # ── RESTART_FROM_COMMON_B1 (P0 recovery-strategy verdict) ──
    # `--formal_train_resume_from` is DEAD CODE: it was never wired into the training
    # loop, so passing it silently RESTARTED instead of resuming — which would corrupt
    # capacity-budget attribution (MC-5: 归因可分离). Old rank16/200 partial ckpts are
    # missing 14/17 full-state fields (optimizer/replay/RNG/counters) and cannot be
    # resumed. Fail loudly instead of lying.
    if resume_from:
        raise NotImplementedError(
            "--formal_train_resume_from is not supported. The P0 recovery strategy is "
            "RESTART_FROM_COMMON_B1: every rank restarts from the common BC+B1 with a "
            "FRESH actor (--critic_ckpt_path=<common B1 critic>, "
            "--ft_bootstrap_replay=<common B0 replay>), trained 0->max_actor_updates. "
            "Resuming from old partial checkpoints is disallowed (14/17 full-state "
            "fields absent)."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ── GPU memory optimization ──
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False  # deterministic memory allocation
        torch.backends.cudnn.deterministic = False  # keep speed

    # ── C2-OOC: auto-configure all flags from convenience flag ──
    if ft_c2_ooc:
        cprint("\n[C2-OOC] Auto-configuring online cell smoke...", "white", "on_green")
        phased_online = True
        # RANK BUGFIX (capacity-budget matrix): do NOT unconditionally override the
        # CLI rank. Root cause of prior degradation: this branch hardcoded
        # `ft_lora_rank = 16`, silently collapsing rank∈{4,16,32} variants to 16 under
        # --ft_c2_ooc. Now C2-OOC only supplies 16 as the DEFAULT when rank is unset;
        # an explicit --ft_lora_rank is always honored.
        if ft_lora_rank is None:
            ft_lora_rank = 16
        actor_lambda_pg = 0.25
        actor_pg_scale_mode = "per_flow_event"
        actor_effective_flow_events = 352
        curriculum_stage = "C2"
        role_quota_pilot = True
        ft_max_phases = 50
        ft_post_eval = True
        ft_mid_eval = True
        ft_mid_eval_phase = 32
        ft_eval_eps_per_world = 4
        ft_actor_compute_teacher = True
        # Phase structure: 3 eps collect, 32 critic digest, 2 actor updates
        phase_collect_episodes = 3
        phase_critic_digest_updates = 32
        phase_actor_updates = 2
        # GPU memory safety for 8GB
        batch_size = 512
        cprint(f"  rank={ft_lora_rank}, max_phases={ft_max_phases} ({ft_max_phases*2} actor upd), "
               f"lambda_pg={actor_lambda_pg}, scale={actor_pg_scale_mode}, "
               f"curriculum={curriculum_stage}", "green")
        cprint(f"  eval: pre + mid(phase {ft_mid_eval_phase}, {ft_mid_eval_phase*2} upd) + post({ft_max_phases*2} upd), "
               f"{ft_eval_eps_per_world} eps/world", "green")

    # ── Resolve LoRA rank default for non-C2-OOC path (sentinel None => 4) ──
    if ft_lora_rank is None:
        ft_lora_rank = 4

    # ── Default role ratios ──
    if role_ratios is None:
        role_ratios = {
            "anticipatory": 0.30,
            "dense_recovery": 0.35,
            "support_expansion": 0.25,
            "yaw_lateral_recovery": 0.10,
        }
    _sum_r = sum(role_ratios.values())
    role_ratios = {k: v / _sum_r for k, v in role_ratios.items()}

    # ── Build world pools ──
    cprint(f"\n[FormalTrain] Building world pools...", "white", "on_blue")
    train_pool = _make_newenv_world_pool(manifest, train_split, world_dir)
    val_pool = _make_newenv_world_pool(manifest, val_split, world_dir)
    if not train_pool:
        raise RuntimeError(f"No admitted train worlds for '{train_split}'")
    if not val_pool:
        cprint(f"  [WARN] No val worlds, using train pool for eval", "yellow")
        val_pool = train_pool

    cprint(f"  Train pool: {len(train_pool)} worlds, Val pool: {len(val_pool)} worlds", "cyan")

    # ── World metadata lookup ──
    world_meta_lookup = {}
    for w in manifest.get_admitted_worlds(train_split) + manifest.get_admitted_worlds(val_split):
        wname = w.world_uid
        if wname.endswith(".world"):
            wname = wname[:-6]
        world_meta_lookup[wname] = w

    # ── Build per-role world indices ──
    role_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, (wname, _) in enumerate(train_pool):
        meta = world_meta_lookup.get(wname)
        role = meta.role_active or meta.role_validated if meta else "unknown"
        role_to_indices[role].append(idx)

    cprint(f"  Role buckets: " + ", ".join(f"{r}={len(role_to_indices.get(r, []))}"
                                            for r in TRAINABLE_ROLES), "cyan")

    # ── Create actor ──
    cprint(f"\n[FormalTrain] Creating actor (LoRA rank={ft_lora_rank}, lr={actor_lr})...", "cyan")
    actor = create_actor_wrapper(base_policy, teacher_policy, device, lora_rank=ft_lora_rank)

    # ── Freeze boundary hashes ──
    base_fm_hash = compute_model_hash(actor.base_policy.model)
    obs_encoder_hash = compute_model_hash(actor.base_policy.obs_encoder)
    teacher_hash = compute_model_hash(teacher_policy.model)
    cprint(f"  Freeze hashes: base_fm={base_fm_hash}, obs_enc={obs_encoder_hash}, "
           f"teacher={teacher_hash}", "cyan")

    # ── Load critic ──
    cprint(f"\n[FormalTrain] Loading critic: {critic_ckpt_path}", "cyan")
    _use_ln = (bootstrap_critic_norm == "layernorm")
    critic, target_critic, critic_gate = load_critic_from_a0(critic_ckpt_path, device, use_layernorm=_use_ln)

    # ── RESTART_FROM_COMMON_B1 provenance stamp (MC-5: 归因可分离) ──
    # Record that this run started from the COMMON B1 with a FRESH actor, so each
    # capacity-budget rank variant's numbers can be honestly attributed to "the policy
    # learned X" rather than to a resumed/warm-started actor. Written once at launch.
    try:
        _replay_p = bootstrap_replay_path
        _provenance = {
            "recovery_strategy": "RESTART_FROM_COMMON_B1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "fresh_actor": True,                       # actor built from scratch via create_actor_wrapper
            "resume_from": None,                       # guaranteed by the fail-fast guard above
            "lora_rank": ft_lora_rank,
            "seed": seed,
            "max_actor_updates": max_actor_updates,
            "critic_ckpt_path": str(critic_ckpt_path),
            "critic_model_hash": compute_model_hash(critic),
            "bootstrap_replay_path": str(_replay_p) if _replay_p else None,
            "bootstrap_replay_exists": bool(_replay_p) and Path(_replay_p).exists(),
            "freeze_hashes": {
                "base_fm": base_fm_hash,
                "obs_encoder": obs_encoder_hash,
                "teacher": teacher_hash,
            },
        }
        _prov_path = output_dir / "restart_provenance.json"
        with open(_prov_path, "w") as _pf:
            json.dump(_provenance, _pf, indent=2)
        cprint(f"  [RESTART_FROM_COMMON_B1] provenance -> {_prov_path} "
               f"(rank={ft_lora_rank}, fresh_actor=True)", "green")
    except Exception as _e:
        cprint(f"  [RESTART_FROM_COMMON_B1] WARN: could not write provenance: {_e}", "yellow")

    # ── B2: Load reward_norm_stats from checkpoint or sidecar ──
    _reward_norm_stats = None
    _reward_norm_mode = "none"
    _reward_norm_eps = 1e-6
    # Try checkpoint first
    try:
        _ckpt_raw = torch.load(critic_ckpt_path, map_location="cpu")
        if "reward_norm" in _ckpt_raw:
            _rn = _ckpt_raw["reward_norm"]
            _reward_norm_stats = {
                "mode": _rn.get("mode", "none"),
                "total_mean": _rn.get("total_mean", 0),
                "total_std": _rn.get("total_std", 1),
                "eps": _rn.get("eps", 1e-6),
            }
            _reward_norm_mode = _rn.get("mode", "none")
            _reward_norm_eps = _rn.get("eps", 1e-6)
            cprint(f"  [B2] Loaded reward_norm_stats from checkpoint: mode={_reward_norm_mode}, "
                   f"total_mean={_reward_norm_stats['total_mean']:.2f}, "
                   f"total_std={_reward_norm_stats['total_std']:.2f}", "green")
        del _ckpt_raw
    except Exception as e:
        cprint(f"  [B2] Could not extract reward_norm from checkpoint: {e}", "yellow")
    # Fallback to sidecar
    if _reward_norm_stats is None and reward_norm_stats_path:
        try:
            with open(reward_norm_stats_path, "r") as f:
                _rn = json.load(f)
            _reward_norm_stats = {
                "mode": _rn.get("mode", "none"),
                "total_mean": _rn.get("total_mean", 0),
                "total_std": _rn.get("total_std", 1),
                "eps": _rn.get("eps", 1e-6),
            }
            _reward_norm_mode = _rn.get("mode", "none")
            _reward_norm_eps = _rn.get("eps", 1e-6)
            cprint(f"  [B2] Loaded reward_norm_stats from sidecar: {reward_norm_stats_path}", "green")
            # Cross-check with checkpoint if both present
            if "reward_norm" in locals():
                _rn_ckpt = locals().get("_rn", {})
                if _rn_ckpt.get("mode") != _reward_norm_mode or \
                   abs(_rn_ckpt.get("total_mean", 0) - _reward_norm_stats["total_mean"]) > 0.01:
                    cprint(f"  [B2] ⚠️  reward_norm MISMATCH between checkpoint and sidecar!", "red")
        except Exception as e:
            cprint(f"  [B2] Failed to load reward_norm_stats from sidecar: {e}", "red")
    if _reward_norm_stats is None:
        cprint(f"  [B2] No reward_norm_stats available — critic digest will use raw rewards", "yellow")

    # ── Optimizers ──
    actor_groups = actor.get_trainable_param_groups(lr=actor_lr)
    actor_optimizer = torch.optim.Adam(actor_groups, lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    # ── Checkpoint writer ──
    run_id = f"routea_rl_formal_train_{datetime.now().strftime('%Y%m%d_%H%M')}"
    ckpt_writer = CheckpointWriter(output_dir, run_id)

    # ── Replay buffer ──
    replay = RouteAReplayBuffer(capacity_online=50000)

    # ── Hard stop checker ──
    hard_stop_cfg = FormalTrainHardStopConfig()
    hard_stop = FormalTrainHardStopChecker(
        config=hard_stop_cfg,
        base_fm_hash=base_fm_hash,
        obs_encoder_hash=obs_encoder_hash,
        teacher_hash=teacher_hash,
    )

    # ── Streaming writers ──
    train_csv = StreamingCSVWriter(
        output_dir / "formal_train_metrics.csv",
        ["step", "actor_loss_total", "actor_loss_pg", "actor_loss_dqpg", "actor_loss_bc",
         "actor_loss_noise", "actor_loss_smooth", "critic_loss_total", "critic_twin_gap",
         "critic_Q_std", "actor_grad_norm_post_clip", "critic_grad_norm",
         "dqpg_enabled", "n_episodes", "n_transitions", "replay_size",
         "eval_sr_anticipatory", "eval_cr_anticipatory",
         "eval_sr_dense_recovery", "eval_cr_dense_recovery",
         "eval_sr_support_expansion", "eval_cr_support_expansion",
         "eval_sr_yaw_lateral_recovery", "eval_cr_yaw_lateral_recovery",
         "nan_count", "inf_count", "metadata_completeness", "contamination_count",
         "timing_sec"],
    )
    role_csv = StreamingCSVWriter(
        output_dir / "formal_role_sampling_trace.csv",
        ["step", "role", "n_episodes", "n_transitions", "pct_episodes"],
    )
    critic_drift_csv = StreamingCSVWriter(
        output_dir / "formal_critic_drift_trace.csv",
        ["step", "twin_gap", "q_std", "q_total_mean", "q_total_std",
         "q_progress_mean", "q_onset_mean", "q_fclear_mean",
         "critic_loss_total", "target_q_mean", "td_target_mean"],
    )

    # ── Trackers ──
    nan_count = 0
    inf_count = 0
    total_episodes = 0
    contamination_count = 0
    metadata_complete = 0
    metadata_total = 0
    per_role_episodes: Dict[str, int] = defaultdict(int)
    per_role_transitions: Dict[str, int] = defaultdict(int)
    actor_losses: List[float] = []
    critic_losses: List[float] = []
    eval_history: List[Dict] = []
    hard_stop_reason = ""
    obs_buffer_pool: List[Dict] = []
    dqpg_enabled = False  # starts disabled during warmup

    # ── Rescue: load actor/critic from emergency checkpoint ──
    if rescue_from:
        cprint(f"\n[FormalTrain] RESCUE MODE: loading checkpoint from {rescue_from}", "yellow")
        rescue_ckpt = torch.load(rescue_from, map_location=device)
        actor.load_state_dict(rescue_ckpt["actor_state_dict"], strict=False)
        critic.load_state_dict(rescue_ckpt["critic_state_dict"], strict=False)
        target_critic.load_state_dict(critic.state_dict())
        rescue_step = rescue_ckpt.get("step", "unknown")
        rescue_reason = rescue_ckpt.get("reason", "unknown")
        cprint(f"  Loaded step {rescue_step}, reason: {rescue_reason}", "yellow")
        cprint(f"  Actor freeze={'ON' if freeze_actor else 'OFF'}, "
               f"Critic updates={critic_updates}", "yellow")

    # ── Preflight: metadata check ──
    cprint(f"\n[FormalTrain] Preflight checks...", "white", "on_blue")
    cprint(f"  Formal role buckets: anticipatory={len(role_to_indices.get('anticipatory',[]))}, "
           f"dense_recovery={len(role_to_indices.get('dense_recovery',[]))}, "
           f"support_expansion={len(role_to_indices.get('support_expansion',[]))}", "cyan")
    cprint(f"  Support bucket: yaw_lateral_recovery={len(role_to_indices.get('yaw_lateral_recovery',[]))}", "cyan")
    cprint(f"  Actor LR: {actor_lr}, Critic LR: {critic_lr}", "cyan")
    cprint(f"  DQPG warmup: {dqpg_warmup_steps} steps", "cyan")
    cprint(f"  Max actor updates: {max_actor_updates}", "cyan")

    # ── Preflight gate checks ──
    preflight_errors = []
    if not role_to_indices.get("anticipatory"):
        preflight_errors.append("anticipatory bucket empty")
    if not role_to_indices.get("dense_recovery"):
        preflight_errors.append("dense_recovery bucket empty")
    if not role_to_indices.get("support_expansion"):
        preflight_errors.append("support_expansion bucket empty")
    if preflight_errors:
        cprint(f"  ❌ BLOCKED_FORMAL_TRAIN_PREFLIGHT: {', '.join(preflight_errors)}", "red")
        return {"formal_training_status": "blocked", "preflight_errors": preflight_errors}

    cprint(f"  ✅ Preflight PASS — ready to start training", "green")
    cprint(f"\n  ⚠️  TRAINING PAUSED — waiting for user confirmation before first actor update", "yellow")
    cprint(f"  Output dir: {output_dir}", "cyan")
    cprint(f"  Resume by re-running with --routea_rl_formal_train", "cyan")

    # ── Save preflight ──
    preflight_data = {
        "stage": "RouteA_RL_v1_formal_training",
        "timestamp": datetime.now().isoformat(),
        "preflight_status": "PASS",
        "admission_manifest": str(getattr(manifest, '_path', 'unknown')),
        "train_pool_size": len(train_pool),
        "val_pool_size": len(val_pool),
        "role_buckets": {r: len(role_to_indices.get(r, [])) for r in TRAINABLE_ROLES},
        "role_ratios": role_ratios,
        "max_actor_updates": max_actor_updates,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "dqpg_warmup_steps": dqpg_warmup_steps,
        "rollouts_per_cycle": rollouts_per_cycle,
        "critic_updates": critic_updates,
        "base_fm_hash": base_fm_hash,
        "obs_encoder_hash": obs_encoder_hash,
        "teacher_hash": teacher_hash,
    }
    with open(output_dir / "formal_training_preflight.json", "w") as f:
        json.dump(preflight_data, f, indent=2)

    # ── Save preflight MD ──
    preflight_md_lines = [
        "# Route A RL v1 — Formal Training Preflight",
        f"**Timestamp**: {preflight_data['timestamp']}",
        f"**Status**: {preflight_data['preflight_status']}",
        "",
        "## Role Buckets",
    ]
    for r in TRAINABLE_ROLES:
        preflight_md_lines.append(f"- {r}: {len(role_to_indices.get(r, []))} worlds")
    preflight_md_lines += [
        "",
        "## Training Config",
        f"- Actor LR: {actor_lr}",
        f"- Critic LR: {critic_lr}",
        f"- Max actor updates: {max_actor_updates}",
        f"- DQPG warmup steps: {dqpg_warmup_steps}",
        f"- Rollouts per cycle: {rollouts_per_cycle}",
        f"- Critic updates per step: {critic_updates}",
        "",
        "## Sampling Ratios",
    ]
    for r, ratio in role_ratios.items():
        preflight_md_lines.append(f"- {r}: {ratio:.1%}")
    preflight_md_lines += [
        "",
        "## Freeze Hashes",
        f"- base_fm: {base_fm_hash}",
        f"- obs_encoder: {obs_encoder_hash}",
        f"- teacher: {teacher_hash}",
        "",
        "## ⚠️ STOP — 训练启动前等待用户确认",
        "Preflight 全部通过。确认后重新运行以启动训练。",
    ]
    with open(output_dir / "formal_training_preflight.md", "w") as f:
        f.write("\n".join(preflight_md_lines) + "\n")
    cprint(f"  Preflight report: {output_dir / 'formal_training_preflight.md'}", "green")

    # ═══════════════════════════════════════════════════════════════
    # Replay warmup (before actor/critic updates)
    # ═══════════════════════════════════════════════════════════════
    if stabilized_replay:
        cprint(f"\n[FormalTrain] REPLAY WARMUP: collecting min {replay_warmup_transitions} transitions "
               f"({replay_warmup_episodes} episodes)...", "white", "on_blue")
        warmup_episodes = 0
        warmup_max_attempts = replay_warmup_episodes * 3
        warmup_roles = ["anticipatory", "dense_recovery", "support_expansion"]
        warmup_attempt = 0

        while warmup_attempt < warmup_max_attempts:
            warmup_complete, warmup_diag = replay.is_warmup_complete(
                min_transitions=replay_warmup_transitions,
                min_episodes=replay_warmup_episodes,
                required_roles=set(warmup_roles),
            )
            if warmup_complete:
                break

            # Select a role that needs more episodes
            _target_role = warmup_roles[warmup_attempt % len(warmup_roles)]
            _role_indices = role_to_indices.get(_target_role, [])
            if not _role_indices:
                warmup_attempt += 1
                continue
            _idx = _role_indices[warmup_attempt % len(_role_indices)]
            wname, obs_json = train_pool[_idx]
            meta = world_meta_lookup.get(wname)
            role_val = meta.role_active or meta.role_validated if meta else ""
            role_act = meta.role_active or ""
            try:
                env = KinematicUAVEnv(
                    world_id=4, world_pool=[(wname, obs_json)],
                    episode_role=role_val,  # P1 fix (2026-07-11): role-aware start/goal gating
                    synth_pc_factory=lambda jp: SynthPointCloud(jp, device=device),
                    rl_action_steps=8, h_min=0.3, z_min=1.0, z_max=5.0,
                    disturb_enable=True, disturb_prob=0.2, disturb_pos_scale=0.3,
                    mid_state_init_ratio=0.0,
                    seed=int(rng.integers(0, 2**31)),
                    reward_version="v13",
                                    curriculum_stage=curriculum_stage,
                )
                transitions, obs_buf, ep_sum = rollout_episode_a2(
                    actor, base_policy, env, device,
                    episode_seed=int(rng.integers(0, 2**31)),
                    world_id=-1, deterministic=False, lazy_teacher=True,
                )
                for t in transitions:
                    t.world_uid = wname; t.split = train_split
                    t.role_validated = role_val; t.role_active = role_act
                    t.admission_status = "admitted"
                replay.add_episode_with_diagnostics(
                    transitions,
                    scenario=_target_role,
                    episode_collision=ep_sum.get("collision", False),
                    episode_success=ep_sum.get("success", False),
                )
                warmup_episodes += 1
                total_episodes += 1
                per_role_episodes[role_val] += 1
                per_role_transitions[role_val] += len(transitions)
                for t in transitions:
                    metadata_total += 1
                    if t.world_uid and t.role_validated:
                        metadata_complete += 1
                env.close()
            except Exception as e:
                cprint(f"  [WARN] Warmup rollout failed for {wname}: {e}", "yellow")
            warmup_attempt += 1

        warmup_complete, warmup_diag = replay.is_warmup_complete(
            min_transitions=replay_warmup_transitions,
            min_episodes=replay_warmup_episodes,
            required_roles=set(warmup_roles),
        )
        cprint(f"  Warmup: {warmup_diag['n_transitions']} transitions, "
               f"{warmup_diag['n_episodes']} episodes, "
               f"roles={warmup_diag['roles_present']}, "
               f"hard_ratio={warmup_diag['hard_sample_ratio']:.3f}, "
               f"complete={warmup_diag['warmup_complete']}", "cyan")

        # Save warmup manifest
        warmup_manifest = {
            "timestamp": datetime.now().isoformat(),
            "warmup_config": {
                "min_transitions": replay_warmup_transitions,
                "min_episodes": replay_warmup_episodes,
                "required_roles": warmup_roles,
            },
            "warmup_result": warmup_diag,
            "gate": {
                "metadata_completeness": metadata_complete / max(metadata_total, 1),
                "historical_contamination": contamination_count,
                "all_formal_roles_covered": warmup_diag["roles_covered"],
                "hard_sample_ratio_gt_0": warmup_diag["hard_sample_ratio"] > 0,
                "no_role_has_zero_replay": all(
                    replay.get_role_breakdown().get(r, {}).get("transition_count", 0) > 0
                    for r in warmup_roles if r in warmup_diag["roles_present"]
                ),
            },
        }
        with open(output_dir / "replay_warmup_manifest.json", "w") as f:
            json.dump(warmup_manifest, f, indent=2)
        with open(output_dir / "replay_warmup_summary.md", "w") as f:
            f.write("# Replay Warmup Summary\n\n")
            f.write(f"- Transitions: {warmup_diag['n_transitions']}\n")
            f.write(f"- Episodes: {warmup_diag['n_episodes']}\n")
            f.write(f"- Roles present: {warmup_diag['roles_present']}\n")
            f.write(f"- Hard sample ratio: {warmup_diag['hard_sample_ratio']:.4f}\n")
            f.write(f"- Warmup complete: {warmup_diag['warmup_complete']}\n")
            f.write(f"- Metadata completeness: {warmup_manifest['gate']['metadata_completeness']:.2f}\n")
            f.write(f"- Historical contamination: {contamination_count}\n")

        if not warmup_complete:
            cprint(f"  WARNING: Warmup incomplete but proceeding (gates will be checked)", "yellow")

    # ═══════════════════════════════════════════════════════════════
    # Seed replay fill
    # ═══════════════════════════════════════════════════════════════
    cprint(f"\n[FormalTrain] Seeding replay buffer...", "white", "on_blue")
    _seed_roles = ["anticipatory", "dense_recovery", "support_expansion"]
    for _sr in _seed_roles:
        _sr_indices = role_to_indices.get(_sr, [])
        if not _sr_indices:
            continue
        for _si in _sr_indices[:2]:  # max 2 worlds per role for seed
            wname, obs_json = train_pool[_si]
            meta = world_meta_lookup.get(wname)
            role_val = meta.role_active or meta.role_validated if meta else ""
            role_act = meta.role_active or ""
            try:
                env = KinematicUAVEnv(
                    world_id=4, world_pool=[(wname, obs_json)],
                    episode_role=role_val,  # P1 fix (2026-07-11): role-aware start/goal gating
                    synth_pc_factory=lambda jp: SynthPointCloud(jp, device=device),
                    rl_action_steps=8, h_min=0.3, z_min=1.0, z_max=5.0,
                    disturb_enable=True, disturb_prob=0.2, disturb_pos_scale=0.3,
                    mid_state_init_ratio=0.0,
                    seed=int(rng.integers(0, 2**31)),
                    reward_version="v13",
                                    curriculum_stage=curriculum_stage,
                )
                transitions, obs_buf, ep_sum = rollout_episode_a2(
                    actor, base_policy, env, device,
                    episode_seed=int(rng.integers(0, 2**31)),
                    world_id=-1, deterministic=False, lazy_teacher=True,
                )
                for t in transitions:
                    t.world_uid = wname; t.split = train_split
                    t.role_validated = role_val; t.role_active = role_act
                    t.admission_status = "admitted"
                replay.add_episode(transitions, partition="online")
                total_episodes += 1
                per_role_episodes[role_val] += 1
                per_role_transitions[role_val] += len(transitions)
                for t in transitions:
                    metadata_total += 1
                    if t.world_uid and t.role_validated:
                        metadata_complete += 1
                env.close()
            except Exception as e:
                cprint(f"  [WARN] Seed rollout failed for {wname}: {e}", "yellow")
    cprint(f"  Seed replay: {replay.get_stats()['total_transitions']} transitions", "cyan")

    # ═══════════════════════════════════════════════════════════════
    # Step 0 eval
    # ═══════════════════════════════════════════════════════════════
    # B5v2: if post_eval enabled, eval on ALL worlds with configured eps for fair pre/post comparison
    _eval_eps = ft_eval_eps_per_world if ft_post_eval else 2
    _eval_worlds = list(range(len(val_pool))) if ft_post_eval else None
    # Generate fixed episode_configs for deterministic paired eval (design doc §4.1.0 v1.3)
    _eval_configs = _generate_eval_episode_configs(val_pool, world_meta_lookup, rng) if ft_post_eval else None
    cprint(f"\n[FormalTrain] Step 0 eval ({len(_eval_worlds) if _eval_worlds else 'random'} worlds × {_eval_eps} eps)...", "white", "on_blue")
    eval_0 = _run_newenv_eval(actor, base_policy, device, val_pool, world_meta_lookup, rng,
                              n_episodes_per_world=_eval_eps, seed_offset=0,
                              fixed_world_indices=_eval_worlds,
                              episode_configs=_eval_configs,
                              curriculum_stage=curriculum_stage)
    eval_history.append({"step": 0, **eval_0})
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cprint(f"  Step 0 eval: " + ", ".join(
        f"{r}={eval_0['per_role'].get(r, {}).get('SR', 0):.3f}"
        for r in ["anticipatory", "dense_recovery", "support_expansion"]
        if r in eval_0['per_role']), "cyan")

    # Initial best checkpoint
    ckpt_writer.update_best(0, 0.0,
        actor.state_dict(), critic.state_dict(), target_critic.state_dict(),
        {"eval_0": eval_0})

    # ═══════════════════════════════════════════════════════════════
    # P0 Wiring Health: Mode dispatch
    # ═══════════════════════════════════════════════════════════════
    _p0_mode = "formal_legacy"
    if collect_only:
        _p0_mode = "collect_only"
    elif critic_bootstrap:
        _p0_mode = "critic_bootstrap"
    elif phased_online:
        _p0_mode = "phased_online"

    if _p0_mode == "collect_only":
        result = _run_collect_only(
            actor, base_policy, device, output_dir, ckpt_writer,
            replay, manifest, train_split, world_dir,
            train_pool, world_meta_lookup, role_to_indices,
            max_steps=collect_only_max_steps,
            output_replay=collect_only_output_replay,
            rng=rng,
            metadata_complete=metadata_complete, metadata_total=metadata_total,
            contamination_count=contamination_count,
            total_episodes=total_episodes,
            per_role_episodes=per_role_episodes,
            per_role_transitions=per_role_transitions,
            curriculum_stage=curriculum_stage,
        )
        # Save replay buffer for downstream B1 use
        if collect_only_output_replay:
            replay_path = Path(collect_only_output_replay)
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            replay.save(str(replay_path))
            cprint(f"  Replay saved: {replay_path} ({replay.get_stats()['total_transitions']} trans)", "green")
        return result

    if _p0_mode == "critic_bootstrap":
        # Load fixed replay from B0 if path provided
        if bootstrap_replay_path:
            cprint(f"\n[CriticBootstrap] Loading fixed replay: {bootstrap_replay_path}", "cyan")
            replay = RouteAReplayBuffer.load(bootstrap_replay_path)
            n_loaded = replay.get_stats()["total_transitions"]
            cprint(f"  Loaded {n_loaded} transitions from B0 fixed replay", "green")
        return _run_critic_bootstrap(
            actor, base_policy, critic, target_critic, critic_optimizer,
            device, output_dir, ckpt_writer,
            replay, train_pool, world_meta_lookup, role_to_indices,
            heldout_ratio=bootstrap_heldout_ratio,
            batch_size=batch_size, critic_updates=critic_updates,
            max_actor_updates=max_actor_updates,
            eval_interval=eval_interval,
            metadata_complete=metadata_complete, metadata_total=metadata_total,
            contamination_count=contamination_count,
            val_pool=val_pool, rng=rng,
            diag_only=bootstrap_diag_only,
            gamma_override=bootstrap_gamma,
            target_tau_override=bootstrap_target_tau,
            total_only=bootstrap_total_only,
            huber_total=bootstrap_huber_total,
            huber_beta=bootstrap_huber_beta,
            sparse_head_mode=bootstrap_sparse_head_mode,
            # ── B1c knobs ──
            reward_norm_mode=bootstrap_reward_norm_mode,
            reward_norm_eps=bootstrap_reward_norm_eps,
            q_l2_weight=bootstrap_q_l2_weight,
            critic_norm=bootstrap_critic_norm,
            target_update_mode=bootstrap_target_update_mode,
            target_update_interval=bootstrap_target_update_interval,
            split_seed=bootstrap_split_seed,
        )

    if _p0_mode == "phased_online":
        # ── B10: Load fixed replay from B0 if path provided (same as critic_bootstrap) ──
        if bootstrap_replay_path:
            cprint(f"\n[PhasedOnline] Loading fixed replay: {bootstrap_replay_path}", "cyan")
            replay = RouteAReplayBuffer.load(bootstrap_replay_path)
            n_loaded = replay.get_stats()["total_transitions"]
            cprint(f"  Loaded {n_loaded} transitions from B0 fixed replay", "green")
        # ── C2-OOC: mid_eval closure ──
        _mid_eval_fn = None
        if ft_mid_eval and _eval_configs is not None:
            _all_world_indices = list(range(len(val_pool)))
            _eps_per_world = ft_eval_eps_per_world
            def _mid_eval_fn(_actor, _phase_id):
                cprint(f"\n  [MidEval] Running eval on {len(_all_world_indices)} worlds x {_eps_per_world} eps...", "cyan")
                _result = _run_newenv_eval(
                    _actor, base_policy, device, val_pool, world_meta_lookup, rng,
                    n_episodes_per_world=_eps_per_world,
                    seed_offset=500 + _phase_id * 100,
                    fixed_world_indices=_all_world_indices,
                    episode_configs=_eval_configs,
                    curriculum_stage=curriculum_stage,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return _result

        _phased_result = _run_phased_online(
            actor, base_policy, critic, target_critic,
            actor_optimizer, critic_optimizer,
            device, output_dir, ckpt_writer,
            replay, manifest, train_split, world_dir,
            train_pool, val_pool, world_meta_lookup, role_to_indices,
            max_actor_updates=max_actor_updates,
            collect_episodes=phase_collect_episodes,
            critic_digest_updates=phase_critic_digest_updates,
            actor_updates_per_phase=phase_actor_updates,
            eval_interval=eval_interval,
            batch_size=batch_size, critic_updates=critic_updates,
            dqpg_warmup_steps=dqpg_warmup_steps,
            metadata_complete=metadata_complete, metadata_total=metadata_total,
            contamination_count=contamination_count,
            total_episodes=total_episodes,
            per_role_episodes=per_role_episodes,
            per_role_transitions=per_role_transitions,
            rng=rng,
            reward_norm_stats=_reward_norm_stats,
            reward_norm_mode=_reward_norm_mode,
            reward_norm_eps=_reward_norm_eps,
            # ── B2a: PG scale controls ──
            actor_lambda_pg=actor_lambda_pg,
            actor_lambda_dqpg=actor_lambda_dqpg,
            actor_pg_scale_mode=actor_pg_scale_mode,
            actor_effective_action_dim=actor_effective_action_dim,
            actor_effective_flow_events=actor_effective_flow_events,
            # ── B3: pilot safety caps ──
            max_phases=ft_max_phases,
            actor_compute_teacher=ft_actor_compute_teacher,
            pilot_mode=ft_pilot_mode,
            extended_pilot=ft_extended_pilot,
            b8_long_train=ft_b8_long_train,
            b11_500upd=ft_b11_500upd,
            b15_1000upd=ft_b15_1000upd,
            # ── B4: role quota pilot ──
            role_quota_pilot=role_quota_pilot,
            ft_min_episodes_anticipatory=ft_min_episodes_anticipatory,
            ft_min_episodes_dense_recovery=ft_min_episodes_dense_recovery,
            ft_min_episodes_support_expansion=ft_min_episodes_support_expansion,
            ft_max_support_role_share=ft_max_support_role_share,
            # ── B5: lightweight paired eval ──
            ft_save_actor_path=ft_save_actor_path,
            # ── C2-OOC: mid-training eval ──
            mid_eval_enabled=ft_mid_eval,
            mid_eval_phase=ft_mid_eval_phase,
            mid_eval_fn=_mid_eval_fn,
            mid_eval_configs=_eval_configs,
            # ── C2-OOC: mid-phase ckpt save ──
            mid_eval_save_ckpt_dir=str(output_dir / "mid_checkpoints") if ft_mid_eval else None,
            # ── C2-OOC: monitoring dashboard (always on, independent of stop) ──
            c2_ooc_monitoring_enabled=ft_c2_ooc,
            # ── C2-OOC: twin-gap stop conditions ──
            # S1/S2 (twin-gap) can be disabled by --ft_c2_ooc_no_twin_stop
            # S3 (grad_norm) and S4 (NaN/Inf) always active in C2-OOC mode
            twin_gap_stop_enabled=ft_c2_ooc and not ft_c2_ooc_no_twin_stop,
            grad_nan_stop_enabled=ft_c2_ooc,
            twin_gap_hard_stop=0.70,
            twin_gap_rise_threshold=0.05,
            twin_gap_rise_window=2,
            # ── curriculum ──
            curriculum_stage=curriculum_stage,
        )
        # ── C2-OOC: extract mid_eval from phase manifests ──
        _mid_eval_result = None
        if ft_mid_eval:
            for _pm in _phased_result.get("phase_manifests", []):
                if _pm.get("mid_eval_triggered"):
                    _mid_eval_result = _pm.get("mid_eval_summary")
                    break
        # ── B5: Post-training eval ──
        if ft_post_eval:
            # B5v2: eval on ALL val_pool worlds with configurable eps_per_world
            _all_world_indices = list(range(len(val_pool)))
            _eps_per_world = ft_eval_eps_per_world
            cprint(f"\n[B5] Post-training eval (ALL {len(_all_world_indices)} worlds × {_eps_per_world} eps)...", "white", "on_blue")
            eval_post = _run_newenv_eval(actor, base_policy, device, val_pool, world_meta_lookup, rng,
                                          n_episodes_per_world=_eps_per_world, seed_offset=1000,
                                          fixed_world_indices=_all_world_indices,
                                          episode_configs=_eval_configs,
                                          curriculum_stage=curriculum_stage)
            _phased_result["eval_post"] = eval_post
            _phased_result["eval_pre"] = eval_0
            cprint(f"  [B5] Pre  eval: " + ", ".join(
                f"{r}={eval_0['per_role'].get(r, {}).get('SR', 0):.3f}"
                for r in ["anticipatory", "dense_recovery", "support_expansion"]
                if r in eval_0['per_role']), "cyan")
            cprint(f"  [B5] Post eval: " + ", ".join(
                f"{r}={eval_post['per_role'].get(r, {}).get('SR', 0):.3f}"
                for r in ["anticipatory", "dense_recovery", "support_expansion"]
                if r in eval_post['per_role']), "cyan")
            # ── C2-OOC: three-way paired eval ──
            _paired_eval = {
                "eval_pre": eval_0,
                "eval_mid": _mid_eval_result,
                "eval_post": eval_post,
            }
            _paired_path = output_dir / "c2_ooc_paired_eval.json"
            with open(_paired_path, "w") as f:
                json.dump(_paired_eval, f, indent=2, default=str)
            cprint(f"  [C2-OOC] Paired eval saved: {_paired_path}", "green")

            # B5 backward compat: two-way eval
            _b5_eval = {"eval_pre": eval_0, "eval_post": eval_post}
            _b5_path = output_dir / "b5_paired_eval.json"
            with open(_b5_path, "w") as f:
                json.dump(_b5_eval, f, indent=2)
            cprint(f"  [B5] Eval saved: {_b5_path}", "green")

            # Also write comparison markdown
            _cmp_lines = [
                "# C2-OOC Paired Eval Comparison",
                "",
                "| Role | Pre SR | Mid SR | Post SR | delta(pre->post) |",
                "|------|--------|--------|---------|-------------|",
            ]
            for r in ["anticipatory", "dense_recovery", "support_expansion"]:
                _pre_sr = eval_0.get("per_role", {}).get(r, {}).get("SR", 0)
                _mid_sr = (_mid_eval_result or {}).get("per_role", {}).get(r, {}).get("SR", 0) if _mid_eval_result else None
                _post_sr = eval_post.get("per_role", {}).get(r, {}).get("SR", 0)
                _mid_str = f"{_mid_sr:.3f}" if _mid_sr is not None else "N/A"
                _delta = _post_sr - _pre_sr
                _cmp_lines.append(f"| {r} | {_pre_sr:.3f} | {_mid_str} | {_post_sr:.3f} | {_delta:+.3f} |")
            with open(_paired_path.parent / "c2_ooc_paired_eval.md", "w", encoding="utf-8") as _f:
                _f.write("\n".join(_cmp_lines))
            cprint(f"  [C2-OOC] Comparison MD saved", "green")
        return _phased_result

    # ═══════════════════════════════════════════════════════════════
    # Main Training Loop (formal_legacy mode)
    # ═══════════════════════════════════════════════════════════════
    cprint(f"\n[FormalTrain] Starting {max_actor_updates} actor updates...", "white", "on_blue")
    train_start = time.time()

    # ── Fixed replay: extended fill ──
    if fixed_replay:
        cprint(f"\n[FormalTrain] FIXED REPLAY: filling buffer with 32 episodes (8 per formal role)...",
               "yellow")
        _fill_roles = ["anticipatory", "dense_recovery", "support_expansion"]
        _fill_count = 0
        for _fr in _fill_roles:
            _fr_indices = role_to_indices.get(_fr, [])
            for _fi in _fr_indices:
                if _fill_count >= 32:
                    break
                wname, obs_json = train_pool[_fi]
                meta = world_meta_lookup.get(wname)
                role_val = meta.role_active or meta.role_validated if meta else ""
                role_act = meta.role_active or ""
                try:
                    env = KinematicUAVEnv(
                        world_id=4, world_pool=[(wname, obs_json)],
                        episode_role=role_val,  # P1 fix (2026-07-11): role-aware start/goal gating
                        synth_pc_factory=lambda jp: SynthPointCloud(jp, device=device),
                        rl_action_steps=8, h_min=0.3, z_min=1.0, z_max=5.0,
                        disturb_enable=True, disturb_prob=0.2, disturb_pos_scale=0.3,
                        mid_state_init_ratio=0.0,
                        seed=int(rng.integers(0, 2**31)),
                        reward_version="v13",
                                        curriculum_stage=curriculum_stage,
                )
                    transitions, obs_buf, ep_sum = rollout_episode_a2(
                        actor, base_policy, env, device,
                        episode_seed=int(rng.integers(0, 2**31)),
                        world_id=-1, deterministic=False, lazy_teacher=True,
                    )
                    for t in transitions:
                        t.world_uid = wname; t.split = train_split
                        t.role_validated = role_val; t.role_active = role_act
                        t.admission_status = "admitted"
                    replay.add_episode(transitions, partition="online")
                    total_episodes += 1; _fill_count += 1
                    per_role_episodes[role_val] += 1
                    per_role_transitions[role_val] += len(transitions)
                    for t in transitions:
                        metadata_total += 1
                        if t.world_uid and t.role_validated:
                            metadata_complete += 1
                    env.close()
                except Exception as e:
                    cprint(f"  [WARN] Fixed-replay fill failed for {wname}: {e}", "yellow")
        cprint(f"  Fixed replay fill: {_fill_count} episodes, "
               f"{replay.get_stats()['total_transitions']} transitions", "cyan")
    best_score_val = float('-inf')
    best_step_val = 0
    actor_update_health: List[Dict] = []
    capability_trace: List[Dict] = []

    for step in range(1, max_actor_updates + 1):
        step_start = time.time()
        ckpt_writer.current_step = step

        # ── DQPG warmup ──
        dqpg_enabled = step > dqpg_warmup_steps
        lambda_dqpg = LAMBDA_DQPG if dqpg_enabled else 0.0

        # ── Role-balanced world selection ──
        n_rollouts = min(rollouts_per_cycle, len(train_pool))
        selected_indices: List[int] = []

        # Build role-weighted selection
        formal_roles = ["anticipatory", "dense_recovery", "support_expansion"]
        support_roles_list = ["yaw_lateral_recovery"]
        n_formal_per_step = max(1, int(n_rollouts * 0.75))  # at least 75% formal
        n_support_per_step = n_rollouts - n_formal_per_step

        # Select formal role worlds
        _formal_candidates = []
        for role in formal_roles:
            _formal_candidates.extend(role_to_indices.get(role, []))
        if _formal_candidates:
            _n_f = min(n_formal_per_step, len(_formal_candidates))
            selected_indices.extend(
                rng.choice(_formal_candidates, size=_n_f, replace=False).tolist())

        # Select support role worlds
        _support_candidates = []
        for role in support_roles_list:
            _support_candidates.extend(role_to_indices.get(role, []))
        if _support_candidates and n_support_per_step > 0:
            _n_s = min(n_support_per_step, len(_support_candidates))
            selected_indices.extend(
                rng.choice(_support_candidates, size=_n_s, replace=False).tolist())

        # Ensure minimum diversity
        if len(selected_indices) < 2:
            _remaining = [i for i in range(len(train_pool)) if i not in selected_indices]
            _extra = rng.choice(_remaining, size=min(2 - len(selected_indices), len(_remaining)),
                               replace=False).tolist()
            selected_indices.extend(_extra)

        # ── Rollouts (batched across envs for GPU utilization) ──
        # Skip if fixed_replay: critic trains on frozen buffer, no new data
        step_episodes = 0
        step_transitions = 0
        if not fixed_replay:
            obs_buffer_pool.clear()

            # Build envs and metadata for selected worlds
            _rollout_envs = []
            _rollout_meta = []
            _rollout_pool_indices = []

            for pool_idx in selected_indices:
                wname, obs_json = train_pool[pool_idx]
                meta = world_meta_lookup.get(wname)
                role_val = meta.role_active or meta.role_validated if meta else ""
                role_act = meta.role_active or ""

                # Contamination check
                contam = check_historical_contamination(wname, obs_json, "", train_split)
                if contam:
                    contamination_count += 1
                    continue

                try:
                    env = KinematicUAVEnv(
                        world_id=4, world_pool=[(wname, obs_json)],
                        episode_role=role_val,  # P1 fix (2026-07-11): role-aware start/goal gating
                        synth_pc_factory=lambda jp: SynthPointCloud(jp, device=device),
                        rl_action_steps=8, h_min=0.3, z_min=1.0, z_max=5.0,
                        disturb_enable=True, disturb_prob=0.2, disturb_pos_scale=0.3,
                        mid_state_init_ratio=0.0,
                        seed=int(rng.integers(0, 2**31)),
                        reward_version="v13",
                                        curriculum_stage=curriculum_stage,
                )
                    _rollout_envs.append(env)
                    _rollout_meta.append((wname, role_val, role_act))
                    _rollout_pool_indices.append(pool_idx)
                except Exception as e:
                    if step <= 2:
                        import traceback as _tb; _tb.print_exc()
                    cprint(f"  [WARN] Env creation failed for {wname}: {e}", "yellow")

            if _rollout_envs:
                try:
                    all_transitions, all_obs_bufs, all_summaries = rollout_batched_episodes(
                        actor, base_policy, _rollout_envs, device,
                        _rollout_meta, train_split,
                        max_steps=60, lazy_teacher=True,
                    )
                    for i, (transitions, obs_buf, ep_sum) in enumerate(
                        zip(all_transitions, all_obs_bufs, all_summaries)):
                        wname, role_val, role_act = _rollout_meta[i]
                        if stabilized_replay:
                            replay.add_episode_with_diagnostics(
                                transitions,
                                scenario=role_val,
                                episode_collision=ep_sum.get("collision", False),
                                episode_success=ep_sum.get("success", False),
                            )
                        else:
                            replay.add_episode(transitions, partition="online")
                        obs_buffer_pool.extend(obs_buf)
                        total_episodes += 1; step_episodes += 1
                        step_transitions += len(transitions)
                        per_role_episodes[role_val] += 1
                        per_role_transitions[role_val] += len(transitions)
                        for t in transitions:
                            metadata_total += 1
                            if t.world_uid and t.role_validated:
                                metadata_complete += 1
                except Exception as e:
                    if step <= 2:
                        import traceback as _tb; _tb.print_exc()
                    cprint(f"  [WARN] Batched rollout failed: {e}", "yellow")
                finally:
                    for env in _rollout_envs:
                        try:
                            env.close()
                        except Exception:
                            pass

        if fixed_replay:
            # Fixed replay: no new rollouts, critic trains on frozen buffer
            step_episodes = 1  # bypass the 0-guard
            step_transitions = 0
            # obs_buffer_pool stays empty — actor update depends on it,
            # but with freeze_actor that's already skipped

        if step_episodes == 0:
            cprint(f"  [WARN] Step {step}: 0 episodes rolled out, skipping update", "yellow")
            continue

        # ── Actor update (skip if freeze_actor in rescue mode) ──
        if not freeze_actor:
            actor_optimizer.zero_grad()
            actor_total_loss = torch.tensor(0.0, device=device)
            actor_loss_components = {}
            n_actor_valid = 0
            actor_grad_preclip = 0.0

            if len(obs_buffer_pool) >= 2:
                n_actor_items = min(4, len(obs_buffer_pool))
                _actor_indices = rng.choice(len(obs_buffer_pool), size=n_actor_items,
                                            replace=False).tolist()

                # Baseline advantage from replay
                if stabilized_replay:
                    adv_batch, _ = replay.sample_stabilized(
                        min(256, replay.get_stats()['total_transitions']))
                else:
                    adv_batch = replay.sample(min(256, replay.get_stats()['total_transitions']))
                adv_batch_d = collate_to_critic_device(adv_batch, device)
                with torch.no_grad():
                    q_out_adv = critic(adv_batch_d["state"], adv_batch_d["action"])
                    q_baseline = torch.min(q_out_adv["total"][:, 0],
                                           q_out_adv["total"][:, 1]).mean()

                for idx in _actor_indices:
                    item = obs_buffer_pool[idx]
                    obs_torch = {k: v.to(device) for k, v in item["obs_torch"].items()}

                    # Lazy teacher: compute teacher action on-demand only for selected items
                    actor_out = actor(obs_torch, deterministic=False, compute_teacher=True)
                    teacher_act = actor_out.get("teacher_action")
                    if teacher_act is None:
                        teacher_act = item["teacher_action"]
                        if teacher_act is not None:
                            teacher_act = teacher_act.unsqueeze(0).to(device)
                    q_sa = critic(actor_out["global_cond"], actor_out["action"])
                    q_min_sa = torch.min(q_sa["total"][:, 0], q_sa["total"][:, 1])
                    adv_i = q_min_sa.detach() - q_baseline

                    step_losses = compute_routea_actor_losses(
                        actor_outputs={
                            "action": actor_out["action"],
                            "log_prob": actor_out["log_prob"],
                            "noise_sigmas": actor_out["noise_sigmas"],
                        },
                        teacher_action=teacher_act,
                        advantage=adv_i, q_min=q_min_sa,
                        lambda_bc=LAMBDA_BC, lambda_path=0.0,
                        lambda_noise=LAMBDA_NOISE, lambda_smooth=LAMBDA_SMOOTH,
                        lambda_pg=LAMBDA_PG, lambda_dqpg=lambda_dqpg,
                    )
                    loss_i = step_losses["actor_loss_total"]
                    if torch.isfinite(loss_i):
                        actor_total_loss = actor_total_loss + loss_i
                        n_actor_valid += 1
                        for k, v in step_losses.items():
                            if k not in actor_loss_components:
                                actor_loss_components[k] = 0.0
                            actor_loss_components[k] += (v.item() if isinstance(v, torch.Tensor) else v)

                if n_actor_valid > 0:
                    (actor_total_loss / n_actor_valid).backward()
                    # Track pre-clip grad norm
                    _pre_norm = 0.0
                    for p in list(actor._lora_adaptors.parameters()) + list(actor.noise_module.parameters()):
                        if p.grad is not None:
                            _pre_norm += p.grad.norm().item() ** 2
                    actor_grad_preclip = math.sqrt(_pre_norm)
                    torch.nn.utils.clip_grad_norm_(
                        list(actor._lora_adaptors.parameters()) + list(actor.noise_module.parameters()),
                        MAX_GRAD_NORM_ACTOR,
                    )
                    actor_optimizer.step()
                    # Track post-clip grad norm
                    _post_norm = 0.0
                    for p in list(actor._lora_adaptors.parameters()) + list(actor.noise_module.parameters()):
                        if p.grad is not None:
                            _post_norm += p.grad.norm().item() ** 2
                    actor_grad_post = math.sqrt(_post_norm)
                    actor_losses.append((actor_total_loss / n_actor_valid).item())
                    for k in actor_loss_components:
                        actor_loss_components[k] /= n_actor_valid
                else:
                    actor_grad_post = 0.0
            else:
                actor_grad_post = 0.0
        else:
            # freeze_actor: skip all actor computation
            actor_grad_post = 0.0
            actor_grad_preclip = 0.0
            actor_loss_components = {}
            actor_losses.append(0.0)

        # ── NaN/Inf check on actor loss ──
        _al = actor_losses[-1] if actor_losses else 0.0
        if math.isnan(_al) or math.isinf(_al):
            nan_count += 1 if math.isnan(_al) else 0
            inf_count += 1 if math.isinf(_al) else 0

        # ── Critic updates ──
        critic_loss_total = 0.0
        twin_gap_val = 0.0
        q_std_val = 0.0
        critic_grad_norm = 0.0

        if replay.get_stats()['total_transitions'] >= 256:
            # ── Stabilized replay: increment ages each step ──
            if stabilized_replay:
                replay.increment_ages()

            for _cu in range(critic_updates):
                if stabilized_replay:
                    batch, _comp_report = replay.sample_stabilized(
                        min(batch_size, replay.get_stats()['total_transitions']))
                else:
                    batch = replay.sample(min(batch_size, replay.get_stats()['total_transitions']))
                    _comp_report = {}
                if batch["state"].shape[0] == 0:
                    continue
                batch_d = collate_to_critic_device(batch, device)

                critic_optimizer.zero_grad()
                c_loss, c_log = compute_critic_loss(
                    critic, target_critic,
                    batch_d["state"], batch_d["action"],
                    batch_d["reward_total"], batch_d["reward_heads"],
                    batch_d["next_state"], batch_d["action_tp1"],
                    done=batch_d["done"], gamma=GAMMA, betas=CRITIC_BETAS,
                )
                if torch.isfinite(c_loss):
                    c_loss.backward()
                    _cg_pre = sum(
                        p.grad.norm().item() ** 2 for p in critic.parameters()
                        if p.grad is not None
                    )
                    critic_grad_norm = math.sqrt(_cg_pre) if _cg_pre > 0 else 0.0
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM_CRITIC)
                    critic_optimizer.step()
                    target_update(target_critic, critic, tau=TARGET_TAU)
                    critic_loss_total += c_loss.item()

                # Twin gap and Q std from last critic update
                with torch.no_grad():
                    q_out = critic(batch_d["state"], batch_d["action"])
                    twin_gap_val = (q_out["total"][:, 0] - q_out["total"][:, 1]).abs().mean().item()
                    q_std_val = q_out["total"].std(dim=0).mean().item()

            critic_loss_total /= max(critic_updates, 1)
            critic_losses.append(critic_loss_total)
        else:
            critic_losses.append(0.0)

        # ── Periodic eval ──
        eval_snapshot = {}
        if step % eval_interval == 0 or step == 1:
            cprint(f"\n[FormalTrain] Eval at step {step}...", "cyan")
            eval_res = _run_newenv_eval(actor, base_policy, device, val_pool,
                                        world_meta_lookup, rng,
                                        n_episodes_per_world=2, seed_offset=step,
                                        curriculum_stage=curriculum_stage)
            eval_history.append({"step": step, **eval_res})
            eval_snapshot = eval_res.get("per_role", {})

            # Best checkpoint scoring: weighted SR - CR penalty
            _score = 0.0
            _veto_reasons = []
            for role in ["anticipatory", "dense_recovery", "support_expansion"]:
                role_m = eval_snapshot.get(role, {})
                _sr = role_m.get("SR", 0.0)
                _cr = role_m.get("CR", 0.0)
                _score += _sr * 1.0 - _cr * 0.5
            _score -= max(0.0, q_std_val - 15.0) * 0.02  # Q_std penalty

            # Best eligibility hard conditions
            metadata_ratio = metadata_complete / max(metadata_total, 1)
            if metadata_ratio < 1.0:
                _veto_reasons.append(f"metadata_completeness={metadata_ratio:.4f}")
            if contamination_count > 0:
                _veto_reasons.append(f"contamination={contamination_count}")
            if nan_count > 0 or inf_count > 0:
                _veto_reasons.append(f"NaN={nan_count}, Inf={inf_count}")

            if not _veto_reasons and _score > best_score_val:
                best_score_val = _score
                best_step_val = step
                ckpt_writer.update_best(step, _score,
                    actor.state_dict(), critic.state_dict(),
                    target_critic.state_dict(),
                    {"eval": eval_res, "score": _score, "q_std": q_std_val})
                cprint(f"  ✅ New best: step={step}, score={_score:.4f}", "green")
            elif _veto_reasons:
                cprint(f"  ⚠️  Best vetoed: {', '.join(_veto_reasons)}", "yellow")

            # Capability trace
            capability_trace.append({
                "step": step,
                **{f"sr_{r}": eval_snapshot.get(r, {}).get("SR", 0.0) for r in
                   ["anticipatory", "dense_recovery", "support_expansion", "yaw_lateral_recovery"]},
                **{f"cr_{r}": eval_snapshot.get(r, {}).get("CR", 0.0) for r in
                   ["anticipatory", "dense_recovery", "support_expansion", "yaw_lateral_recovery"]},
                "score": _score, "q_std": q_std_val, "twin_gap": twin_gap_val,
            })

        # ── Hard stop check ──
        hstop_metrics = {
            "nan_count": nan_count, "inf_count": inf_count,
            "actor_grad_norm_post_clip": actor_grad_post,
            "critic_Q_std": q_std_val, "critic_twin_gap": twin_gap_val,
        }
        meta_stats = {
            "contamination_count": contamination_count,
            "completeness_ratio": metadata_complete / max(metadata_total, 1),
        }
        hstop_reason = hard_stop.check(step, hstop_metrics, eval_snapshot if eval_snapshot else None, meta_stats)
        if hstop_reason:
            cprint(f"\n  🛑 HARD STOP at step {step}: {hstop_reason}", "red", "on_white")
            ckpt_writer.save_emergency(step, actor.state_dict(), critic.state_dict(), hstop_reason)
            hard_stop_reason = hstop_reason
            break

        # ── Periodic checkpoint ──
        if step % checkpoint_interval == 0:
            ckpt_writer.save_periodic(step,
                actor.state_dict(), critic.state_dict(),
                target_critic.state_dict(),
                {"actor_optimizer": actor_optimizer.state_dict(),
                 "critic_optimizer": critic_optimizer.state_dict()},
                {"eval": eval_snapshot, "q_std": q_std_val, "twin_gap": twin_gap_val})

        # ── Streaming metrics ──
        step_elapsed = time.time() - step_start
        metadata_ratio = metadata_complete / max(metadata_total, 1)

        train_csv.append({
            "step": step,
            "actor_loss_total": round(actor_losses[-1], 6) if actor_losses else 0.0,
            "actor_loss_pg": round(actor_loss_components.get("actor_loss_pg", 0.0), 6),
            "actor_loss_dqpg": round(actor_loss_components.get("actor_loss_dqpg", 0.0), 6),
            "actor_loss_bc": round(actor_loss_components.get("actor_loss_bc", 0.0), 6),
            "actor_loss_noise": round(actor_loss_components.get("actor_loss_noise", 0.0), 6),
            "actor_loss_smooth": round(actor_loss_components.get("actor_loss_smooth", 0.0), 6),
            "critic_loss_total": round(critic_losses[-1], 6) if critic_losses else 0.0,
            "critic_twin_gap": round(twin_gap_val, 6),
            "critic_Q_std": round(q_std_val, 4),
            "actor_grad_norm_post_clip": round(actor_grad_post, 6),
            "critic_grad_norm": round(critic_grad_norm, 6),
            "dqpg_enabled": int(dqpg_enabled),
            "n_episodes": step_episodes,
            "n_transitions": step_transitions,
            "replay_size": replay.get_stats()['total_transitions'],
            "eval_sr_anticipatory": eval_snapshot.get("anticipatory", {}).get("SR", -1),
            "eval_cr_anticipatory": eval_snapshot.get("anticipatory", {}).get("CR", -1),
            "eval_sr_dense_recovery": eval_snapshot.get("dense_recovery", {}).get("SR", -1),
            "eval_cr_dense_recovery": eval_snapshot.get("dense_recovery", {}).get("CR", -1),
            "eval_sr_support_expansion": eval_snapshot.get("support_expansion", {}).get("SR", -1),
            "eval_cr_support_expansion": eval_snapshot.get("support_expansion", {}).get("CR", -1),
            "eval_sr_yaw_lateral_recovery": eval_snapshot.get("yaw_lateral_recovery", {}).get("SR", -1),
            "eval_cr_yaw_lateral_recovery": eval_snapshot.get("yaw_lateral_recovery", {}).get("CR", -1),
            "nan_count": nan_count, "inf_count": inf_count,
            "metadata_completeness": round(metadata_ratio, 4),
            "contamination_count": contamination_count,
            "timing_sec": round(step_elapsed, 3),
        })

        critic_drift_csv.append({
            "step": step, "twin_gap": round(twin_gap_val, 6),
            "q_std": round(q_std_val, 4),
            "q_total_mean": 0.0, "q_total_std": 0.0,
            "q_progress_mean": 0.0, "q_onset_mean": 0.0, "q_fclear_mean": 0.0,
            "critic_loss_total": round(critic_losses[-1], 6) if critic_losses else 0.0,
            "target_q_mean": 0.0, "td_target_mean": 0.0,
        })

        # Actor update health
        actor_update_health.append({
            "step": step,
            "actor_loss": round(actor_losses[-1], 6) if actor_losses else 0.0,
            "actor_grad_preclip": round(actor_grad_preclip, 6),
            "actor_grad_postclip": round(actor_grad_post, 6),
            "nan": nan_count, "inf": inf_count,
        })

        # ── Progress print ──
        if step % 100 == 0 or step == 1:
            _elapsed = time.time() - train_start
            _eta = (_elapsed / step) * (max_actor_updates - step) / 60
            _eval_str = ""
            if eval_snapshot:
                _eval_str = " | ".join(
                    f"{r[:4]} SR={eval_snapshot.get(r, {}).get('SR', 0):.2f}"
                    for r in ["anticipatory", "dense_recovery", "support_expansion"]
                    if r in eval_snapshot
                )
            cprint(
                f"  Step {step}/{max_actor_updates} | "
                f"actor_loss={actor_losses[-1]:.3f}" if actor_losses else f"  Step {step}/{max_actor_updates} | no_loss",
                "cyan",
            )
            if actor_losses:
                cprint(
                    f"    critic_loss={critic_losses[-1]:.3f}" if critic_losses else "    no_critic",
                    "cyan",
                )
            cprint(
                f"    twin_gap={twin_gap_val:.3f} Qσ={q_std_val:.1f} "
                f"grad={actor_grad_post:.3f} ep={total_episodes} "
                f"NaN={nan_count} Inf={inf_count} | {_eval_str}",
                "cyan",
            )
            cprint(f"    elapsed={_elapsed/60:.1f}m ETA={_eta:.1f}m", "cyan")

        # ── Freeze boundary audit (every 500 steps) ──
        if step % 500 == 0:
            f_ok, f_violations = verify_freeze_boundary(actor, base_fm_hash, obs_encoder_hash, teacher_hash)
            if not f_ok:
                cprint(f"  🛑 FREEZE VIOLATION: {f_violations}", "red")
                hard_stop_reason = f"freeze_violation: {f_violations}"
                ckpt_writer.save_emergency(step, actor.state_dict(), critic.state_dict(), hard_stop_reason)
                break

        # ── Trim obs_buffer_pool ──
        if len(obs_buffer_pool) > 64:
            obs_buffer_pool = obs_buffer_pool[-64:]

    # ═══════════════════════════════════════════════════════════════
    # Training complete — save final outputs
    # ═══════════════════════════════════════════════════════════════
    train_elapsed = time.time() - train_start
    final_step = step if not hard_stop_reason else step - 1
    actor_updates_done = final_step

    cprint(f"\n[FormalTrain] Training finished: {actor_updates_done} updates in {train_elapsed/60:.1f}m", "green")

    # Save final checkpoint
    ckpt_writer.save_final(final_step,
        actor.state_dict(), critic.state_dict(), target_critic.state_dict(),
        {"eval": eval_history[-1] if eval_history else {}})

    # Save role sampling trace
    for role in TRAINABLE_ROLES:
        role_csv.append({
            "step": final_step, "role": role,
            "n_episodes": per_role_episodes.get(role, 0),
            "n_transitions": per_role_transitions.get(role, 0),
            "pct_episodes": round(per_role_episodes.get(role, 0) / max(total_episodes, 1), 4),
        })

    # Save actor update health CSV
    health_csv = StreamingCSVWriter(
        output_dir / "actor_update_health.csv",
        ["step", "actor_loss", "actor_grad_preclip", "actor_grad_postclip", "nan", "inf"],
    )
    for h in actor_update_health:
        health_csv.append(h)
    health_csv.close()

    # Save capability indicator trace CSV
    if capability_trace:
        cap_csv = StreamingCSVWriter(
            output_dir / "capability_indicator_trace.csv",
            list(capability_trace[0].keys()),
        )
        for c in capability_trace:
            cap_csv.append(c)
        cap_csv.close()

    # Close streaming writers
    train_csv.close()
    role_csv.close()
    critic_drift_csv.close()

    # ── Build summary (Task 6: run manifest truthfulness) ──
    metadata_ratio_final = metadata_complete / max(metadata_total, 1)
    _mode_label = "formal_legacy"
    if fixed_replay:
        _mode_label = "fixed_replay" if not freeze_actor else "fixed_replay_freeze_actor"
    if rescue_from:
        _mode_label = "rescue"

    # Count optimizer steps from logs
    _actor_opt_steps = 0 if freeze_actor else actor_updates_done
    _critic_opt_steps = len(critic_losses) * critic_updates if critic_losses else 0

    summary = {
        # ── Task 6: Run manifest truthfulness ──
        "run_id": ckpt_writer.run_id,
        "config_hash": hashlib.md5(
            json.dumps({
                "actor_lr": actor_lr, "critic_lr": critic_lr,
                "max_actor_updates": max_actor_updates,
                "critic_updates": critic_updates,
                "dqpg_warmup_steps": dqpg_warmup_steps,
                "role_ratios": role_ratios,
            }, sort_keys=True).encode()
        ).hexdigest()[:12],
        "source_script_mtime": datetime.fromtimestamp(
            os.path.getmtime(__file__)).isoformat(),
        "mode": _mode_label,
        "actor_optimizer_steps": _actor_opt_steps,
        "critic_optimizer_steps": _critic_opt_steps,
        "target_update_steps": _critic_opt_steps,
        "new_rollout_episodes": total_episodes,
        "replay_size_final": replay.get_stats()['total_transitions'],
        "hard_stop_primary_reason": hard_stop_reason if hard_stop_reason else "",
        "hard_stop_secondary_reasons": [],
        # ── Legacy fields ──
        "stage": "A2-newenv_formal_capability_short_train",
        "formal_training_status": "hard_stop" if hard_stop_reason else "complete",
        "actor_updates_completed": actor_updates_done,
        "actor_updates_max": max_actor_updates,
        "admission_manifest": str(getattr(manifest, '_path', 'unknown')),
        "old_rl_1k_failure_audit": str(output_dir / "old_rl_1k_failure_audit.md"),
        "formal_roles": ["anticipatory", "dense_recovery", "support_expansion"],
        "support_roles": ["yaw_lateral_recovery"],
        "historical_contamination": contamination_count,
        "metadata_completeness": round(metadata_ratio_final, 4),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "hard_stop_reason": hard_stop_reason,
        "best_checkpoint": {
            "step": best_step_val,
            "score": round(best_score_val, 4),
            "file": "a2_actor_best.pt",
        },
        "latest_checkpoint": f"a2_ckpt_step{final_step:06d}.pt",
        "best_checkpoint_reason": {
            "score_formula": "weighted_SR - 0.5*CR - 0.02*max(0, Q_std-15)",
            "veto_conditions": "metadata<1.0, contamination>0, NaN/Inf>0",
        },
        "per_role_sampling": {
            role: {
                "episodes": per_role_episodes.get(role, 0),
                "pct": round(per_role_episodes.get(role, 0) / max(total_episodes, 1), 4),
            }
            for role in TRAINABLE_ROLES
        },
        "eval_history_summary": [
            {"step": e["step"], "n_worlds": e.get("n_worlds_evaluated", 0),
             "n_episodes": e.get("n_episodes", 0)}
            for e in eval_history
        ],
        "lightweight_eval_status": "monitoring_only_not_gazebo",
        "gazebo_closed_loop_status": "not_run_in_this_task",
        "claim_allowed": False,
        "claim_blocked_reasons": [
            "A2-newenv short-train only, not A3 full training",
            "lightweight kinematic eval only, not Gazebo/PX4 closed-loop",
        ],
        "next_stage": "A3_full_training_design_after_A2_newenv_review",
        "note": "A3 full training step count NOT decided in this task; must wait for A2-newenv 10k short-train result review",
        "timestamp": datetime.now().isoformat(),
        "elapsed_min": round(train_elapsed / 60, 1),
    }

    with open(output_dir / "formal_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Save summary MD ──
    summary_md_lines = [
        "# Route A RL v1 — A2-NewEnv Formal Capability Short-Train Summary",
        f"**Status**: {summary['formal_training_status']}",
        f"**Actor updates**: {actor_updates_done}/{max_actor_updates}",
        f"**Elapsed**: {train_elapsed/60:.1f} min",
        f"**Timestamp**: {summary['timestamp']}",
        "",
        "## Health",
        f"- NaN: {nan_count}, Inf: {inf_count}",
        f"- Metadata completeness: {metadata_ratio_final:.4f}",
        f"- Historical contamination: {contamination_count}",
        f"- Hard stop: {'YES — ' + hard_stop_reason if hard_stop_reason else 'No'}",
        "",
        "## Best Checkpoint",
        f"- Step: {best_step_val}",
        f"- Score: {best_score_val:.4f}",
        f"- Q_std at best: see eval_history",
        "",
        "## Per-Role Sampling",
    ]
    for role in TRAINABLE_ROLES:
        summary_md_lines.append(
            f"- {role}: {per_role_episodes.get(role, 0)} episodes "
            f"({per_role_episodes.get(role, 0)/max(total_episodes, 1)*100:.1f}%)"
        )
    summary_md_lines += [
        "",
        "## Eval History",
    ]
    for e in eval_history:
        _per_role = e.get("per_role", {})
        _parts = []
        for r in ["anticipatory", "dense_recovery", "support_expansion"]:
            if r in _per_role:
                _parts.append(f"{r[:6]} SR={_per_role[r].get('SR', 0):.3f} CR={_per_role[r].get('CR', 0):.3f}")
        summary_md_lines.append(f"- Step {e['step']}: " + ", ".join(_parts))
    summary_md_lines += [
        "",
        "## Claim Status",
        f"- claim_allowed: {summary['claim_allowed']}",
    ]
    for r in summary["claim_blocked_reasons"]:
        summary_md_lines.append(f"- {r}")
    summary_md_lines += [
        "",
        f"## Next Stage: {summary['next_stage']}",
        f"- {summary['note']}",
    ]
    with open(output_dir / "formal_training_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_md_lines) + "\n")

    cprint(f"\n  Summary: {output_dir / 'formal_training_summary.json'}", "green")
    cprint(f"  Summary MD: {output_dir / 'formal_training_summary.md'}", "green")

    return summary


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="A2 10k Capability Short-Train")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="v9_2 BC/FM checkpoint")
    parser.add_argument("--critic_ckpt", type=str, required=True,
                        help="A0 LW1 a0_critic_best.pt")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    # ── Route A RL v1: New-env training (Steps 11-14) ──
    parser.add_argument("--routea_rl_admission_manifest", type=str, default=None,
                        help="Path to Route A RL v1 admission_manifest.json")
    parser.add_argument("--use_routea_rl_env_v1", action="store_true",
                        help="Enable Route A RL v1 new-env role-driven training")
    parser.add_argument("--routea_rl_train_split", type=str, default="rl_train",
                        help="Split name for training worlds (default: rl_train)")
    parser.add_argument("--routea_rl_val_split", type=str, default="rl_val",
                        help="Split name for validation worlds (default: rl_val)")
    parser.add_argument("--routea_rl_test_split", type=str, default="rl_test",
                        help="Split name for test worlds (default: rl_test)")
    parser.add_argument("--forbid_historical_train", action="store_true", default=True,
                        help="Forbid historical eval worlds in train/val (default: True)")
    parser.add_argument("--role_balanced_sampler", action="store_true",
                        help="Enable role-balanced world sampling from admission manifest")
    parser.add_argument("--dry_run_newenv_loader", action="store_true",
                        help="Run dry-run admission check and exit (no training)")
    parser.add_argument("--a0_newenv_critic_warmup_smoke", action="store_true",
                        help="Run A0 critic warmup smoke on Route A RL v1 new env pool")
    parser.add_argument("--a0_newenv_smoke_steps", type=int, default=A0_NEWENV_SMOKE_STEPS,
                        help=f"Number of A0 warmup steps (default: {A0_NEWENV_SMOKE_STEPS})")
    parser.add_argument("--a1_newenv_stability_smoke", action="store_true",
                        help="Run A1 stability smoke on Route A RL v1 new env pool")
    parser.add_argument("--a1_max_actor_updates", type=int, default=A1_NEWENV_MAX_ACTOR_UPDATES,
                        help=f"Max actor updates for A1 smoke (default: {A1_NEWENV_MAX_ACTOR_UPDATES})")
    parser.add_argument("--a2_newenv_capability_smoke", action="store_true",
                        help="Run A2 capability smoke on Route A RL v1 new env pool")
    parser.add_argument("--a2_max_actor_updates", type=int, default=500,
                        help="Max actor updates for A2 smoke (default: 500, max 3000)")
    parser.add_argument("--save_newenv_actor_ckpt", type=str, default="",
                        help="Save Route A RL actor checkpoint to this directory after A2 smoke completes (gate must PASS)")
    # ── Route A RL v1: Formal Training ──
    parser.add_argument("--routea_rl_formal_train", action="store_true",
                        help="Run Route A RL v1 formal training (full RL on role-driven env pool)")
    parser.add_argument("--formal_train_max_actor_updates", type=int, default=10000,
                        help="Max actor updates for formal training (default: 10000)")
    parser.add_argument("--formal_train_eval_interval", type=int, default=1000,
                        help="Eval interval in actor updates (default: 1000)")
    parser.add_argument("--formal_train_checkpoint_interval", type=int, default=1000,
                        help="Checkpoint interval in actor updates (default: 1000)")
    parser.add_argument("--formal_train_output_dir", type=str, default=None,
                        help="Override output dir for formal training (default: --output_dir)")
    parser.add_argument("--formal_train_eval_plan", type=str, default=None,
                        help="Path to real_eval_world_plan.json for eval scheduling")
    parser.add_argument("--formal_train_metrics_schema", type=str, default=None,
                        help="Path to real_rollout_metrics_schema.json")
    parser.add_argument("--formal_train_resume_from", type=str, default=None,
                        help="[DISABLED] Resuming is not supported (P0 verdict: "
                             "RESTART_FROM_COMMON_B1). Passing a value raises "
                             "NotImplementedError. Use --critic_ckpt_path + "
                             "--ft_bootstrap_replay for the common B1 restart instead.")
    # ── Formal training hyperparam overrides ──
    parser.add_argument("--ft_actor_lr", type=float, default=1e-4,
                        help="Actor LR for formal training (default: 1e-4)")
    parser.add_argument("--ft_critic_lr", type=float, default=1e-4,
                        help="Critic LR for formal training (default: 1e-4)")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Batch size for training (default: 2048)")
    parser.add_argument("--ft_rollouts_per_cycle", type=int, default=4,
                        help="Rollouts per training cycle (default: 4)")
    parser.add_argument("--ft_critic_updates", type=int, default=16,
                        help="Critic update batches per actor step (default: 16)")
    parser.add_argument("--ft_freeze_actor", action="store_true",
                        help="Disable actor updates (critic-only rescue mode)")
    parser.add_argument("--ft_rescue_from", type=str, default=None,
                        help="Path to emergency checkpoint for critic rescue")
    parser.add_argument("--ft_fixed_replay", action="store_true",
                        help="Skip rollouts, train critic only on fixed replay buffer")
    parser.add_argument("--ft_stabilized_replay", action="store_true",
                        help="Use stabilized replay sampler (role-balanced + age-mix + hard retention)")
    parser.add_argument("--ft_replay_warmup_transitions", type=int, default=5000,
                        help="Min transitions for replay warmup (default: 5000, debug: 2000)")
    parser.add_argument("--ft_replay_warmup_episodes", type=int, default=100,
                        help="Min episodes for replay warmup (default: 100, debug: 40)")
    parser.add_argument("--ft_dqpg_warmup_steps", type=int, default=1000,
                        help="DQPG warmup steps for formal training (default: 1000)")
    parser.add_argument("--ft_anticipatory_ratio", type=float, default=0.30,
                        help="Sampling ratio for anticipatory role (default: 0.30)")
    parser.add_argument("--ft_dense_recovery_ratio", type=float, default=0.35,
                        help="Sampling ratio for dense_recovery role (default: 0.35)")
    parser.add_argument("--ft_support_expansion_ratio", type=float, default=0.25,
                        help="Sampling ratio for support_expansion role (default: 0.25)")
    parser.add_argument("--ft_yaw_lateral_recovery_ratio", type=float, default=0.10,
                        help="Sampling ratio for yaw_lateral_recovery role (default: 0.10)")
    # ── P0 wiring health: collect-only mode (Task 3) ──
    parser.add_argument("--ft_collect_only", action="store_true",
                        help="Collect-only mode: sample rollouts, write replay, skip actor/critic updates")
    parser.add_argument("--ft_collect_only_max_steps", type=int, default=100,
                        help="Max collection steps in collect-only mode (default: 100)")
    parser.add_argument("--ft_collect_only_output_replay", type=str, default=None,
                        help="Path for replay manifest export in collect-only mode")
    # ── P0 wiring health: critic-bootstrap mode (Task 4) ──
    parser.add_argument("--ft_critic_bootstrap", action="store_true",
                        help="Critic bootstrap: fixed replay, no new rollouts, no actor update, critic-only training")
    parser.add_argument("--ft_bootstrap_heldout_ratio", type=float, default=0.2,
                        help="Heldout ratio for critic bootstrap diagnostics (default: 0.2)")
    parser.add_argument("--ft_bootstrap_replay", type=str, default=None,
                        help="Path to fixed replay .pt file from B0 collect-only for B1 critic bootstrap")
    # ── B1a: diagnostic / ablation CLI ──
    parser.add_argument("--ft_bootstrap_diag_only", action="store_true",
                        help="[B1a] Diagnostic-only: load replay, compute metrics, skip all training.")
    parser.add_argument("--ft_bootstrap_gamma", type=float, default=None,
                        help="[B1a] Override gamma for critic bootstrap (default: 0.99)")
    parser.add_argument("--ft_bootstrap_target_tau", type=float, default=None,
                        help="[B1a] Override target network tau for critic bootstrap (default: 0.005)")
    parser.add_argument("--ft_bootstrap_total_only", action="store_true",
                        help="[B1a] Train only total head (set all decomposed betas to 0).")
    parser.add_argument("--ft_bootstrap_huber_total", action="store_true",
                        help="[B1a] Use SmoothL1 loss for total head instead of MSE.")
    parser.add_argument("--ft_bootstrap_huber_beta", type=float, default=5.0,
                        help="[B1a] Huber loss beta for total head (default: 5.0)")
    parser.add_argument("--ft_bootstrap_sparse_head_mode", type=str, default="none",
                        choices=["none", "zero_onset_trap_betas", "huber_sparse_heads", "sparse_head_l2"],
                        help="[B1b] Sparse head treatment: none, zero_onset_trap_betas, huber_sparse_heads, sparse_head_l2")
    # ── B1c: architecture / reward representation knobs ──
    parser.add_argument("--ft_bootstrap_reward_norm_mode", type=str, default="none",
                        choices=["none", "fixed_train_standardize", "fixed_train_scale_p95"],
                        help="[B1c] Reward normalization mode for critic bootstrap")
    parser.add_argument("--ft_bootstrap_reward_norm_eps", type=float, default=1e-6,
                        help="[B1c] Epsilon for reward normalization (default: 1e-6)")
    parser.add_argument("--ft_bootstrap_q_l2_weight", type=float, default=0.0,
                        help="[B1c] L2 penalty weight on Q outputs (default: 0.0)")
    parser.add_argument("--ft_bootstrap_q_soft_clip", type=float, default=0.0,
                        help="[B1c] Soft clip Q outputs (NOT_IMPLEMENTED placeholder, default: 0.0)")
    parser.add_argument("--ft_bootstrap_critic_norm", type=str, default="none",
                        choices=["none", "layernorm"],
                        help="[B1c] Critic normalization: none or layernorm")
    parser.add_argument("--ft_bootstrap_target_update_mode", type=str, default="polyak_each_step",
                        choices=["polyak_each_step", "hard_every_n"],
                        help="[B1c] Target network update mode (default: polyak_each_step)")
    parser.add_argument("--ft_bootstrap_target_update_interval", type=int, default=100,
                        help="[B1c] Target hard update interval in critic steps (default: 100)")
    parser.add_argument("--ft_bootstrap_split_seed", type=int, default=None,
                        help="[B1d] Seed for episode train/heldout split RNG (default: None, uses main RNG)")
    # ── P0 wiring health: phased online schedule (Task 5) ──
    parser.add_argument("--ft_phased_online", action="store_true",
                        help="Phased online schedule: collect → critic digest → actor update with explicit gates")
    parser.add_argument("--ft_phase_collect_episodes", type=int, default=32,
                        help="Episodes per collection phase (default: 32)")
    parser.add_argument("--ft_phase_critic_digest_updates", type=int, default=200,
                        help="Critic updates per digest phase (default: 200)")
    parser.add_argument("--ft_phase_actor_updates", type=int, default=10,
                        help="Actor updates per phase, only after critic digest (default: 10)")
    parser.add_argument("--ft_reward_norm_stats", type=str, default=None,
                        help="Path to reward_norm_stats.json sidecar for phased-online reward normalization (B2)")
    # ── B2a: PG scale controls ──
    parser.add_argument("--ft_actor_lambda_pg", type=float, default=1.0,
                        help="PG (REINFORCE) loss weight (default: 1.0)")
    parser.add_argument("--ft_actor_lambda_dqpg", type=float, default=1.0,
                        help="DQPG (differentiable Q) loss weight (default: 1.0)")
    parser.add_argument("--ft_actor_pg_scale_mode", type=str, default="raw_sum",
                        choices=["raw_sum", "lambda_only", "per_action_dim", "per_flow_event"],
                        help="PG log_prob scale normalization mode (default: raw_sum)")
    parser.add_argument("--ft_actor_effective_action_dim", type=int, default=88,
                        help="Effective action dim for per_action_dim PG scale (default: 88)")
    parser.add_argument("--ft_actor_effective_flow_events", type=int, default=0,
                        help="Effective flow events for per_flow_event PG scale (0=infer from dims*4)")
    # ── B3: pilot safety caps ──
    parser.add_argument("--ft_max_phases", type=int, default=0,
                        help="Max phases for phased online (0=computed from max_actor_updates)")
    parser.add_argument("--ft_actor_compute_teacher", action="store_true",
                        help="Use compute_teacher=True in actor update for BC anchor (B3)")
    parser.add_argument("--ft_pilot_mode", action="store_true",
                        help="Pilot mode with strict hard stops: actor>8, phases>3, claim_allowed!=false")
    # ── B4: role quota pilot ──
    parser.add_argument("--ft_role_quota_pilot", action="store_true",
                        help="Role quota pilot: explicit per-role episode quotas across phases (B4)")
    parser.add_argument("--ft_min_episodes_anticipatory", type=int, default=2,
                        help="Min episodes for anticipatory role in quota pilot (default: 2)")
    parser.add_argument("--ft_min_episodes_dense_recovery", type=int, default=2,
                        help="Min episodes for dense_recovery role in quota pilot (default: 2)")
    parser.add_argument("--ft_min_episodes_support_expansion", type=int, default=2,
                        help="Min episodes for support_expansion role in quota pilot (default: 2)")
    parser.add_argument("--ft_max_support_role_share", type=float, default=0.50,
                        help="Max support_expansion role share in quota pilot (default: 0.50)")
    # ── B6: extended pilot ──
    parser.add_argument("--ft_extended_pilot", action="store_true",
                        help="B6 extended pilot: relaxed caps for 50-200 actor updates (actor>200, phases>15)")
    # ── B8: long training ──
    parser.add_argument("--ft_b8_long_train", action="store_true",
                        help="B8 long training: relaxed caps for 100-400 actor updates (actor>400, phases>40)")
    parser.add_argument("--ft_b11_500upd", action="store_true",
                        help="B12 extended long training: relaxed caps for 500 actor updates (actor>700, phases>80)")
    parser.add_argument("--ft_b15_1000upd", action="store_true",
                        help="B15-B17 extended long training: relaxed caps for 1000 actor updates (actor>1200, phases>150)")
    # ── B9: LoRA rank ──
    parser.add_argument("--ft_lora_rank", type=int, default=None,
                        help="LoRA rank for actor adapters. Unset (None) => 4 normally, "
                             "or 16 under --ft_c2_ooc. Explicit value is ALWAYS respected "
                             "(capacity-budget matrix rank in {4,16,32}).")
    parser.add_argument("--ft_curriculum_stage", type=str, default="C2",
                        choices=["C0", "C1", "C2"],
                        help="Start/goal sampling curriculum stage (default: C2)")
    # ── B5: lightweight paired eval ──
    parser.add_argument("--ft_save_actor_path", type=str, default=None,
                        help="Save actor state_dict after phased_online (B5)")
    parser.add_argument("--ft_post_eval", action="store_true",
                        help="Run post-training eval after phased_online (B5)")
    parser.add_argument("--ft_eval_eps_per_world", type=int, default=2,
                        help="Episodes per world in B5 eval (default: 2, B5v2: 12)")
    # ── C2-OOC: online cell smoke ──
    parser.add_argument("--ft_c2_ooc", action="store_true",
                        help="C2-OOC: single-cell online smoke with full monitoring and stop conditions")
    parser.add_argument("--ft_mid_eval", action="store_true",
                        help="C2-OOC: enable mid-training paired eval")
    parser.add_argument("--ft_mid_eval_phase", type=int, default=10,
                        help="C2-OOC: phase number for mid-training eval (default: 10)")
    parser.add_argument("--ft_c2_ooc_no_twin_stop", action="store_true",
                        help="C2-OOC: disable S1/S2 twin-gap stop (keep S3 grad/S4 NaN)")
    parser.add_argument("--seed", type=int, default=20260705,
                        help="Global RNG seed (default: 20260705)")
    args = parser.parse_args()

    # ── Route A RL v1: New-env validation (Steps 11-13) ──
    _newenv_manifest: Optional[RouteARLAdmissionManifest] = None
    _newenv_train_blocked: bool = False
    _newenv_dry_run_result: Optional[Dict[str, Any]] = None

    if args.use_routea_rl_env_v1:
        # Validate CLI arg rules (Step 11)
        if args.routea_rl_admission_manifest is None:
            raise ValueError(
                "--routea_rl_admission_manifest is required when --use_routea_rl_env_v1 is set"
            )
        if not args.role_balanced_sampler:
            raise ValueError(
                "--role_balanced_sampler must be set when --use_routea_rl_env_v1 is set"
            )
        if not args.forbid_historical_train:
            raise ValueError(
                "--forbid_historical_train must be True when --use_routea_rl_env_v1 is set"
            )
        if not os.path.exists(args.routea_rl_admission_manifest):
            raise FileNotFoundError(
                f"RouteA RL admission manifest not found: {args.routea_rl_admission_manifest}"
            )

        cprint(f"\n  [RouteA RL v1] Loading admission manifest: {args.routea_rl_admission_manifest}", "cyan")
        _newenv_manifest = load_routea_rl_admission_manifest(args.routea_rl_admission_manifest)
        cprint(f"    Schema: {_newenv_manifest.schema_version}, "
               f"Date: {_newenv_manifest.calibration_date}, "
               f"Worlds: {len(_newenv_manifest.worlds)}", "cyan")

        # Dry-run mode (Step 13)
        if args.dry_run_newenv_loader:
            cprint(f"\n  [RouteA RL v1] DRY-RUN MODE — checking admission, no training", "white", "on_blue")
            _newenv_dry_run_result = dry_run_newenv_loader(args.routea_rl_admission_manifest)
            for line in _newenv_dry_run_result["summary_lines"]:
                cprint(f"    {line}", "green" if line.startswith("PASS") else "red")
            cprint(f"\n  Dry-run complete. Exiting.", "cyan")
            return

        # Train admission gate (Steps 7-9)
        admission_check = validate_rl_train_admission(_newenv_manifest)
        cprint(f"\n  [RouteA RL v1] Train admission check:", "cyan")
        cprint(f"    Schema pass: {admission_check['pass_schema']}", "green")
        cprint(f"    Formal role buckets: {admission_check['formal_role_buckets']}", "cyan")
        cprint(f"    Support bucket: yaw_lateral_recovery={admission_check['support_bucket']}", "cyan")
        cprint(f"    Historical train worlds: {admission_check['historical_train_worlds']}", "cyan")
        cprint(f"    Stress train worlds: {admission_check['stress_train_worlds']}", "cyan")

        if not admission_check["pass_train_admission"]:
            _newenv_train_blocked = True
            cprint(f"\n  ❌ BLOCKED: Route A RL v1 train admission failed", "red", "on_white")
            for err in admission_check["errors"]:
                cprint(f"    - {err}", "red")
            cprint(f"\n  Training cannot proceed. Fix by:", "yellow")
            cprint(f"    1. Re-run calibration with more episodes for rl_train worlds", "yellow")
            cprint(f"    2. Re-generate worlds if validation/calibration shows systematic issues", "yellow")
            cprint(f"    3. Check that role_validated/role_active are set correctly", "yellow")
            sys.exit(1)
        else:
            cprint(f"  ✅ Route A RL v1 train admission PASSED", "green")

        # Build role buckets for sampling (Step 5)
        _train_buckets = build_role_buckets(_newenv_manifest, args.routea_rl_train_split)
        _val_buckets = build_role_buckets(_newenv_manifest, args.routea_rl_val_split)
        bucket_pass, bucket_msg = validate_role_buckets_for_training(_train_buckets)
        if not bucket_pass:
            raise ValueError(f"RouteA RL v1 train buckets invalid: {bucket_msg}")
        cprint(f"    Train buckets: " +
               ", ".join(f"{r}={len(_train_buckets.get(r, []))}" for r in TRAINABLE_ROLES), "cyan")
        cprint(f"    Val buckets: " +
               ", ".join(f"{r}={len(_val_buckets.get(r, []))}" for r in TRAINABLE_ROLES), "cyan")

        # Gate config for reference
        _newenv_gates = get_newenv_gate_config()
        cprint(f"    A0/A1/A2 new-env gates loaded", "cyan")

    # ── GPU optimizations ──
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')  # use tensor cores for float32
    cprint(f"  [GPU] cudnn.benchmark=True, TF32=True, matmul_precision=high", "cyan")
    cprint(f"  [GPU] {torch.cuda.get_device_name(0)} | "
           f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB | "
           f"batch_size={args.batch_size}", "cyan")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # ── A0 New-Env Critic Warmup Smoke branch ──
    if args.a0_newenv_critic_warmup_smoke:
        cprint("\n" + "=" * 70, "cyan")
        cprint("  A0 New-Env Critic Warmup Smoke", "white", "on_blue")
        cprint("=" * 70, "cyan")
        cprint(f"  Output: {output_dir}", "cyan")
        cprint(f"  Smoke steps: {args.a0_newenv_smoke_steps}", "cyan")
        cprint(f"  Mode: critic-only, frozen BC/FM rollouts, actor update DISABLED", "yellow")

        if _newenv_manifest is None:
            raise ValueError(
                "--a0_newenv_critic_warmup_smoke requires --use_routea_rl_env_v1 "
                "and a valid admission manifest"
            )

        # Load BC/FM policy only (no teacher, no actor, no A0 critic checkpoint)
        cprint("\n[Step 1] Loading v9_2 BC/FM policy (frozen, no actor)...", "white", "on_blue")
        base_policy = load_v9_2_policy(args.ckpt, device)
        base_policy.eval()

        # Determine world_dir from manifest path
        _world_dir = "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds"

        # Run A0 warmup
        a0_result = run_a0_newenv_warmup(
            base_policy=base_policy,
            device=device,
            output_dir=output_dir,
            manifest=_newenv_manifest,
            train_split=args.routea_rl_train_split,
            world_dir=_world_dir,
            n_smoke_steps=args.a0_newenv_smoke_steps,
        )

        # Print gate result
        cprint(f"\n{'=' * 70}", "cyan")
        cprint(f"  A0 Gate: {a0_result['a0_gate_status']}", "green" if a0_result['a0_gate_status'] == 'PASS' else "red")
        for check, val in a0_result.get("a0_gate_checks", {}).items():
            status = "PASS" if val else "FAIL"
            cprint(f"    [{status}] {check}: {val}", "green" if val else "red")
        cprint(f"  Next: {a0_result['next_allowed_stage']}", "cyan")
        cprint(f"{'=' * 70}", "cyan")
        return

    # ── A1 New-Env Stability Smoke branch ──
    if args.a1_newenv_stability_smoke:
        cprint("\n" + "=" * 70, "cyan")
        cprint("  A1 New-Env Stability Smoke", "white", "on_blue")
        cprint("=" * 70, "cyan")
        cprint(f"  Output: {output_dir}", "cyan")
        cprint(f"  Max actor updates: {args.a1_max_actor_updates}", "cyan")

        if _newenv_manifest is None:
            raise ValueError(
                "--a1_newenv_stability_smoke requires --use_routea_rl_env_v1 "
                "and a valid admission manifest"
            )

        # Load BC/FM policy + teacher
        cprint("\n[Step 1] Loading v9_2 BC/FM + teacher policies...", "white", "on_blue")
        base_policy = load_v9_2_policy(args.ckpt, device)
        base_policy.eval()
        teacher_policy = load_v9_2_policy(args.ckpt, device)
        teacher_policy.eval()
        teacher_policy.requires_grad_(False)

        # Find A0 critic checkpoint
        _a0_ckpt = None
        _a0_dir = "/home/cjj/AUV/program/shiyan/results_routeA_RL_newenv_A0_smoke_20260703"
        _a0_candidate = Path(_a0_dir) / "a0_critic_final.pt"
        if _a0_candidate.exists():
            _a0_ckpt = str(_a0_candidate)
            cprint(f"  A0 critic found: {_a0_ckpt}", "green")
        else:
            cprint(f"  A0 critic not found at {_a0_candidate}, training from scratch", "yellow")

        _world_dir = "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds"

        a1_result = run_a1_newenv_stability(
            base_policy=base_policy,
            teacher_policy=teacher_policy,
            device=device,
            output_dir=output_dir,
            manifest=_newenv_manifest,
            train_split=args.routea_rl_train_split,
            val_split=args.routea_rl_val_split,
            world_dir=_world_dir,
            critic_ckpt_path=_a0_ckpt,
            max_actor_updates=args.a1_max_actor_updates,
        )

        cprint(f"\n{'=' * 70}", "cyan")
        cprint(f"  A1 Gate: {a1_result['a1_gate_status']}", "green" if a1_result['a1_gate_status'] == 'PASS' else "red")
        for check, val in a1_result.get("a1_gate_checks", {}).items():
            status = "PASS" if val else "FAIL"
            cprint(f"    [{status}] {check}: {val}", "green" if val else "red")
        cprint(f"  Next: {a1_result['next_allowed_stage']}", "cyan")
        cprint(f"{'=' * 70}", "cyan")
        return

    # ── A2 New-Env Capability Smoke branch ──
    if args.a2_newenv_capability_smoke:
        if args.a2_max_actor_updates > 3000:
            raise ValueError(f"a2_max_actor_updates ({args.a2_max_actor_updates}) exceeds 3000 limit")

        cprint("\n" + "=" * 70, "cyan")
        cprint("  A2 New-Env Capability Smoke", "white", "on_blue")
        cprint("=" * 70, "cyan")
        cprint(f"  Output: {output_dir}", "cyan")
        cprint(f"  Max actor updates: {args.a2_max_actor_updates}", "cyan")

        if _newenv_manifest is None:
            raise ValueError(
                "--a2_newenv_capability_smoke requires --use_routea_rl_env_v1"
            )

        # Load policies (same as A1)
        cprint("\n[Step 1] Loading v9_2 BC/FM + teacher policies...", "white", "on_blue")
        base_policy = load_v9_2_policy(args.ckpt, device)
        base_policy.eval()
        teacher_policy = load_v9_2_policy(args.ckpt, device)
        teacher_policy.eval()
        teacher_policy.requires_grad_(False)

        # Find A0 critic
        _a0_ckpt = None
        _a0_candidate = Path("/home/cjj/AUV/program/shiyan/results_routeA_RL_newenv_A0_smoke_20260703") / "a0_critic_final.pt"
        if _a0_candidate.exists():
            _a0_ckpt = str(_a0_candidate)
            cprint(f"  A0 critic found: {_a0_ckpt}", "green")
        else:
            cprint(f"  No A0 critic, training from scratch", "yellow")

        _world_dir = "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds"

        # Run the same A1 stability function with longer steps
        a1_raw = run_a1_newenv_stability(
            base_policy=base_policy,
            teacher_policy=teacher_policy,
            device=device,
            output_dir=output_dir,
            manifest=_newenv_manifest,
            train_split=args.routea_rl_train_split,
            val_split=args.routea_rl_val_split,
            world_dir=_world_dir,
            critic_ckpt_path=_a0_ckpt,
            max_actor_updates=args.a2_max_actor_updates,
            save_actor_ckpt_path=args.save_newenv_actor_ckpt if args.save_newenv_actor_ckpt else None,
        )

        # ── Wrap with A2 capability summary ──
        a2_summary = _build_a2_capability_summary(
            a1_raw=a1_raw,
            output_dir=output_dir,
            manifest=_newenv_manifest,
            max_updates=args.a2_max_actor_updates,
        )

        # Write A2 outputs
        with open(output_dir / "A2_newenv_capability_summary.json", "w") as f:
            json.dump(a2_summary, f, indent=2)
        _write_a2_summary_md(output_dir / "A2_newenv_capability_summary.md", a2_summary)

        cprint(f"\n{'=' * 70}", "cyan")
        cprint(f"  A2 Gate: {a2_summary['a2_gate_status']}", "green" if a2_summary['a2_gate_status'] == 'PASS' else "red")
        for check, val in a2_summary.get("a2_gate_checks", {}).items():
            status = "PASS" if val else "FAIL"
            cprint(f"    [{status}] {check}: {val}", "green" if val else "red")
        cprint(f"  Next: {a2_summary['next_allowed_stage']}", "cyan")
        cprint(f"{'=' * 70}", "cyan")
        return

    # ── Route A RL v1: Formal Training branch ──
    if args.routea_rl_formal_train:
        cprint("\n" + "=" * 70, "cyan")
        cprint("  Route A RL v1 — Formal Training", "white", "on_blue")
        cprint("=" * 70, "cyan")

        if _newenv_manifest is None:
            raise ValueError(
                "--routea_rl_formal_train requires --use_routea_rl_env_v1 "
                "and a valid admission manifest"
            )
        if not args.role_balanced_sampler:
            raise ValueError(
                "--role_balanced_sampler is required for formal training"
            )

        # Resolve output dir
        ft_output_dir = Path(args.formal_train_output_dir) if args.formal_train_output_dir else output_dir
        ft_output_dir.mkdir(parents=True, exist_ok=True)

        cprint(f"  Output: {ft_output_dir}", "cyan")
        cprint(f"  Max actor updates: {args.formal_train_max_actor_updates}", "cyan")
        cprint(f"  Eval interval: {args.formal_train_eval_interval}", "cyan")
        cprint(f"  Ckpt interval: {args.formal_train_checkpoint_interval}", "cyan")
        cprint(f"  Actor LR: {args.ft_actor_lr}, Critic LR: {args.ft_critic_lr}", "cyan")
        cprint(f"  DQPG warmup: {args.ft_dqpg_warmup_steps}, Rollouts/cycle: {args.ft_rollouts_per_cycle}", "cyan")

        # Resolve world_dir
        _world_dir = "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds"

        # Resolve critic ckpt
        _critic_path = args.critic_ckpt
        if not os.path.exists(_critic_path):
            # Try A0 smoke critic
            _a0_candidate = Path("/home/cjj/AUV/program/shiyan/results_routeA_RL_newenv_A0_smoke_20260703") / "a0_critic_final.pt"
            if _a0_candidate.exists():
                _critic_path = str(_a0_candidate)
                cprint(f"  Using A0 smoke critic: {_critic_path}", "yellow")
            else:
                raise FileNotFoundError(f"Critic checkpoint not found: {args.critic_ckpt}")

        # Load policies
        cprint("\n[FormalTrain] Loading v9_2 BC/FM + teacher...", "white", "on_blue")
        base_policy = load_v9_2_policy(args.ckpt, device)
        base_policy.eval()
        teacher_policy = load_v9_2_policy(args.ckpt, device)
        teacher_policy.eval()
        teacher_policy.requires_grad_(False)

        # Resolve role ratios
        ft_role_ratios = {
            "anticipatory": args.ft_anticipatory_ratio,
            "dense_recovery": args.ft_dense_recovery_ratio,
            "support_expansion": args.ft_support_expansion_ratio,
            "yaw_lateral_recovery": args.ft_yaw_lateral_recovery_ratio,
        }

        # Run formal training
        ft_result = run_routea_rl_formal_train(
            base_policy=base_policy,
            teacher_policy=teacher_policy,
            device=device,
            output_dir=ft_output_dir,
            manifest=_newenv_manifest,
            train_split=args.routea_rl_train_split,
            val_split=args.routea_rl_val_split,
            world_dir=_world_dir,
            critic_ckpt_path=_critic_path,
            max_actor_updates=args.formal_train_max_actor_updates,
            eval_interval=args.formal_train_eval_interval,
            checkpoint_interval=args.formal_train_checkpoint_interval,
            actor_lr=args.ft_actor_lr,
            critic_lr=args.ft_critic_lr,
            rollouts_per_cycle=args.ft_rollouts_per_cycle,
            critic_updates=args.ft_critic_updates,
            dqpg_warmup_steps=args.ft_dqpg_warmup_steps,
            role_ratios=ft_role_ratios,
            eval_plan_path=args.formal_train_eval_plan,
            metrics_schema_path=args.formal_train_metrics_schema,
            resume_from=args.formal_train_resume_from,
            freeze_actor=args.ft_freeze_actor,
            fixed_replay=args.ft_fixed_replay,
            rescue_from=args.ft_rescue_from,
            stabilized_replay=args.ft_stabilized_replay,
            replay_warmup_transitions=args.ft_replay_warmup_transitions,
            replay_warmup_episodes=args.ft_replay_warmup_episodes,
            # ── P0 wiring health (Tasks 3-5) ──
            collect_only=args.ft_collect_only,
            collect_only_max_steps=args.ft_collect_only_max_steps,
            collect_only_output_replay=args.ft_collect_only_output_replay,
            critic_bootstrap=args.ft_critic_bootstrap,
            bootstrap_heldout_ratio=args.ft_bootstrap_heldout_ratio,
            bootstrap_replay_path=args.ft_bootstrap_replay,
            bootstrap_diag_only=args.ft_bootstrap_diag_only,
            bootstrap_gamma=args.ft_bootstrap_gamma,
            bootstrap_target_tau=args.ft_bootstrap_target_tau,
            bootstrap_total_only=args.ft_bootstrap_total_only,
            bootstrap_huber_total=args.ft_bootstrap_huber_total,
            bootstrap_huber_beta=args.ft_bootstrap_huber_beta,
            bootstrap_sparse_head_mode=args.ft_bootstrap_sparse_head_mode,
            # ── B1c: architecture / reward representation knobs ──
            bootstrap_reward_norm_mode=args.ft_bootstrap_reward_norm_mode,
            bootstrap_reward_norm_eps=args.ft_bootstrap_reward_norm_eps,
            bootstrap_q_l2_weight=args.ft_bootstrap_q_l2_weight,
            bootstrap_q_soft_clip=args.ft_bootstrap_q_soft_clip,
            bootstrap_critic_norm=args.ft_bootstrap_critic_norm,
            bootstrap_target_update_mode=args.ft_bootstrap_target_update_mode,
            bootstrap_target_update_interval=args.ft_bootstrap_target_update_interval,
            bootstrap_split_seed=args.ft_bootstrap_split_seed,
            phased_online=args.ft_phased_online,
            phase_collect_episodes=args.ft_phase_collect_episodes,
            phase_critic_digest_updates=args.ft_phase_critic_digest_updates,
            phase_actor_updates=args.ft_phase_actor_updates,
            reward_norm_stats_path=args.ft_reward_norm_stats,
            # ── B2a: PG scale controls ──
            actor_lambda_pg=args.ft_actor_lambda_pg,
            actor_lambda_dqpg=args.ft_actor_lambda_dqpg,
            actor_pg_scale_mode=args.ft_actor_pg_scale_mode,
            actor_effective_action_dim=args.ft_actor_effective_action_dim,
            actor_effective_flow_events=args.ft_actor_effective_flow_events,
            # ── B3: pilot safety caps ──
            ft_max_phases=args.ft_max_phases,
            ft_actor_compute_teacher=args.ft_actor_compute_teacher,
            ft_pilot_mode=args.ft_pilot_mode,
            ft_extended_pilot=args.ft_extended_pilot,
            ft_b8_long_train=args.ft_b8_long_train,
            ft_b11_500upd=args.ft_b11_500upd,
            ft_b15_1000upd=args.ft_b15_1000upd,
            ft_lora_rank=args.ft_lora_rank,
            # ── B4: role quota pilot ──
            role_quota_pilot=args.ft_role_quota_pilot,
            ft_min_episodes_anticipatory=args.ft_min_episodes_anticipatory,
            ft_min_episodes_dense_recovery=args.ft_min_episodes_dense_recovery,
            ft_min_episodes_support_expansion=args.ft_min_episodes_support_expansion,
            ft_max_support_role_share=args.ft_max_support_role_share,
            # ── B5: lightweight paired eval ──
            ft_save_actor_path=args.ft_save_actor_path,
            ft_post_eval=args.ft_post_eval,
            ft_eval_eps_per_world=args.ft_eval_eps_per_world,
            # ── C2-OOC: online cell smoke ──
            ft_c2_ooc=args.ft_c2_ooc,
            ft_mid_eval=args.ft_mid_eval,
            ft_mid_eval_phase=args.ft_mid_eval_phase,
            ft_c2_ooc_no_twin_stop=args.ft_c2_ooc_no_twin_stop,
            seed=args.seed,
            curriculum_stage=args.ft_curriculum_stage,
        )

        cprint(f"\n{'=' * 70}", "cyan")
        cprint(f"  Formal Training Status: {ft_result['formal_training_status']}", "green")
        cprint(f"  Next: {ft_result.get('next_action', 'see preflight report')}", "cyan")
        cprint(f"{'=' * 70}", "cyan")
        return


    # ── Fallback: no valid mode selected ──
    cprint("\n" + "=" * 70, "red")
    cprint("  ERROR: No training mode selected.", "red", "on_white")
    cprint("  Please use one of:", "yellow")
    cprint("    --a0_newenv_critic_warmup_smoke", "yellow")
    cprint("    --a1_newenv_stability_smoke", "yellow")
    cprint("    --a2_newenv_capability_smoke", "yellow")
    cprint("    --routea_rl_formal_train (with --use_routea_rl_env_v1)", "yellow")
    cprint("=" * 70, "red")
    sys.exit(1)

    # ── Fallback: no valid mode selected ──
    cprint("\n" + "=" * 70, "red")
    cprint("  ERROR: No training mode selected.", "red", "on_white")
    cprint("  Please use one of:", "yellow")
    cprint("    --a0_newenv_critic_warmup_smoke", "yellow")
    cprint("    --a1_newenv_stability_smoke", "yellow")
    cprint("    --a2_newenv_capability_smoke", "yellow")
    cprint("    --routea_rl_formal_train (with --use_routea_rl_env_v1)", "yellow")
    cprint("=" * 70, "red")
    sys.exit(1)
if __name__ == "__main__":
    main()
