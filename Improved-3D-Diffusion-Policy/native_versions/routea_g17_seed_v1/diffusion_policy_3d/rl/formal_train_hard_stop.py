#!/usr/bin/env python3
"""
formal_train_hard_stop.py — Formal Training Hard Stop Checker

Design: zhidao/Guide_RouteA_RL_FormalTraining_v1.md §Step 9

Monitors Route A RL v1 formal training for conditions that require
immediate stop + emergency checkpoint. Independent of the old
HardStopChecker which is tied to legacy scenario taxonomy.

Checks (any single trigger → hard stop):
  1. NaN/Inf in any loss or gradient
  2. Historical contamination > 0
  3. Metadata completeness < 1.0
  4. Actor grad norm post-clip > 1.05 for repeated windows
  5. Critic Q_std > configured limit for repeated windows
  6. Twin Q gap > configured limit for repeated windows
  7. Collision rate explosion in any formal role (vs baseline)
  8. support_expansion degenerates to z-only (no lateral coverage)
  9. Reward hacking: reward rises while SR/CR all worsen
  10. Base FM / obs encoder / teacher hash changes unexpectedly
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════
# Default thresholds
# ═══════════════════════════════════════════════════════════════════

DEFAULT_WINDOW_SIZE = 3  # consecutive eval windows for trend checks
DEFAULT_ACTOR_GRAD_THRESHOLD = 1.05
DEFAULT_QSTD_LIMIT = 22.0
DEFAULT_TWIN_GAP_LIMIT = 0.55
DEFAULT_COLLISION_RATE_GUARD = 0.30  # max allowed CR increase vs baseline (30pp)


@dataclass
class FormalTrainHardStopConfig:
    """Configurable thresholds for formal training hard stop."""
    window_size: int = DEFAULT_WINDOW_SIZE
    actor_grad_threshold: float = DEFAULT_ACTOR_GRAD_THRESHOLD
    qstd_limit: float = DEFAULT_QSTD_LIMIT
    twin_gap_limit: float = DEFAULT_TWIN_GAP_LIMIT
    collision_rate_guard: float = DEFAULT_COLLISION_RATE_GUARD
    # Per-role baseline CR (from 1080ep paired rollout) for CR explosion detection
    baseline_cr: Dict[str, float] = field(default_factory=lambda: {
        "anticipatory": 0.324,
        "dense_recovery": 0.833,
        "support_expansion": 0.733,
        "yaw_lateral_recovery": 0.829,
    })
    # Min low_density_coverage for support_expansion to avoid z-only
    z_only_coverage_threshold: float = 0.05


class FormalTrainHardStopChecker:
    """Monitor hard stop conditions for Route A RL v1 formal training.

    Usage:
        checker = FormalTrainHardStopChecker(config)
        for step in range(n_steps):
            metrics = {...}  # per-step training metrics
            eval_metrics = {...}  # per-role eval metrics (only at eval steps)
            stop_reason = checker.check(step, metrics, eval_metrics)
            if stop_reason:
                print(f"HARD STOP: {stop_reason}")
                break
    """

    def __init__(self, config: Optional[FormalTrainHardStopConfig] = None,
                 base_fm_hash: str = "", obs_encoder_hash: str = "",
                 teacher_hash: str = ""):
        self.cfg = config or FormalTrainHardStopConfig()
        self.window_size = self.cfg.window_size

        # Sliding windows
        self.actor_grad_window: List[float] = []
        self.qstd_window: List[float] = []
        self.twin_gap_window: List[float] = []
        self.reward_window: List[float] = []
        self.sr_window: Dict[str, List[float]] = defaultdict(list)
        self.cr_window: Dict[str, List[float]] = defaultdict(list)

        # Baseline hashes for freeze verification
        self.base_fm_hash = base_fm_hash
        self.obs_encoder_hash = obs_encoder_hash
        self.teacher_hash = teacher_hash

        # Track consecutive bad windows
        self.consecutive_actor_grad_bad = 0
        self.consecutive_qstd_bad = 0
        self.consecutive_twin_gap_bad = 0

        # Store all triggered stops for reporting
        self.stop_history: List[Dict[str, Any]] = []

    def check(self, step: int, metrics: Dict[str, Any],
              eval_metrics: Optional[Dict[str, Any]] = None,
              metadata_stats: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Check all hard stop conditions. Returns stop reason or None.

        Args:
            step: Current training step (0-indexed).
            metrics: Per-step training metrics dict with keys:
                - nan_inf_count, actor_grad_norm_post_clip,
                - critic_Q_std, critic_twin_gap, reward_total
            eval_metrics: Per-role eval dict (only at eval steps). Keys are role names.
            metadata_stats: Dict with contamination_count, completeness_ratio.
        """
        reasons: List[str] = []

        # ── 1. NaN/Inf ──
        nan_count = metrics.get("nan_count", metrics.get("nan_inf_count", 0))
        inf_count = metrics.get("inf_count", 0)
        if nan_count > 0 or inf_count > 0:
            reasons.append(f"NaN={nan_count}, Inf={inf_count}")

        # ── 2. Historical contamination ──
        if metadata_stats:
            contamination = metadata_stats.get("contamination_count", 0)
            if contamination > 0:
                reasons.append(f"historical_contamination={contamination}")

        # ── 3. Metadata completeness ──
        if metadata_stats:
            completeness = metadata_stats.get("completeness_ratio", 1.0)
            if completeness < 1.0:
                reasons.append(f"metadata_completeness={completeness:.4f} < 1.0")

        # ── 4. Actor grad norm post-clip ──
        actor_grad = metrics.get("actor_grad_norm_post_clip", 0.0)
        self.actor_grad_window.append(actor_grad)
        if len(self.actor_grad_window) >= self.window_size:
            recent = self.actor_grad_window[-self.window_size:]
            if all(g > self.cfg.actor_grad_threshold for g in recent):
                self.consecutive_actor_grad_bad += 1
                if self.consecutive_actor_grad_bad >= 2:
                    reasons.append(
                        f"actor_grad > {self.cfg.actor_grad_threshold} for "
                        f"{self.consecutive_actor_grad_bad + self.window_size - 1} windows"
                    )
            else:
                self.consecutive_actor_grad_bad = 0

        # ── 5. Critic Q_std ──
        q_std = metrics.get("critic_Q_std", metrics.get("q_std", 0.0))
        self.qstd_window.append(q_std)
        if q_std > self.cfg.qstd_limit:
            self.consecutive_qstd_bad += 1
            if self.consecutive_qstd_bad >= self.window_size:
                reasons.append(f"Q_std > {self.cfg.qstd_limit} for {self.window_size} windows "
                               f"(last={q_std:.1f})")
        else:
            self.consecutive_qstd_bad = 0

        # ── 6. Twin Q gap ──
        twin_gap = metrics.get("critic_twin_gap", metrics.get("twin_gap", 0.0))
        self.twin_gap_window.append(twin_gap)
        if twin_gap > self.cfg.twin_gap_limit:
            self.consecutive_twin_gap_bad += 1
            if self.consecutive_twin_gap_bad >= self.window_size:
                reasons.append(f"twin_gap > {self.cfg.twin_gap_limit} for {self.window_size} windows "
                               f"(last={twin_gap:.3f})")
        else:
            self.consecutive_twin_gap_bad = 0

        # ── 7. Collision rate explosion (per role, at eval steps) ──
        if eval_metrics:
            for role in ["anticipatory", "dense_recovery", "support_expansion"]:
                role_eval = eval_metrics.get(role, {})
                cr = role_eval.get("CR", None)
                if cr is not None:
                    baseline = self.cfg.baseline_cr.get(role, 0.5)
                    if cr > baseline + self.cfg.collision_rate_guard:
                        reasons.append(
                            f"{role} CR={cr:.3f} > baseline {baseline:.3f} + "
                            f"{self.cfg.collision_rate_guard}"
                        )
                    self.cr_window[role].append(cr)

        # ── 8. support_expansion z-only degeneration ──
        if eval_metrics:
            se_eval = eval_metrics.get("support_expansion", {})
            coverage = se_eval.get("low_density_coverage",
                                   se_eval.get("successful_low_density_coverage", None))
            if coverage is not None and coverage < self.cfg.z_only_coverage_threshold:
                sr = se_eval.get("SR", 0.0)
                if sr > 0.1:  # only flag if SR is non-trivial (not just failing everything)
                    reasons.append(
                        f"support_expansion coverage={coverage:.4f} < "
                        f"{self.cfg.z_only_coverage_threshold} (z-only, SR={sr:.3f})"
                    )

        # ── 9. Reward hacking: reward rises while SR/CR all worsen ──
        reward_total = metrics.get("reward_total",
                                   metrics.get("reward_mean", None))
        if reward_total is not None:
            self.reward_window.append(float(reward_total))
            if eval_metrics and len(self.reward_window) >= self.window_size:
                # Check reward trend
                recent_reward = self.reward_window[-self.window_size:]
                reward_rising = all(
                    recent_reward[i] < recent_reward[i + 1]
                    for i in range(len(recent_reward) - 1)
                )
                if reward_rising:
                    # Check SR trend for all formal roles
                    all_worsening = True
                    for role in ["anticipatory", "dense_recovery", "support_expansion"]:
                        sr_hist = self.sr_window.get(role, [])
                        if len(sr_hist) >= self.window_size:
                            recent_sr = sr_hist[-self.window_size:]
                            if not all(recent_sr[i] >= recent_sr[i + 1]
                                       for i in range(len(recent_sr) - 1)):
                                all_worsening = False
                                break
                        else:
                            all_worsening = False
                            break
                    if all_worsening:
                        reasons.append(
                            f"reward_hacking: reward rising {self.window_size} windows "
                            f"while all formal role SR worsening"
                        )

            # Track per-role SR for reward hacking check
            if eval_metrics:
                for role in ["anticipatory", "dense_recovery", "support_expansion"]:
                    role_eval = eval_metrics.get(role, {})
                    sr = role_eval.get("SR", None)
                    if sr is not None:
                        self.sr_window[role].append(float(sr))

        # ── 10. Freeze boundary verification ──
        # Checked externally (model hash comparison), reported via
        # the 'freeze_violation' flag in metrics. If flag is set, stop.
        if metrics.get("freeze_violation", False):
            reasons.append("base FM / obs_encoder / teacher hash changed unexpectedly")

        # ── Trim windows ──
        max_window = self.window_size * 3
        for w in [self.actor_grad_window, self.qstd_window, self.twin_gap_window,
                  self.reward_window]:
            while len(w) > max_window:
                w.pop(0)
        for role_w in list(self.sr_window.values()) + list(self.cr_window.values()):
            while len(role_w) > max_window:
                role_w.pop(0)

        # ── Compose result ──
        if reasons:
            stop_reason = "; ".join(reasons)
            self.stop_history.append({
                "step": step,
                "reasons": reasons,
                "metrics_snapshot": {
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                    "actor_grad": actor_grad,
                    "q_std": q_std,
                    "twin_gap": twin_gap,
                },
            })
            return stop_reason
        return None

    def get_stop_report(self) -> Dict[str, Any]:
        """Generate a summary report of all hard stop events."""
        return {
            "total_stops": len(self.stop_history),
            "stop_events": self.stop_history,
            "window_state": {
                "actor_grad_window": self.actor_grad_window[-self.window_size:] if self.actor_grad_window else [],
                "qstd_window": self.qstd_window[-self.window_size:] if self.qstd_window else [],
                "twin_gap_window": self.twin_gap_window[-self.window_size:] if self.twin_gap_window else [],
            },
            "config": {
                "window_size": self.cfg.window_size,
                "actor_grad_threshold": self.cfg.actor_grad_threshold,
                "qstd_limit": self.cfg.qstd_limit,
                "twin_gap_limit": self.cfg.twin_gap_limit,
                "collision_rate_guard": self.cfg.collision_rate_guard,
            },
        }


def compute_model_hash(model: Any, num_samples: int = 100) -> str:
    """Compute a lightweight hash of model parameters for freeze verification.

    Samples the first num_samples parameters to keep this fast.
    """
    hasher = hashlib.sha256()
    count = 0
    for name, param in model.named_parameters():
        if count >= num_samples:
            break
        hasher.update(name.encode())
        hasher.update(param.data.cpu().numpy().tobytes())
        count += 1
    return hasher.hexdigest()[:16]


def verify_freeze_boundary(actor: Any, base_fm_hash: str,
                           obs_encoder_hash: str,
                           teacher_hash: str) -> Tuple[bool, List[str]]:
    """Verify that frozen components haven't changed.

    Returns:
        (passed, violations): True if all hashes match, list of violation descriptions.
    """
    violations = []

    # Check base FM
    current_fm_hash = compute_model_hash(actor.base_policy.model)
    if base_fm_hash and current_fm_hash != base_fm_hash:
        violations.append(f"base_fm hash changed: {base_fm_hash[:8]} → {current_fm_hash[:8]}")

    # Check obs encoder
    current_enc_hash = compute_model_hash(actor.base_policy.obs_encoder)
    if obs_encoder_hash and current_enc_hash != obs_encoder_hash:
        violations.append(f"obs_encoder hash changed: {obs_encoder_hash[:8]} → {current_enc_hash[:8]}")

    # Check teacher
    if hasattr(actor, 'teacher_policy') and actor.teacher_policy is not None:
        current_teacher_hash = compute_model_hash(actor.teacher_policy.model)
        if teacher_hash and current_teacher_hash != teacher_hash:
            violations.append(f"teacher hash changed: {teacher_hash[:8]} → {current_teacher_hash[:8]}")

    return len(violations) == 0, violations
