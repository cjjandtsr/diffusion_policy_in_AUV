"""
p0_4_instrumentation.py — P0-4: Short-train instrumentation interface (§8.4)

训练日志模块，支持能力归因。设计为 mixin/middleware 模式，不依赖具体训练循环实现。

Required logs (per §8.4):
  critic_loss_total, critic_loss_heads/*
  actor_loss_pg, actor_loss_bc, actor_loss_path, actor_loss_noise
  noise_sigma_mean/std/min/max, noise_sigma_per_fm_step
  adapter_grad_norm, adapter_grad_norm_per_layer
  Q_total_mean, Q_head_means/*, Q_head_std/*, Q_head_minmax/*
  reward_terms_episode_sum/*
  onset_distance/time, future_clearance_softmin, trap_recovery_rate
  deadend_dwell_steps, out_of_support_rate, successful_low_density_coverage
  path_length_ratio, loop_detected_rate

用法:
  logger = RouteATrainingLogger(log_dir=...)
  logger.log_train_step(step, {
      'critic_loss_total': ...,
      'actor_loss_pg': ...,
      'noise_sigma_mean': ...,
      ...
  })
  logger.log_episode(ep_data)  # episode结束时调用
  logger.log_epoch(epoch, metrics)  # epoch结束时调用
  logger.save()
"""

import json, os, time, numpy as np
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════
# Log keys (一对一映射 §8.4 要求)
# ═══════════════════════════════════════════════════════════════════

REQUIRED_TRAIN_LOG_KEYS = [
    "critic_loss_total",
    "actor_loss_pg",
    "actor_loss_bc",
    "actor_loss_path",
    "actor_loss_noise",
    "noise_sigma_mean",
    "noise_sigma_std",
    "noise_sigma_min",
    "noise_sigma_max",
    "adapter_grad_norm",
    "Q_total_mean",
]

REQUIRED_EPISODE_LOG_KEYS = [
    "reward_terms_episode_sum",
    "onset_distance",
    "onset_time",
    "future_clearance_softmin",
    "trap_recovery_rate",
    "deadend_dwell_steps",
    "out_of_support_rate",
    "successful_low_density_coverage",
    "path_length_ratio",
    "loop_detected_rate",
]


class RouteATrainingLogger:
    """Training instrumentation for Route A-RF short-train.

    Manages two log streams:
      - train_log: per-step scalar metrics
      - episode_log: per-episode aggregated metrics
      - epoch_log: per-epoch aggregated metrics
    """

    def __init__(self, log_dir: str, log_freq: int = 10,
                 tensorboard: bool = False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_freq = log_freq
        self.tensorboard = tensorboard

        # In-memory buffers
        self._train_history: List[Dict] = []
        self._episode_history: List[Dict] = []
        self._epoch_metrics: Dict[int, Dict] = {}

        # Running accumulators for epoch-level aggregation
        self._epoch_step_count = 0
        self._epoch_step_accum = defaultdict(float)
        self._epoch_step_sq_accum = defaultdict(float)

        # TensorBoard writer (lazy init)
        self._tb_writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(str(self.log_dir))
            except ImportError:
                print("[P0-4] tensorboard not available, falling back to JSON-only")

        self._start_time = time.time()
        self._last_log_time = time.time()

    # ── Per-step logging ──

    def log_train_step(self, step: int, metrics: Dict, force: bool = False):
        """Log one training step's metrics.

        Args:
            step: Global training step
            metrics: Dict with keys from REQUIRED_TRAIN_LOG_KEYS (and any extras)
            force: If True, flush to history even if step % log_freq != 0
        """
        # Validate required keys
        for k in REQUIRED_TRAIN_LOG_KEYS:
            if k not in metrics:
                pass  # Allow missing keys (may be absent before component is wired)

        # Accumulate for epoch stats
        self._epoch_step_count += 1
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.number)):
                self._epoch_step_accum[k] += float(v)
                self._epoch_step_sq_accum[k] += float(v) ** 2

        # Write to history at log_freq
        if force or (step % self.log_freq == 0):
            entry = {"step": step, **metrics}
            self._train_history.append(entry)

            if self._tb_writer:
                for k, v in metrics.items():
                    if isinstance(v, (int, float, np.number)):
                        self._tb_writer.add_scalar(f"train/{k}", v, step)

            self._last_log_time = time.time()

    def log_noise_sigma_per_step(self, step: int, sigma_per_step: np.ndarray):
        """Log per-FM-step noise sigma (shape: (num_fm_steps,)).

        Special handling: log as separate scalar per step index + histogram.
        """
        self.log_train_step(step, {
            f"noise_sigma_fm_step_{i}": float(sigma_per_step[i])
            for i in range(len(sigma_per_step))
        })

    def log_q_head_stats(self, step: int, q_means: List[float],
                         q_stds: List[float], q_mins: List[float],
                         q_maxs: List[float]):
        """Log per-head Q statistics.

        Args:
            q_means: [mean_Q1, mean_Q2, ...] per head
            q_stds: [std_Q1, std_Q2, ...]
            q_mins: [min_Q1, min_Q2, ...]
            q_maxs: [max_Q1, max_Q2, ...]
        """
        metrics = {}
        n_heads = len(q_means)
        for i in range(n_heads):
            metrics[f"Q_head_mean_{i}"] = q_means[i]
            metrics[f"Q_head_std_{i}"] = q_stds[i]
            metrics[f"Q_head_min_{i}"] = q_mins[i]
            metrics[f"Q_head_max_{i}"] = q_maxs[i]
        metrics["Q_total_mean"] = float(np.mean(q_means))
        self.log_train_step(step, metrics)

    def log_adapter_grad_norms(self, step: int,
                               grad_norms: Dict[str, float]):
        """Log per-layer adapter gradient norms.

        Args:
            grad_norms: {layer_name: grad_norm_value}
        """
        metrics = {"adapter_grad_norm": float(np.mean(list(grad_norms.values())))}
        for name, norm in grad_norms.items():
            safe_name = name.replace(".", "_")
            metrics[f"adapter_grad_norm_{safe_name}"] = norm
        self.log_train_step(step, metrics)

    # ── Episode-level logging ──

    def log_episode(self, ep_data: Dict):
        """Log one completed episode's aggregated metrics.

        Args:
            ep_data: dict with keys:
                - reward_terms: Dict[str, float] (episode sum per term)
                - metrics: Dict[str, float] (diagnostic metrics)
                - reached / collided / done_reason
        """
        entry = {
            "reward_terms_episode_sum": ep_data.get("reward_terms", {}),
            "onset_distance": ep_data.get("onset_distance"),
            "onset_time": ep_data.get("onset_time"),
            "future_clearance_softmin": ep_data.get("future_clearance_softmin"),
            "trap_recovery_rate": ep_data.get("trap_recovery_rate"),
            "deadend_dwell_steps": ep_data.get("deadend_dwell_steps"),
            "out_of_support_rate": ep_data.get("out_of_support_rate"),
            "successful_low_density_coverage": ep_data.get("successful_low_density_coverage"),
            "path_length_ratio": ep_data.get("path_length_ratio"),
            "loop_detected_rate": ep_data.get("loop_detected_rate"),
            "reached": ep_data.get("reached"),
            "collided": ep_data.get("collided"),
            "done_reason": ep_data.get("done_reason"),
            "reward_total": ep_data.get("reward_total"),
            "length": ep_data.get("length"),
        }
        self._episode_history.append(entry)

        if self._tb_writer:
            ep_idx = len(self._episode_history) - 1
            for k, v in entry.items():
                if isinstance(v, (int, float, np.number)):
                    self._tb_writer.add_scalar(f"episode/{k}", v, ep_idx)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float, np.number)):
                            self._tb_writer.add_scalar(f"episode/reward_{kk}", vv, ep_idx)

    # ── Epoch-level aggregation ──

    def finish_epoch(self, epoch: int, extra_metrics: Optional[Dict] = None):
        """Aggregate step-level metrics into epoch stats and reset accumulators.

        Args:
            epoch: Epoch index
            extra_metrics: Dict of additional epoch-level metrics
        """
        n = max(self._epoch_step_count, 1)
        epoch_data = OrderedDict([("epoch", epoch), ("steps_in_epoch", self._epoch_step_count)])

        for k in sorted(self._epoch_step_accum.keys()):
            mean_v = self._epoch_step_accum[k] / n
            var_v = (self._epoch_step_sq_accum[k] / n) - mean_v ** 2
            epoch_data[f"{k}_mean"] = round(float(mean_v), 6)
            epoch_data[f"{k}_std"] = round(float(max(var_v, 0) ** 0.5), 6)

        if extra_metrics:
            epoch_data.update(extra_metrics)

        # Episode aggregation
        if self._episode_history:
            n_ep = len(self._episode_history)
            epoch_data["n_episodes"] = n_ep
            epoch_data["SR"] = round(
                float(np.mean([e["reached"] for e in self._episode_history if e["reached"] is not None])), 4)
            epoch_data["CR"] = round(
                float(np.mean([e["collided"] for e in self._episode_history if e["collided"] is not None])), 4)

            # Reward term means
            term_keys = set()
            for e in self._episode_history:
                term_keys.update(e.get("reward_terms_episode_sum", {}).keys())
            for tk in sorted(term_keys):
                vals = [e["reward_terms_episode_sum"].get(tk, 0) for e in self._episode_history]
                epoch_data[f"reward_{tk}_mean"] = round(float(np.mean(vals)), 4)

        self._epoch_metrics[epoch] = epoch_data
        self._reset_epoch_accumulators()

        if self._tb_writer:
            for k, v in epoch_data.items():
                if isinstance(v, (int, float, np.number)):
                    self._tb_writer.add_scalar(f"epoch/{k}", v, epoch)

        # Save checkpoint every epoch
        self.save(epoch=epoch)
        return epoch_data

    def _reset_epoch_accumulators(self):
        self._epoch_step_count = 0
        self._epoch_step_accum.clear()
        self._epoch_step_sq_accum.clear()

    # ── Persistence ──

    def save(self, epoch: Optional[int] = None):
        """Save log state to JSON."""
        # Train history (last 5000 entries to cap file size)
        train_trimmed = self._train_history[-5000:] if len(self._train_history) > 5000 else self._train_history

        state = {
            "meta": {
                "log_type": "Route A-RF training instrumentation",
                "n_train_steps_logged": len(self._train_history),
                "n_episodes_logged": len(self._episode_history),
                "n_epochs": len(self._epoch_metrics),
                "start_time": self._start_time,
                "last_log_time": self._last_log_time,
            },
            "train_log": train_trimmed,
            "episode_log": self._episode_history,
            "epoch_metrics": {str(k): v for k, v in self._epoch_metrics.items()},
        }

        fname = f"instrumentation_log{'' if epoch is None else f'_epoch{epoch}'}.json"
        path = self.log_dir / fname
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        return path

    def close(self):
        """Final save and cleanup."""
        self.save()
        if self._tb_writer:
            self._tb_writer.close()
