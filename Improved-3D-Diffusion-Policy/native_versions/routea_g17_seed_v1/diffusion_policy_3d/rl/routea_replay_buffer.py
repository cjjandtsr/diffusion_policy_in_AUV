"""
routea_replay_buffer.py — Route A-RF Replay Buffer (§6)

Three-partition replay buffer with chunk-structured transitions:

    B_offline_prior (30%): frozen dataset / BC rollout feature-action pairs
    B_online_recent  (50%): current actor's recent N episodes
    B_frontier_event (20%): rare success/failure events

A0 mode: all partitions use FIFO, uniform ratio sampling.
A1+ upgrade path: priority-based sampling for B_frontier_event.

Design: RouteA_ReinFlow_FM_RL_Design_v1_2.md §6
"""

import random
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Transition fields per §6.2
REWARD_TERM_NAMES = [
    "base", "pbrs", "onset", "future_clearance",
    "trap", "flip", "support", "guard",
]


@dataclass
class RouteATransition:
    """Single RL transition for Route A-RF.

    RL step: H=8, s_t -> action_chunk(8,11) -> s_t+H, reward over chunk.
    Bootstrap uses action_chunk_tp1 (A_{t+H}) for TD target Q(s_{t+H}, A_{t+H}).
    """
    obs_feature_t: np.ndarray          # (1536,) global_cond at s_t
    action_chunk: np.ndarray           # (8, 11) action chunk A_t from policy
    action_chunk_tp1: np.ndarray       # (8, 11) next action A_{t+H} for bootstrap TD target
    reward_total: float                # sum of reward_terms
    reward_terms: Dict[str, float]     # {base, pbrs, onset, ..., guard}
    obs_feature_tp1: np.ndarray        # (1536,) global_cond at s_t+H
    done: bool                         # terminal flag

    # ── Metadata (for diagnostics/filtering, not required by critic) ──
    done_reason: str = ""
    world_id: int = -1
    episode_seed: int = -1
    start: Optional[np.ndarray] = None        # (3,) start pos
    goal: Optional[np.ndarray] = None          # (3,) goal pos
    path_length_so_far: float = 0.0
    reward_metrics: Optional[Dict] = None
    event_tags: Dict[str, bool] = field(default_factory=dict)
    # ── R3: Dense/frontier replay metadata (§Step 5.1) ──
    collision: bool = False
    min_clearance: float = float('inf')
    episode_collision: bool = False
    episode_success: bool = False
    scenario: str = ""
    # ── Route A RL v1: New-env metadata (§Step 6) ──
    world_uid: str = ""                     # e.g. "routeA_rl_v1__anticipatory__rl_train__front_offset__000000"
    split: str = ""                         # rl_train / rl_val / rl_test / stress_only / historical_eval_only
    role_generated: str = ""                # role from world generator
    role_validated: str = ""                # role after L1/L2/L3 validation
    role_active: str = ""                   # role after calibration admission
    generator_family: str = ""              # e.g. "front_offset_obstacle"
    paired_seed_group: str = ""             # seed group identifier
    admission_status: str = ""              # e.g. "admitted_anticipatory" / "needs_more_eval"
    # ── Effective Replay Stabilization: Hard/frontier sample diagnostics (§Step 2) ──
    is_hard_sample: bool = False            # marked as hard/frontier sample
    hard_reason: str = ""                   # e.g. "collision", "low_clearance", "high_td_error", "low_return"
    frontier_score: float = 0.0             # continuous frontier score (higher = harder)
    td_target_abs: float = 0.0              # |TD target| for this transition (set after critic update)
    td_error_abs: float = 0.0               # |TD error| for this transition (set after critic update)
    collision_or_violation: bool = False    # collision or safety violation in this step
    timeout_flag: bool = False              # episode ended by timeout
    replay_age_steps: int = 0               # age in training steps since insertion (0 = newest)

    def validate(self) -> bool:
        """Check transition integrity."""
        if self.obs_feature_t.shape != (1536,):
            return False
        if self.action_chunk.shape != (8, 11):
            return False
        if self.action_chunk_tp1.shape != (8, 11):
            return False
        if not np.isfinite(self.reward_total):
            return False
        for name in REWARD_TERM_NAMES:
            v = self.reward_terms.get(name, 0.0)
            if not np.isfinite(v):
                return False
        if self.obs_feature_tp1.shape != (1536,):
            return False
        return True


class RouteAReplayBuffer:
    """Three-partition replay buffer for Route A-RF.

    A0 mode:
      - B_online_recent is the primary partition (frozen actor rollout)
      - B_offline_prior and B_frontier_event are optional (can be empty)
      - Uniform sampling with configurable ratios
      - No priority/IS weighting

    A1+ upgrades:
      - B_frontier_event: priority-based sampling
      - B_offline_prior: pre-filled from frozen dataset re-encoded
    """

    def __init__(self,
                 capacity_online: int = 100000,
                 capacity_frontier: int = 20000,
                 capacity_offline: int = 50000):
        self.b_offline_prior = deque(maxlen=capacity_offline)
        self.b_online_recent = deque(maxlen=capacity_online)
        self.b_frontier_event = deque(maxlen=capacity_frontier)

    # ── Insertion ──

    def add_online(self, trans: RouteATransition):
        """Add to B_online_recent."""
        self.b_online_recent.append(trans)

    def add_offline(self, trans: RouteATransition):
        """Add to B_offline_prior."""
        self.b_offline_prior.append(trans)

    def add_frontier(self, trans: RouteATransition):
        """Add to B_frontier_event."""
        self.b_frontier_event.append(trans)

    def add(self, trans: RouteATransition, partition: str = "online"):
        """Add transition to specified partition."""
        if partition == "online":
            self.add_online(trans)
        elif partition == "offline":
            self.add_offline(trans)
        elif partition == "frontier":
            self.add_frontier(trans)
        else:
            raise ValueError(f"Unknown partition: {partition}")

    def add_episode(self,
                    transitions: List[RouteATransition],
                    partition: str = "online"):
        """Add a full episode's transitions."""
        for trans in transitions:
            self.add(trans, partition=partition)

    # ── Sampling ──

    def _sample_partition(self,
                          buffer: deque,
                          n: int) -> List[RouteATransition]:
        """Uniformly sample n items from a partition (without replacement)."""
        if len(buffer) == 0:
            return []
        k = min(n, len(buffer))
        return random.sample(list(buffer), k)

    def sample(self,
               batch_size: int,
               ratios: Tuple[float, float, float] = (0.3, 0.5, 0.2)
               ) -> Dict[str, torch.Tensor]:
        """Sample batch across partitions.

        Args:
            batch_size: Desired batch size
            ratios: (offline_ratio, online_ratio, frontier_ratio)
                    Default (0.3, 0.5, 0.2) matching §6.1.

        Returns:
            dict with keys matching compute_critic_loss() signature:
                state:         (B, 1536)
                action:        (B, 8, 11)
                reward_total:  (B,)
                reward_heads:  {name: (B,)}
                next_state:    (B, 1536)
                action_tp1:    (B, 8, 11)  — A_{t+H} for bootstrap TD target
                done:          (B,)
        """
        # Compute per-partition sample sizes
        n_offline = int(batch_size * ratios[0])
        n_online = int(batch_size * ratios[1])
        n_frontier = int(batch_size * ratios[2])

        # Round to exact batch_size
        remainder = batch_size - n_offline - n_online - n_frontier
        if remainder > 0:
            n_online += remainder

        # Sample from each partition
        offline_samples = self._sample_partition(self.b_offline_prior, n_offline)
        online_samples = self._sample_partition(self.b_online_recent, n_online)
        frontier_samples = self._sample_partition(self.b_frontier_event, n_frontier)

        all_samples: List[RouteATransition] = (
            offline_samples + online_samples + frontier_samples
        )

        # P2: Reallocate when partitions underfill (e.g. empty offline/frontier in A0).
        # Track already-sampled IDs to avoid duplicates in underfill pass.
        if len(all_samples) < batch_size:
            already_sampled = set(id(t) for t in all_samples)
            n_needed = batch_size - len(all_samples)
            extras = []
            for buf in [self.b_online_recent, self.b_offline_prior, self.b_frontier_event]:
                if len(buf) == 0 or n_needed <= 0:
                    continue
                pool = [t for t in buf if id(t) not in already_sampled]
                if len(pool) == 0:
                    # Still underfilled — accept duplicates rather than empty batch
                    pool = list(buf)
                k = min(n_needed, len(pool))
                sampled = random.sample(pool, k)
                extras.extend(sampled)
                already_sampled.update(id(t) for t in sampled)
                n_needed -= k
            all_samples.extend(extras)

        if len(all_samples) == 0:
            # Return empty batch with correct shapes
            device = torch.device("cpu")
            dummy = {
                "state": torch.zeros(0, 1536, device=device),
                "action": torch.zeros(0, 8, 11, device=device),
                "action_tp1": torch.zeros(0, 8, 11, device=device),
                "reward_total": torch.zeros(0, device=device),
                "reward_heads": {name: torch.zeros(0, device=device)
                                 for name in REWARD_TERM_NAMES},
                "next_state": torch.zeros(0, 1536, device=device),
                "done": torch.zeros(0, dtype=torch.bool, device=device),
            }
            return dummy

        return collate_to_critic(all_samples)

    # ── Effective Replay Stabilization: Stabilized sampling (Step 4) ──

    def add_episode_with_diagnostics(
        self,
        transitions: List[RouteATransition],
        scenario: str = "",
        episode_collision: bool = False,
        episode_success: bool = False,
    ):
        """Add episode transitions with hard/frontier sample tagging.

        Hard sample criteria (any transition meeting any):
          - collision_or_violation = true
          - min_clearance < 0.5 (dense threshold)
          - episode ended in collision
          - episode ended in timeout (potential deadend)

        Hard samples are retained, never silently filtered.
        """
        for t in transitions:
            t.scenario = scenario or t.scenario
            t.episode_collision = episode_collision
            t.episode_success = episode_success
            t.collision_or_violation = (
                getattr(t, 'collision', False)
                or episode_collision
                or getattr(t, 'done_reason', '') == 'collision'
            )
            t.timeout_flag = getattr(t, 'done_reason', '') == 'timeout'
            t.replay_age_steps = 0

            # Mark hard samples
            min_cl = getattr(t, 'min_clearance', float('inf'))
            if (t.collision_or_violation
                or (min_cl < 0.5 and min_cl >= 0)
                or (episode_collision and getattr(t, 'done', False))):
                t.is_hard_sample = True
                reasons = []
                if t.collision_or_violation:
                    reasons.append("collision")
                if min_cl < 0.5 and min_cl >= 0:
                    reasons.append("low_clearance")
                if episode_collision and getattr(t, 'done', False):
                    reasons.append("episode_collision_terminal")
                t.hard_reason = "+".join(reasons) if reasons else "unknown"
                t.frontier_score = 1.0 if t.collision_or_violation else 0.5

        # Add all to online partition
        for t in transitions:
            self.add_online(t)

        # Also add hard samples to frontier partition (retained, not filtered)
        hard_transitions = [t for t in transitions if t.is_hard_sample]
        for t in hard_transitions:
            self.add_frontier(t)

    def increment_ages(self):
        """Increment replay_age_steps for all transitions in all partitions."""
        for buf in [self.b_online_recent, self.b_offline_prior, self.b_frontier_event]:
            for t in buf:
                t.replay_age_steps += 1

    def get_role_breakdown(self) -> Dict:
        """Return per-role transition counts and hard sample ratios."""
        all_trans = (list(self.b_online_recent) + list(self.b_offline_prior)
                     + list(self.b_frontier_event))
        roles = defaultdict(lambda: {"count": 0, "hard_count": 0})
        for t in all_trans:
            role = getattr(t, 'role_active', '') or getattr(t, 'role_validated', '') or 'unknown'
            roles[role]["count"] += 1
            if getattr(t, 'is_hard_sample', False):
                roles[role]["hard_count"] += 1
        result = {}
        for role, stats in roles.items():
            total = max(stats["count"], 1)
            result[role] = {
                "transition_count": stats["count"],
                "hard_sample_count": stats["hard_count"],
                "hard_sample_ratio": round(stats["hard_count"] / total, 4),
            }
        return result

    def get_age_distribution(self) -> Dict:
        """Return age distribution with bins: new(<=200), mid(<=1000), old(>1000)."""
        all_trans = (list(self.b_online_recent) + list(self.b_offline_prior)
                     + list(self.b_frontier_event))
        bins = {"new": 0, "mid": 0, "old": 0}
        for t in all_trans:
            age = getattr(t, 'replay_age_steps', 0)
            if age <= 200:
                bins["new"] += 1
            elif age <= 1000:
                bins["mid"] += 1
            else:
                bins["old"] += 1
        total = max(sum(bins.values()), 1)
        return {
            "new_count": bins["new"],
            "mid_count": bins["mid"],
            "old_count": bins["old"],
            "new_ratio": round(bins["new"] / total, 4),
            "mid_ratio": round(bins["mid"] / total, 4),
            "old_ratio": round(bins["old"] / total, 4),
        }

    def get_all_transitions(self) -> List:
        """Return all transitions from all partitions (for persistence)."""
        return (list(self.b_online_recent) + list(self.b_offline_prior)
                + list(self.b_frontier_event))

    def add_transitions_batch(self,
                              transitions: List,
                              partition: str = "online"):
        """Add a batch of transitions (e.g. from disk) to a partition."""
        for trans in transitions:
            self.add(trans, partition=partition)

    def save(self, path: str):
        """Persist all transitions to disk via pickle."""
        import pickle
        all_trans = self.get_all_transitions()
        data = {
            "transitions": all_trans,
            "n_transitions": len(all_trans),
            "n_online": len(self.b_online_recent),
            "n_offline": len(self.b_offline_prior),
            "n_frontier": len(self.b_frontier_event),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str):
        """Load replay buffer from a pickle file."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        buf = cls(capacity_online=max(50000, data["n_transitions"]))
        buf.add_transitions_batch(data["transitions"], partition="online")
        return buf

    def is_warmup_complete(self,
                           min_transitions: int = 5000,
                           min_episodes: int = 100,
                           required_roles: Optional[set] = None) -> Tuple[bool, Dict]:
        """Check if replay warmup conditions are met.

        Returns:
            (complete, diagnostics_dict)
        """
        all_trans = (list(self.b_online_recent) + list(self.b_offline_prior)
                     + list(self.b_frontier_event))
        n_transitions = len(all_trans)

        # Count unique episodes by world_uid + episode_seed
        episodes = set()
        for t in all_trans:
            uid = getattr(t, 'world_uid', '')
            seed = getattr(t, 'episode_seed', -1)
            episodes.add(f"{uid}__{seed}")
        n_episodes = len(episodes)

        # Role coverage
        roles_present = set()
        for t in all_trans:
            role = getattr(t, 'role_active', '') or getattr(t, 'role_validated', '')
            if role:
                roles_present.add(role)

        required = required_roles or {"anticipatory", "dense_recovery", "support_expansion"}
        roles_covered = required.issubset(roles_present)

        hard_count = sum(1 for t in all_trans if getattr(t, 'is_hard_sample', False))
        hard_ratio = hard_count / max(n_transitions, 1)

        complete = (n_transitions >= min_transitions
                    and n_episodes >= min_episodes
                    and roles_covered
                    and hard_ratio > 0)

        return complete, {
            "n_transitions": n_transitions,
            "n_episodes": n_episodes,
            "min_transitions_required": min_transitions,
            "min_episodes_required": min_episodes,
            "roles_present": sorted(roles_present),
            "roles_required": sorted(required),
            "roles_covered": roles_covered,
            "hard_sample_count": hard_count,
            "hard_sample_ratio": round(hard_ratio, 4),
            "warmup_complete": complete,
        }

    def sample_stabilized(
        self,
        batch_size: int,
        config: Optional[Dict] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Stabilized replay batch sampler with role-balance + age-mix + hard retention.

        Composition targets:
          - Role-balanced across formal roles
          - Age mix: 25% new (<=200), 35% mid (201-1000), 40% old (>1000)
          - Hard sample retention: >=20% of batch if available, <=50% initially
          - World-balanced guard: warn if single world >50% of batch

        Falls back gracefully if age bins underfill, logging fallback reason.

        Returns:
            (batch_dict, composition_report)
        """
        cfg = config or {}
        target_hard_ratio = cfg.get("hard_sample_ratio", 0.20)
        max_hard_ratio = cfg.get("max_hard_ratio", 0.50)
        target_new_ratio = cfg.get("new_sample_ratio", 0.25)
        target_mid_ratio = cfg.get("mid_sample_ratio", 0.35)
        target_old_ratio = cfg.get("old_sample_ratio", 0.40)
        max_new_ratio = cfg.get("max_new_sample_ratio", 0.60)

        report = {
            "age_mix_fallback_reason": "",
            "actual_new_sample_ratio": 0.0,
            "actual_hard_sample_ratio": 0.0,
            "actual_role_ratio": {},
        }

        # Collect all transitions with metadata
        all_trans = (list(self.b_online_recent) + list(self.b_offline_prior)
                     + list(self.b_frontier_event))
        if len(all_trans) == 0:
            return self.sample(batch_size), report

        # Partition by role
        by_role = defaultdict(list)
        for t in all_trans:
            role = getattr(t, 'role_active', '') or getattr(t, 'role_validated', '') or 'unknown'
            by_role[role].append(t)

        formal_roles = ["anticipatory", "dense_recovery", "support_expansion"]
        active_roles = [r for r in formal_roles if r in by_role and len(by_role[r]) > 0]
        if not active_roles:
            active_roles = list(by_role.keys())

        # Allocate batch slots per role (balanced)
        per_role_n = max(1, batch_size // len(active_roles))
        allocations = {}
        remaining = batch_size
        for i, role in enumerate(active_roles):
            if i == len(active_roles) - 1:
                allocations[role] = remaining
            else:
                allocations[role] = min(per_role_n, remaining)
                remaining -= allocations[role]

        # Sample within each role with age-mix
        sampled = []
        fallback_reasons = []

        for role, n_target in allocations.items():
            pool = by_role[role]
            if len(pool) == 0:
                fallback_reasons.append(f"{role}_empty_pool")
                continue

            # Age-partition pool
            new_pool = [t for t in pool if getattr(t, 'replay_age_steps', 0) <= 200]
            mid_pool = [t for t in pool if 200 < getattr(t, 'replay_age_steps', 0) <= 1000]
            old_pool = [t for t in pool if getattr(t, 'replay_age_steps', 0) > 1000]

            n_new = int(n_target * target_new_ratio)
            n_mid = int(n_target * target_mid_ratio)
            n_old = int(n_target * target_old_ratio)

            # Cap new samples
            actual_new = min(n_new, len(new_pool))
            if actual_new < n_new and len(new_pool) > 0:
                fallback_reasons.append(f"{role}_new_underfill_{actual_new}/{n_new}")

            # Redistribute unfilled new slots
            shortfall_new = n_new - actual_new
            n_mid += shortfall_new // 2
            n_old += shortfall_new - (shortfall_new // 2)

            actual_mid = min(n_mid, len(mid_pool))
            actual_old = min(n_old, len(old_pool))

            # Sample from each age bin
            if actual_new > 0 and new_pool:
                sampled.extend(random.sample(new_pool, actual_new))
            if actual_mid > 0 and mid_pool:
                sampled.extend(random.sample(mid_pool, actual_mid))
            if actual_old > 0 and old_pool:
                sampled.extend(random.sample(old_pool, actual_old))

            total_sampled_role = actual_new + actual_mid + actual_old
            if total_sampled_role < n_target:
                already = set(id(t) for t in sampled)
                remaining_pool = [t for t in pool if id(t) not in already]
                extra_needed = n_target - total_sampled_role
                if remaining_pool:
                    extra = random.sample(remaining_pool, min(extra_needed, len(remaining_pool)))
                    sampled.extend(extra)
                    fallback_reasons.append(f"{role}_global_fallback_{len(extra)}")

        # Hard sample retention (use id-based lookup to avoid numpy eq issues)
        sampled_ids = set(id(t) for t in sampled)
        hard_in_sample = [t for t in sampled if getattr(t, 'is_hard_sample', False)]
        hard_available = [t for t in all_trans if getattr(t, 'is_hard_sample', False)
                          and id(t) not in sampled_ids]
        current_hard_ratio = len(hard_in_sample) / max(len(sampled), 1)

        if current_hard_ratio < target_hard_ratio and hard_available:
            n_hard_needed = int(len(sampled) * target_hard_ratio) - len(hard_in_sample)
            n_hard_max = int(len(sampled) * max_hard_ratio) - len(hard_in_sample)
            n_hard_add = min(n_hard_needed, len(hard_available))
            n_hard_add = min(n_hard_add, n_hard_max)
            if n_hard_add > 0:
                hard_to_add = random.sample(hard_available, n_hard_add)
                non_hard = [(i, t) for i, t in enumerate(sampled)
                            if not getattr(t, 'is_hard_sample', False)]
                n_replace = min(n_hard_add, len(non_hard))
                # Remove by index (descending to preserve indices)
                for idx, _ in sorted(non_hard[:n_replace], key=lambda x: -x[0]):
                    sampled.pop(idx)
                sampled.extend(hard_to_add[:n_replace])

        # World-balanced guard: warn if any world >50%
        world_counts = defaultdict(int)
        for t in sampled:
            wid = getattr(t, 'world_uid', '') or str(getattr(t, 'world_id', -1))
            world_counts[wid] += 1
        max_world_ratio = max(world_counts.values()) / max(len(sampled), 1)
        if max_world_ratio > 0.5:
            fallback_reasons.append(f"world_imbalance_{max_world_ratio:.2f}")

        # Build report
        if fallback_reasons:
            report["age_mix_fallback_reason"] = "; ".join(fallback_reasons)

        hard_final = [t for t in sampled if getattr(t, 'is_hard_sample', False)]
        new_final = [t for t in sampled if getattr(t, 'replay_age_steps', 0) <= 200]

        report["actual_new_sample_ratio"] = round(len(new_final) / max(len(sampled), 1), 4)
        report["actual_hard_sample_ratio"] = round(len(hard_final) / max(len(sampled), 1), 4)

        role_counts = defaultdict(int)
        for t in sampled:
            role = getattr(t, 'role_active', '') or getattr(t, 'role_validated', '') or 'unknown'
            role_counts[role] += 1
        for role, count in role_counts.items():
            report["actual_role_ratio"][role] = round(count / max(len(sampled), 1), 4)

        # Cap: log warning if new sample ratio exceeds max
        if report["actual_new_sample_ratio"] > max_new_ratio:
            report["age_mix_fallback_reason"] += (
                f"; new_sample_ratio_capped_{report['actual_new_sample_ratio']:.2f}"
            )

        # Ensure exact batch_size
        if len(sampled) > batch_size:
            sampled = random.sample(sampled, batch_size)
        elif len(sampled) < batch_size:
            sampled_ids = set(id(t) for t in sampled)
            remaining_pool = [t for t in all_trans if id(t) not in sampled_ids]
            if remaining_pool:
                extra = random.sample(remaining_pool,
                                      min(batch_size - len(sampled), len(remaining_pool)))
                sampled.extend(extra)

        if len(sampled) == 0:
            return self.sample(batch_size), report

        return collate_to_critic(sampled), report

    # ── Statistics ──

    def __len__(self) -> int:
        return len(self.b_online_recent) + len(self.b_offline_prior) + len(self.b_frontier_event)

    def get_stats(self) -> Dict:
        """Return partition sizes and composition."""
        n_offline = len(self.b_offline_prior)
        n_online = len(self.b_online_recent)
        n_frontier = len(self.b_frontier_event)
        total_real = n_offline + n_online + n_frontier
        denom = max(total_real, 1)  # avoid division by zero for ratios
        return {
            "n_offline_prior": n_offline,
            "n_online_recent": n_online,
            "n_frontier_event": n_frontier,
            "total_transitions": total_real,  # real count, not clamped
            "offline_ratio": round(n_offline / denom, 3),
            "online_ratio": round(n_online / denom, 3),
            "frontier_ratio": round(n_frontier / denom, 3),
        }

    # ── R3: Detailed stats (§Step 5.3) ──
    def get_detailed_stats(self) -> Dict:
        """Return detailed composition including dense/collision tag ratios."""
        base = self.get_stats()
        all_trans = list(self.b_online_recent) + list(self.b_offline_prior) + list(self.b_frontier_event)
        n_total = max(len(all_trans), 1)
        n_collision = sum(1 for t in all_trans if getattr(t, 'collision', False))
        n_dense = sum(1 for t in all_trans if getattr(t, 'scenario', '') == 'dense')
        n_frontier_tagged = sum(1 for t in all_trans if getattr(t, 'episode_collision', False)
                                or getattr(t, 'min_clearance', float('inf')) < 1.0)
        base["collision_tag_ratio"] = round(n_collision / n_total, 4)
        base["dense_tag_ratio"] = round(n_dense / n_total, 4)
        base["frontier_tagged_count"] = n_frontier_tagged
        base["frontier_tagged_ratio"] = round(n_frontier_tagged / n_total, 4)
        return base

    # ── R3: Add episode with frontier flagging (§Step 5.2) ──
    def add_episode_with_frontier(
        self,
        transitions: List[RouteATransition],
        scenario: str = "",
        episode_collision: bool = False,
        episode_success: bool = False,
        dense_min_clearance_threshold: float = 1.0,
        collision_frontier_boost: bool = True,
    ):
        """Add episode transitions, flagging frontier events.

        Frontier criteria (any transition in episode meets any):
          - episode_collision == True
          - min_clearance < dense_min_clearance_threshold
          - scenario == 'dense'
          - collision_frontier_boost: individual collision transitions also go to frontier
        """
        # Tag each transition with episode-level metadata
        for t in transitions:
            t.scenario = scenario
            t.episode_collision = episode_collision
            t.episode_success = episode_success

        # Determine if episode qualifies as frontier
        is_frontier_episode = (
            episode_collision
            or any(getattr(t, 'min_clearance', float('inf')) < dense_min_clearance_threshold for t in transitions)
            or scenario == 'dense'
        )

        # Add all to online
        for t in transitions:
            self.add_online(t)

        # Add frontier copies
        if is_frontier_episode:
            for t in transitions:
                self.add_frontier(t)
        elif collision_frontier_boost:
            # Individual collision transitions also go to frontier
            for t in transitions:
                if getattr(t, 'collision', False):
                    self.add_frontier(t)

    def clear_online(self):
        """Clear B_online_recent (e.g. before a new rollout phase)."""
        self.b_online_recent.clear()


def collate_to_critic(
    transitions: List[RouteATransition]
) -> Dict[str, torch.Tensor]:
    """Collate RouteATransitions into batch format for compute_critic_loss().

    Args:
        transitions: List of transitions

    Returns:
        state:         (B, 1536)
        action:        (B, 8, 11)
        action_tp1:    (B, 8, 11)  — A_{t+H} for bootstrap TD target
        reward_total:  (B,)
        reward_heads:  {name: (B,)}
        next_state:    (B, 1536)
        done:          (B,)
    """
    B = len(transitions)
    device = torch.device("cpu")

    state = torch.zeros(B, 1536, device=device)
    action = torch.zeros(B, 8, 11, device=device)
    action_tp1 = torch.zeros(B, 8, 11, device=device)
    reward_total = torch.zeros(B, device=device)
    reward_heads = {name: torch.zeros(B, device=device) for name in REWARD_TERM_NAMES}
    next_state = torch.zeros(B, 1536, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)

    for i, t in enumerate(transitions):
        state[i] = torch.from_numpy(t.obs_feature_t)
        action[i] = torch.from_numpy(t.action_chunk)
        action_tp1[i] = torch.from_numpy(t.action_chunk_tp1)
        reward_total[i] = t.reward_total
        for name in REWARD_TERM_NAMES:
            reward_heads[name][i] = t.reward_terms.get(name, 0.0)
        next_state[i] = torch.from_numpy(t.obs_feature_tp1)
        done[i] = bool(t.done)

    return {
        "state": state,
        "action": action,
        "action_tp1": action_tp1,
        "reward_total": reward_total,
        "reward_heads": reward_heads,
        "next_state": next_state,
        "done": done,
    }


def collate_to_critic_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Move collated batch to target device."""
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor)
            else {hk: hv.to(device) for hk, hv in v.items()})
        for k, v in batch.items()
    }


# ═══════════════════════════════════════════════════════════════════
# Sanity test
# ═══════════════════════════════════════════════════════════════════

def _test_replay_buffer():
    """Test replay buffer basic operations."""
    from termcolor import cprint

    print("=" * 60)
    print("RouteAReplayBuffer — sanity test")
    print("=" * 60)

    buffer = RouteAReplayBuffer(capacity_online=1000)

    # Add some transitions
    for i in range(50):
        t = RouteATransition(
            obs_feature_t=np.random.randn(1536).astype(np.float32),
            action_chunk=np.random.randn(8, 11).astype(np.float32),
            action_chunk_tp1=np.random.randn(8, 11).astype(np.float32),
            reward_total=float(np.random.randn()),
            reward_terms={name: float(np.random.randn()) for name in REWARD_TERM_NAMES},
            obs_feature_tp1=np.random.randn(1536).astype(np.float32),
            done=bool(i % 5 == 0),
            world_id=4 + (i % 5),
            episode_seed=i,
        )
        assert t.validate(), f"Transition {i} validation failed"
        buffer.add_online(t)

    stats = buffer.get_stats()
    print(f"  Buffer stats: {stats}")
    assert stats["n_online_recent"] == 50, f"Expected 50, got {stats['n_online_recent']}"

    # Sample a batch
    batch = buffer.sample(batch_size=16)
    B = batch["state"].shape[0]
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  state:       {batch['state'].shape}")
    print(f"  action:      {batch['action'].shape}")
    print(f"  action_tp1:  {batch['action_tp1'].shape}")
    print(f"  reward_total:{batch['reward_total'].shape}")
    print(f"  reward_heads:{len(batch['reward_heads'])} terms")
    print(f"  next_state:  {batch['next_state'].shape}")
    print(f"  done:        {batch['done'].shape}")

    assert B > 0, f"Empty batch (B={B})"
    assert batch["state"].shape == (B, 1536)
    assert batch["action"].shape == (B, 8, 11)
    assert batch["action_tp1"].shape == (B, 8, 11)
    assert batch["reward_total"].shape == (B,)
    assert batch["next_state"].shape == (B, 1536)
    assert batch["done"].shape == (B,)
    assert len(batch["reward_heads"]) == 8

    # Test empty sample
    empty = RouteAReplayBuffer()
    empty_batch = empty.sample(16)
    assert empty_batch["state"].shape == (0, 1536), "Empty batch should have 0 samples"
    print(f"  ✅ Empty batch returns 0 samples")

    print(f"\n  ✅ All replay buffer tests passed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    _test_replay_buffer()
