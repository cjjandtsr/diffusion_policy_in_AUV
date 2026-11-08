"""R6-F scheduler contract (F-07 → R6-TUNE-1, Guide §11.7.1-11.7.2).

Two branches:

- ``none`` (default): LR constant (the optimizer's configured value),
  state is STRUCTURED — ``{method, version, step_count}``, never an
  empty dict — and ``step_count`` advances so the per-batch metrics and
  the cross-field resume invariant apply uniformly.
- ``official_cosine_warmup``: a small adapted port of the frozen vendor
  ``CosineAnnealingWarmupRestarts`` (third_party/dppo_official_cc7234a/
  util/scheduler.py at vendor commit cc7234a, MIT, katsura-jp). The
  vendor class is the BEHAVIOR ORACLE — parity is pinned test-side
  (test_scheduler_lr_sequence_parity_with_vendor). The port keeps the
  vendor's no-arg ``step()`` semantics (the epoch-arg path is unused in
  the production call sites) and replaces the torch ``_LRScheduler``
  state coupling with an explicit, checkpointable state.

§11.7.2 semantics live at the call site (train_loop): critic steps once
per committed PPO update; actor steps only when it actually stepped
(critic-only warmup window); a failed/unpublished update commits NO
scheduler state (step happens after ppo_update returns successfully);
per-batch metrics record actor/critic LR before/after + step counts.

Config vs state: every LRSchedulerConfig field enters the resolved
config hash (config_hash); the checkpoint carries only mutable state
(step_count/step_in_cycle/cycle/cur_cycle_steps/max_lr) plus method and
a state-schema version. A resume under a different schedule is a
different run (identity gate), and any single-field state tamper fails
validation before any tensor load (cross-field: critic step_count ==
ppo_update_count; actor == max(0, ppo_update_count - warmup)).
"""

import math
from dataclasses import dataclass
from typing import Dict, List

SCHEDULER_STATE_VERSION = "v1"

SCHEDULER_METHODS = ("none", "official_cosine_warmup")


@dataclass(frozen=True)
class LRSchedulerConfig:
    """Scheduler configuration — ALL fields enter the resolved config
    hash; none of them is checkpointed as mutable state."""

    method: str = "none"  # "none" | "official_cosine_warmup"
    first_cycle_steps: int = 10
    warmup_steps: int = 0
    min_lr: float = 1e-4
    max_lr: float = 1e-3
    cycle_mult: float = 1.0
    gamma: float = 1.0

    def __post_init__(self) -> None:
        assert self.method in SCHEDULER_METHODS, (
            f"scheduler config: unknown method {self.method!r}"
        )
        if self.method == "official_cosine_warmup":
            # the vendor's own construction assert (util/scheduler.py:55)
            assert self.warmup_steps < self.first_cycle_steps, (
                "scheduler config: warmup_steps must be < first_cycle_steps"
            )
            assert self.max_lr > 0.0, "scheduler config: max_lr must be positive"
            assert self.min_lr <= self.max_lr, (
                "scheduler config: min_lr must not exceed max_lr"
            )
            assert self.cycle_mult >= 1.0, (
                "scheduler config: cycle_mult must be >= 1.0"
            )


class CosineAnnealingWarmupRestarts:
    """One scheduler instance for one optimizer (actor/critic separate —
    §11.7.2). ``none`` keeps LR constant; the official branch is the
    vendor port above. ``step_count`` counts committed steps on BOTH
    branches (per-batch metrics + cross-field resume invariant)."""

    def __init__(self, optimizer, cfg: LRSchedulerConfig) -> None:
        self.optimizer = optimizer
        self.cfg = cfg
        self.step_count = 0
        if cfg.method == "none":
            return
        self.first_cycle_steps = cfg.first_cycle_steps
        self.cycle_mult = cfg.cycle_mult
        self.base_max_lr = cfg.max_lr
        self.max_lr = cfg.max_lr
        self.min_lr = cfg.min_lr
        self.warmup_steps = cfg.warmup_steps
        self.gamma = cfg.gamma
        self.cur_cycle_steps = cfg.first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = -1
        self.last_epoch = -1
        # The vendor subclasses torch's _LRScheduler, whose constructor
        # performs ONE implicit step() (last_epoch=-1 -> epoch 0). The
        # port replicates it for exact sequence parity — WITHOUT counting
        # it as a committed step (step_count = committed ppo-update
        # steps, the cross-field invariant; the vendor's implicit step is
        # likewise not an iteration).
        self.last_epoch = 0
        self.step_in_cycle = 0
        # torch-equivalent base_lrs setup (the vendor's super().__init__
        # records the optimizer's initial LR before the implicit step).
        # The value is transient — init_lr() overwrites it below — but
        # get_lr() needs base_lrs defined at construction.
        self._init_lr()
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
        # vendor write order: init_lr() runs LAST (scheduler.py:69-78) —
        # the implicit step's get_lr() is overwritten, so construction LR
        # is ALWAYS min_lr. Gate B R6-TUNE-1 REWORK (2026-08-19): the old
        # order (_init_lr() only first) left get_lr()'s result in place,
        # which at warmup_steps=0 takes the cosine branch and yields
        # max_lr — a 100x divergence from the vendor AND a guaranteed
        # cold-resume optimizer-contract drift (save freezes min_lr as
        # the anchor).
        self._init_lr()

    def _init_lr(self) -> None:
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self) -> List[float]:
        """Vendor get_lr (three branches) — identical arithmetic."""
        if self.step_in_cycle == -1:
            return list(self.base_lrs)
        if self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        return [
            base_lr
            + (self.max_lr - base_lr)
            * (
                1
                + math.cos(
                    math.pi
                    * (self.step_in_cycle - self.warmup_steps)
                    / (self.cur_cycle_steps - self.warmup_steps)
                )
            )
            / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None) -> None:
        """Vendor no-arg path only (the epoch-arg path is unused by the
        production call sites and not ported)."""
        if self.cfg.method == "none":
            self.step_count += 1
            return
        assert epoch is None, "scheduler: explicit-epoch step is not ported"
        self.last_epoch = self.last_epoch + 1
        self.step_in_cycle = self.step_in_cycle + 1
        if self.step_in_cycle >= self.cur_cycle_steps:
            self.cycle += 1
            self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
            self.cur_cycle_steps = (
                int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                + self.warmup_steps
            )
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.step_count += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

    @property
    def lr(self) -> float:
        """Current applied LR (param group 0 — single group in practice)."""
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> Dict[str, object]:
        """Per-object state (method-conditional; never an empty dict)."""
        state = {
            "method": self.cfg.method,
            "version": SCHEDULER_STATE_VERSION,
            "step_count": self.step_count,
        }
        if self.cfg.method == "official_cosine_warmup":
            state.update(
                step_in_cycle=self.step_in_cycle,
                cycle=self.cycle,
                cur_cycle_steps=self.cur_cycle_steps,
                max_lr=self.max_lr,
            )
        return state

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Structural validation THEN load; re-applies the LR at the
        loaded step position (a restored scheduler must continue the
        sequence — the next update observes the loaded-step LR)."""
        if state.get("method") != self.cfg.method:
            raise ValueError(
                f"scheduler state: method mismatch "
                f"({state.get('method')!r} vs {self.cfg.method!r})"
            )
        if state.get("version") != SCHEDULER_STATE_VERSION:
            raise ValueError(
                f"scheduler state: version mismatch "
                f"({state.get('version')!r} vs {SCHEDULER_STATE_VERSION!r})"
            )
        if self.cfg.method == "none":
            if set(state) != {"method", "version", "step_count"}:
                raise ValueError(
                    f"scheduler state: none schema violation (keys "
                    f"{sorted(state)})"
                )
            self.step_count = state["step_count"]
            return
        step_count = state["step_count"]
        step_in_cycle = state["step_in_cycle"]
        cycle = state["cycle"]
        cur_cycle_steps = state["cur_cycle_steps"]
        max_lr = state["max_lr"]
        if not isinstance(step_count, int) or step_count < 0:
            raise ValueError(f"scheduler state: bad step_count {step_count!r}")
        if cycle < 0:
            raise ValueError(f"scheduler state: bad cycle {cycle!r}")
        if not (0 <= step_in_cycle < cur_cycle_steps):
            raise ValueError(
                f"scheduler state: step_in_cycle {step_in_cycle} out of "
                f"[0, {cur_cycle_steps})"
            )
        if cur_cycle_steps <= self.warmup_steps:
            raise ValueError(
                f"scheduler state: cur_cycle_steps {cur_cycle_steps} must "
                f"exceed warmup_steps {self.warmup_steps}"
            )
        expected_max = self.base_max_lr * (self.cfg.gamma ** cycle)
        if max_lr != expected_max:
            raise ValueError(
                f"scheduler state: max_lr {max_lr} != config-derived "
                f"{expected_max}"
            )
        self.step_count = step_count
        self.step_in_cycle = step_in_cycle
        self.cycle = cycle
        self.cur_cycle_steps = cur_cycle_steps
        self.max_lr = max_lr
        self.last_epoch = step_count - 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


def build_lr_schedulers(actor_optimizer, critic_optimizer, cfg: LRSchedulerConfig):
    """Independent actor/critic instances (§11.7.2 — never shared)."""
    return (
        CosineAnnealingWarmupRestarts(actor_optimizer, cfg),
        CosineAnnealingWarmupRestarts(critic_optimizer, cfg),
    )


def step_scheduler_pair(actor_sched, critic_sched, *, actor_stepped: bool) -> None:
    """§11.7.2: critic steps once per COMMITTED update; actor only when
    it actually stepped (critic-only warmup leaves the actor at
    step_count 0). Called by the train loop exactly once per successful
    ppo_update — a failed/unpublished update never reaches here."""
    critic_sched.step()
    if actor_stepped:
        actor_sched.step()


def _scheduler_axes(sched: CosineAnnealingWarmupRestarts) -> Dict[str, object]:
    """Checkpoint per-side axes (method-conditional: none keeps only
    step_count — official-only axes in a none state are a schema
    violation)."""
    if sched.cfg.method == "none":
        return {"step_count": sched.step_count}
    return {
        "step_count": sched.step_count,
        "step_in_cycle": sched.step_in_cycle,
        "cycle": sched.cycle,
        "cur_cycle_steps": sched.cur_cycle_steps,
        "max_lr": sched.max_lr,
    }


def scheduler_state_dict(actor_sched, critic_sched) -> Dict[str, object]:
    """Checkpoint field shape: {method, version, actor, critic} — the
    exact form the resume identity gate validates."""
    return {
        "method": actor_sched.cfg.method,
        "version": SCHEDULER_STATE_VERSION,
        "actor": _scheduler_axes(actor_sched),
        "critic": _scheduler_axes(critic_sched),
    }


def validate_scheduler_state(
    state: Dict[str, object],
    cfg: LRSchedulerConfig,
    *,
    ppo_update_count: int,
    n_critic_warmup_updates: int,
) -> None:
    """Resume-side validation (before any tensor load): method/version/
    schema per side + internal bounds + cross-field step counts.

    Cross-field invariants (§11.7.2): critic.step_count ==
    ppo_update_count (critic steps every committed update); actor
    .step_count == max(0, ppo_update_count - n_critic_warmup_updates)
    (actor steps only after the warmup window)."""
    if state.get("method") != cfg.method:
        raise ValueError(
            f"scheduler state: method mismatch "
            f"({state.get('method')!r} vs {cfg.method!r})"
        )
    if state.get("version") != SCHEDULER_STATE_VERSION:
        raise ValueError(
            f"scheduler state: version mismatch "
            f"({state.get('version')!r} vs {SCHEDULER_STATE_VERSION!r})"
        )
    expected_actor = max(0, ppo_update_count - n_critic_warmup_updates)
    for side, expected in (("actor", expected_actor), ("critic", ppo_update_count)):
        axes = state[side]
        if axes.get("step_count") != expected:
            raise ValueError(
                f"scheduler state: {side}.step_count {axes.get('step_count')} "
                f"!= cross-field {expected} (ppo_update_count="
                f"{ppo_update_count}, warmup={n_critic_warmup_updates})"
            )
        if cfg.method == "none":
            if set(axes) != {"step_count"}:
                raise ValueError(
                    f"scheduler state: none schema violation on {side} "
                    f"(keys {sorted(axes)})"
                )
            continue
        if set(axes) != {"step_count", "step_in_cycle", "cycle",
                         "cur_cycle_steps", "max_lr"}:
            raise ValueError(
                f"scheduler state: schema violation on {side} (keys "
                f"{sorted(axes)})"
            )
        step_in_cycle = axes["step_in_cycle"]
        cycle = axes["cycle"]
        cur_cycle_steps = axes["cur_cycle_steps"]
        if not (0 <= step_in_cycle < cur_cycle_steps):
            raise ValueError(
                f"scheduler state: {side}.step_in_cycle {step_in_cycle} out "
                f"of [0, {cur_cycle_steps})"
            )
        if cycle < 0:
            raise ValueError(f"scheduler state: {side}.cycle {cycle} < 0")
        if cur_cycle_steps <= cfg.warmup_steps:
            raise ValueError(
                f"scheduler state: {side}.cur_cycle_steps {cur_cycle_steps} "
                f"must exceed warmup_steps {cfg.warmup_steps}"
            )
        expected_max = cfg.max_lr * (cfg.gamma ** cycle)
        if axes["max_lr"] != expected_max:
            raise ValueError(
                f"scheduler state: {side}.max_lr {axes['max_lr']} != "
                f"config-derived {expected_max}"
            )


def load_scheduler_state(
    state: Dict[str, object], actor_sched, critic_sched
) -> None:
    """Apply the checkpoint state to fresh scheduler instances (the
    resume gate runs validate_scheduler_state with the counters FIRST;
    per-object load_state_dict re-validates structure). The per-side
    axes carry no method/version — injected from the top level."""
    for side, sched in (("actor", actor_sched), ("critic", critic_sched)):
        per_object = dict(
            state[side], method=state["method"], version=state["version"]
        )
        sched.load_state_dict(per_object)
