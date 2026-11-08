"""Reward normalization — single source contract (checklist G, 附录 H) +
R6-F running reward scaling (F-07 → R6-TUNE-1, Guide §11.7.3).

Reward semantics and any normalization state have exactly ONE defined
source — the env's Reward module (identified by
``runtime["reward_version"]``). The checkpoint's ``reward_norm_state``
is the only carrier of the state, and a loaded state is never
overwritten by a live one.

Two branches:

- ``none`` (default): the RL core adds no normalization; ``params``
  stays structured — ``{state_version, step_count}`` — never an empty
  dict (step_count advances so per-batch metrics and the cross-field
  resume invariant apply uniformly).
- ``official_running_return_std``: a small adapted port of the frozen
  vendor ``RunningRewardScaler`` (third_party/dppo_official_cc7234a/
  util/reward_scaling.py at vendor commit cc7234a, PPG-based). The
  vendor class is the BEHAVIOR ORACLE — parity is pinned test-side
  (test_scaler_math_parity_with_vendor_fixed_sequence).

§11.7.3 semantics: the env produces RAW rewards and artifact/episode
returns stay raw; the scaler touches ONLY the training reward consumed
by GAE/value targets (the single consumer inside ppo_update);
eval/BC-baseline/paired-capability metrics never update the training
scaler. The scaler's gamma must equal the contract's gamma_env
(single-source discount — a mismatch is a hard construction error).

First flags are derived from episode ids, NOT supplied by the caller:
``first[e, 0]`` compares against the per-slot last episode id carried
across batches (state field — the vendor receives the equivalent from
its leading ``done_venv``), ``first[e, t>0]`` compares consecutive ids
within the batch. The PRODUCTION entry is ``scale_flat`` (the flat
rollout buffer with per-transition slot ids — buffer order preserves
each slot's time order, so the vendor's slot-parallel recurrence over
(n_envs, T) is reproduced exactly); ``__call__`` is the 2D vendor-parity
oracle form and delegates to it.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict

import numpy as np

REWARD_SCALE_STATE_VERSION = "v1"

REWARD_SCALE_METHODS = ("none", "official_running_return_std")


@dataclass(frozen=True)
class RewardScaleConfig:
    """Reward scaling configuration — ALL fields enter the resolved
    config hash; none of them is checkpointed as mutable state."""

    method: str = "none"  # "none" | "official_running_return_std"
    gamma: float = 0.99
    epsilon: float = 1e-8  # variance floor in the transform
    cliprew: float = 10.0  # transform clip bound

    def __post_init__(self) -> None:
        assert self.method in REWARD_SCALE_METHODS, (
            f"reward scale config: unknown method {self.method!r}"
        )
        if self.method == "official_running_return_std":
            assert 0.0 < self.gamma < 1.0, (
                f"reward scale config: gamma {self.gamma} must be in (0, 1)"
            )
            assert self.epsilon > 0.0, (
                f"reward scale config: epsilon {self.epsilon} must be positive"
            )
            assert self.cliprew > 0.0, (
                f"reward scale config: cliprew {self.cliprew} must be positive"
            )


class _RunningMeanStd:
    """Vendor port (reward_scaling.py:13-39) — delta-based moments."""

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(())
        self.var = np.ones(())
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: float, batch_var: float, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = m2 / (tot_count - 1)
        self.count = tot_count


class RunningRewardScaler:
    """Per-env-slot running return scaler (vendor math). State
    (checkpointed per batch): carried per-slot return, per-slot last
    episode id (first-flag derivation), running mean/var/count over all
    slots, step_count (== ppo_update_count cross-field invariant)."""

    def __init__(self, cfg: RewardScaleConfig, num_envs: int, gamma_env: float) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self.step_count = 0
        if cfg.method == "none":
            return
        if cfg.gamma != gamma_env:
            raise ValueError(
                f"reward scale: gamma {cfg.gamma} must equal contract "
                f"gamma_env {gamma_env} (single-source discount)"
            )
        self.ret = np.zeros(num_envs, dtype=np.float64)
        self.last_episode_id = [None] * num_envs
        self.ret_rms = _RunningMeanStd(epsilon=1e-4)
        self.cliprew = cfg.cliprew
        self.epsilon = cfg.epsilon
        self.gamma = cfg.gamma

    def derive_first(self, episode_ids: np.ndarray) -> np.ndarray:
        """First flags from episode ids: t=0 compares against the per-slot
        last id carried across batches; t>0 compares consecutive ids."""
        n_envs, tlen = episode_ids.shape
        first = np.zeros((n_envs, tlen), dtype=np.int64)
        for e in range(n_envs):
            for t in range(tlen):
                if t == 0:
                    prev = self.last_episode_id[e]
                    first[e, t] = 1 if (prev is None
                                        or episode_ids[e, t] != prev) else 0
                else:
                    first[e, t] = (
                        1 if episode_ids[e, t] != episode_ids[e, t - 1] else 0
                    )
        return first

    def transform(self, reward: np.ndarray) -> np.ndarray:
        """Vendor transform (reward_scaling.py:68-73)."""
        return np.clip(
            reward / np.sqrt(self.ret_rms.var + self.epsilon),
            -self.cliprew,
            self.cliprew,
        )

    def scale_flat(
        self,
        rewards: np.ndarray,
        episode_ids: np.ndarray,
        env_ids: np.ndarray,
    ) -> np.ndarray:
        """Production entry: scale one FLAT training-reward buffer
        (per-transition slot ids). The buffer is a chronological
        interleaving of the per-slot time series, so buffer order
        preserves each slot's own time order — the exact recurrence the
        vendor runs slot-parallel over (n_envs, T). First flags derive
        from the per-slot last episode id (carried across batches — the
        vendor's leading ``done_venv``). ``none`` is the identity.
        Transactional: a raise here fails the update before publish, so
        no checkpoint ever contains a half-advanced scaler."""
        self.step_count += 1
        if self.cfg.method == "none":
            return rewards
        n = len(rewards)
        rets = np.zeros(n, dtype=np.float64)
        for t in range(n):
            e = int(env_ids[t])
            prev = self.last_episode_id[e]
            first_t = 1 if (prev is None or episode_ids[t] != prev) else 0
            self.ret[e] = (
                rewards[t] + (1 - first_t) * self.gamma * self.ret[e]
            )
            rets[t] = self.ret[e]
            self.last_episode_id[e] = episode_ids[t]
        self.ret_rms.update(rets)
        return self.transform(rewards)

    def __call__(self, reward: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
        """Vendor-parity 2D interface ((n_envs, T) per-slot arrays — the
        unit-test oracle form, reward_scaling.py:76-87). Delegates to the
        flat production recurrence: slot-major flattening preserves each
        slot's time order, so the per-slot ret sequences — and the
        aggregated running moments — are identical. ``derive_first``
        (above) documents the first-flag rule on the 2D form."""
        n_envs, tlen = reward.shape
        flat = self.scale_flat(
            reward.reshape(-1),
            episode_ids.reshape(-1),
            np.repeat(np.arange(n_envs), tlen),
        )
        return flat.reshape(n_envs, tlen)

    def state_dict(self) -> Dict[str, object]:
        """Method-conditional params payload (JSON-serializable; never an
        empty dict)."""
        if self.cfg.method == "none":
            return {
                "state_version": REWARD_SCALE_STATE_VERSION,
                "step_count": self.step_count,
            }
        return {
            "state_version": REWARD_SCALE_STATE_VERSION,
            "step_count": self.step_count,
            "ret": [float(x) for x in self.ret],
            "mean": float(self.ret_rms.mean),
            "var": float(self.ret_rms.var),
            "count": float(self.ret_rms.count),
            "last_episode_id": [
                (x.item() if isinstance(x, np.generic) else x)
                for x in self.last_episode_id
            ],
        }

    def load_state_dict(self, params: Dict[str, object]) -> None:
        """Structural validation THEN load (the resume gate runs
        validate_reward_scale_state with the counters FIRST)."""
        if params.get("state_version") != REWARD_SCALE_STATE_VERSION:
            raise ValueError(
                f"reward scale state: state_version mismatch "
                f"({params.get('state_version')!r})"
            )
        if self.cfg.method == "none":
            if set(params) != {"state_version", "step_count"}:
                raise ValueError(
                    f"reward scale state: none schema violation (keys "
                    f"{sorted(params)})"
                )
            self.step_count = params["step_count"]
            return
        if set(params) != {"state_version", "step_count", "ret", "mean",
                           "var", "count", "last_episode_id"}:
            raise ValueError(
                f"reward scale state: schema violation (keys "
                f"{sorted(params)})"
            )
        ret = params["ret"]
        last_ids = params["last_episode_id"]
        if not isinstance(params["step_count"], int) or params["step_count"] < 0:
            raise ValueError(
                f"reward scale state: bad step_count {params['step_count']!r}"
            )
        if len(ret) != self.num_envs:
            raise ValueError(
                f"reward scale state: ret length {len(ret)} != num_envs "
                f"{self.num_envs}"
            )
        if len(last_ids) != self.num_envs:
            raise ValueError(
                f"reward scale state: last_episode_id length {len(last_ids)} "
                f"!= num_envs {self.num_envs}"
            )
        for value in ret:
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"reward scale state: non-finite ret {value!r}")
        for value in (params["mean"], params["var"], params["count"]):
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"reward scale state: non-finite rms {value!r}")
        if params["var"] < 0.0:
            raise ValueError(f"reward scale state: negative var {params['var']}")
        if params["count"] < 0.0:
            raise ValueError(f"reward scale state: negative count {params['count']}")
        kinds = {type(x) for x in last_ids}
        if not kinds or not kinds.issubset((int, str)):
            raise ValueError(
                f"reward scale state: bad last_episode_id {last_ids!r}"
            )
        if len(kinds) > 1:
            raise ValueError(
                f"reward scale state: mixed-type last_episode_id {last_ids!r}"
            )
        self.step_count = params["step_count"]
        self.ret = np.asarray(ret, dtype=np.float64)
        self.ret_rms.mean = np.asarray(params["mean"], dtype=np.float64)
        self.ret_rms.var = np.asarray(params["var"], dtype=np.float64)
        self.ret_rms.count = float(params["count"])
        self.last_episode_id = list(last_ids)


@dataclass(frozen=True)
class RewardNormState:
    """Single-source reward-normalization state (checkpoint §4.16).

    ``version`` comes from the runtime (``runtime["reward_version"]``) —
    never guessed locally. ``source="env"`` pins the only allowed
    provider. ``method`` mirrors the RewardScaleConfig branch and
    ``params`` carries the scaler's method-conditional state (structured
    for BOTH branches — never an empty dict).
    """

    version: str  # reward version, e.g. "v1.3"
    source: str = "env"  # the env Reward module is the ONLY source
    method: str = "none"  # "none" | "official_running_return_std"
    params: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Checkpoint form (§4.16 ``reward_norm_state``)."""
        return asdict(self)


def validate_reward_scale_state(
    state: Dict[str, object],
    cfg: RewardScaleConfig,
    *,
    ppo_update_count: int,
) -> None:
    """Resume-side validation (before any tensor load): method/source/
    state-version/schema per branch + finiteness/bounds + cross-field
    step_count == ppo_update_count (the scaler advances once per
    committed PPO update, transactionally)."""
    if state.get("method") != cfg.method:
        raise ValueError(
            f"reward scale state: method mismatch "
            f"({state.get('method')!r} vs {cfg.method!r})"
        )
    if state.get("source") != "env":
        raise ValueError(
            f"reward scale state: source {state.get('source')!r} != 'env'"
        )
    params = state.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"reward scale state: params {params!r} not a dict")
    if params.get("state_version") != REWARD_SCALE_STATE_VERSION:
        raise ValueError(
            f"reward scale state: state_version mismatch "
            f"({params.get('state_version')!r})"
        )
    if params.get("step_count") != ppo_update_count:
        raise ValueError(
            f"reward scale state: step_count {params.get('step_count')} != "
            f"ppo_update_count {ppo_update_count}"
        )
    if cfg.method == "none":
        if set(params) != {"state_version", "step_count"}:
            raise ValueError(
                f"reward scale state: none schema violation (keys "
                f"{sorted(params)})"
            )
        return
    if set(params) != {"state_version", "step_count", "ret", "mean", "var",
                       "count", "last_episode_id"}:
        raise ValueError(
            f"reward scale state: schema violation (keys {sorted(params)})"
        )
    ret = params["ret"]
    last_ids = params["last_episode_id"]
    if len(ret) != len(last_ids):
        raise ValueError(
            f"reward scale state: ret length {len(ret)} != last_episode_id "
            f"length {len(last_ids)}"
        )
    for value in ret:
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"reward scale state: non-finite ret {value!r}")
    for value in (params["mean"], params["var"], params["count"]):
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"reward scale state: non-finite rms {value!r}")
    if params["var"] < 0.0:
        raise ValueError(f"reward scale state: negative var {params['var']}")
    if params["count"] < 0.0:
        raise ValueError(f"reward scale state: negative count {params['count']}")
    kinds = {type(x) for x in last_ids}
    if not kinds or not kinds.issubset((int, str)):
        raise ValueError(f"reward scale state: bad last_episode_id {last_ids!r}")
    if len(kinds) > 1:
        raise ValueError(
            f"reward scale state: mixed-type last_episode_id {last_ids!r}"
        )


def load_scaler_state(state: Dict[str, object], scaler: RunningRewardScaler) -> None:
    """Apply the checkpoint params to a fresh scaler (the resume gate
    runs validate_reward_scale_state with the counters FIRST; the
    per-object load re-validates structure)."""
    scaler.load_state_dict(state["params"])
