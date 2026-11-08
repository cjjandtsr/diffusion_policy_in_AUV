"""Episode task/role contract and start/goal admission (design §4.8-4.10).

- EpisodeTaskSpec: role is a first-class episode field; role is not 1:1 with
  world (§4.8 [JUDGMENT]).
- RoleSampler: target-ratio sampling with +/-2pp gate over 10k dry resets.
- AdmissionChecker: the 7 start/goal admission rules (§4.9); 50 failures
  raise AdmissionExhausted instead of falling back to fixed unverified
  coordinates ([JUDGMENT]).
- UAVVectorEnvAdapter: boundary that shape-blocks the (n_envs, 16, 11)
  full-trajectory mis-route (§4.10 [PROVEN]).
"""

import base64
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import pickle
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from diffusion_policy_3d.env.collision_geom import (
    compute_obstacle_clearance,
    load_obstacles,
)
from diffusion_policy_3d.env.kinematic_uav_env_v2 import KinematicUAVEnv
from diffusion_policy_3d.env.traj_utils import normalize_angle
from diffusion_policy_3d.model.common.synth_pointcloud import (
    CAM_MOUNT_X,
    CAM_MOUNT_Y,
    CAM_MOUNT_Z,
    DepthCameraIntrinsics,
    SynthPointCloud,
)
from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.planner import (
    astar_3d,
    build_occupancy_grid,
)
from diffusion_policy_3d.rl.dppo_uav.validated_synth_adapter import (
    ValidatedSynthAdapter,
    frame_hit_count,
)

ROLE_NAMES = (
    "anticipatory",
    "dense_recovery",
    "support_expansion",
    "yaw_lateral_recovery",
)
# R7-8 ruling 2026-08-24: the manifest carries three literal splits;
# "rl_eval" is the CONVENTIONAL name for rl_val ∪ rl_test, never a
# literal manifest split. Registry/spec construction must therefore
# accept the three literal splits (the old ("rl_train", "rl_eval")
# enum rejected every rl_val/rl_test manifest entry).
SPLIT_NAMES = ("rl_train", "rl_val", "rl_test")

# [ASSUMPTION] target ratios: three main-training roles split evenly,
# yaw_lateral_recovery auxiliary (§4.8: auxiliary robustness role, not a
# fourth capability). Tune from 10k-reset calibration if needed.
DEFAULT_ROLE_TARGETS: Dict[str, float] = {
    "anticipatory": 0.30,
    "dense_recovery": 0.30,
    "support_expansion": 0.30,
    "yaw_lateral_recovery": 0.10,
}

# Mirrors the frozen env _ROLE_SAMPLING_PARAMS (kinematic_uav_env_v2.py:324).
ROLE_ADMISSION_PARAMS: Dict[str, Dict[str, object]] = {
    "anticipatory": {"d_xy": (10, 24), "start_cl": 0.60, "goal_cl": 2.50},
    "dense_recovery": {"d_xy": (5, 18), "start_cl": 0.60, "goal_cl": 1.50},
    "support_expansion": {"d_xy": (8, 26), "start_cl": 0.80, "goal_cl": 2.00},
    "yaw_lateral_recovery": {"d_xy": (6, 18), "start_cl": 0.80, "goal_cl": 2.00},
}

D_3D_FLOOR = 4.0  # unified 3D distance floor (env behavior)
STRETCH_RANGE = (1.1, 1.8)  # dense_recovery path-stretch gate (env behavior)
MAX_ADMISSION_ATTEMPTS = 50


class AdmissionExhausted(Exception):
    """50 failed admission attempts for a task — sampler must switch world."""


@dataclass(frozen=True)
class EpisodeTaskSpec:
    """One episode task definition (design §4.8).

    ``manifest_entry_hash`` is the MANIFEST SOURCE hash: it covers
    the entry's fixed recorded fields only — it does NOT cover the runtime
    SAMPLED start/goal/yaw (that is ``episode_config_hash``, computed at
    reset and carried on EpisodeConfigRecord).
    """

    world_uid: str
    obstacle_json: str
    role: str
    split: str
    admission_id: str
    generator_version: str
    manifest_entry_hash: str

    def __post_init__(self) -> None:
        if self.role not in ROLE_NAMES:
            raise ValueError(f"unknown role {self.role!r}, expected one of {ROLE_NAMES}")
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"unknown split {self.split!r}, expected one of {SPLIT_NAMES}")


@dataclass(frozen=True)
class EpisodeConfigRecord:
    """One admitted runtime episode: identity + sampled geometry +
    admission metrics, hash-bound.

    ``manifest_entry_hash`` is the spec's source hash; ``episode_config_hash``
    is the canonical hash over {identity, episode_seed, start, goal, start_yaw}
    — the sampled config, so any resample under the same manifest entry
    yields a different hash. Admission metrics are recorded for auditing but
    are NOT part of the hash (they are deterministic from start/goal/world;
    path_stretch/path_length are 0.0 when not applicable / not computed).
    """

    episode_id: str
    world_uid: str
    role: str
    split: str
    admission_id: str
    episode_seed: int
    start: Tuple[float, float, float]
    goal: Tuple[float, float, float]
    start_yaw: float
    start_clearance: float
    goal_clearance: float
    reachability_pass: bool
    path_length: float
    path_stretch: float
    manifest_entry_hash: str
    episode_config_hash: str


@dataclass(frozen=True)
class AdmissionResult:
    pass_: bool
    reasons: Tuple[str, ...]
    # Admission metrics (computed in check(); None when a rule never
    # ran — the caller records 0.0 placeholders via build_episode_config_record)
    start_clearance: Optional[float] = None
    goal_clearance: Optional[float] = None
    reachability_pass: Optional[bool] = None
    path_length: Optional[float] = None
    path_stretch: Optional[float] = None


@dataclass
class StepBatch:
    """One batched env step (design §4.10)."""

    obs: dict
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: List[dict]

    def __iter__(self):
        """Duck-typed train_loop 5-tuple order: (obs, reward, terminated,
        truncated, info) — keeps the train_loop unpack unchanged while
        preserving the designed StepBatch return type."""
        yield self.obs
        yield self.reward
        yield self.terminated
        yield self.truncated
        yield self.info


class TaskSpecRegistry:
    """Builds EpisodeTaskSpec lists from manifest entries; keeps splits apart.

    Every entry MUST carry a non-empty ``manifest_entry_hash``: rule-6
    consistency is hash-proxy over the entry's FIXED recorded fields — the
    sampled start/goal/role are covered by episode_config_hash at reset. An
    empty hash would spuriously fail every rule-6 comparison against a
    populated manifest.
    """

    def __init__(self, entries: List[Dict[str, str]]) -> None:
        self._specs = [
            EpisodeTaskSpec(
                world_uid=e["world_uid"],
                # The default is the path the environment ACTUALLY loads —
                # <uid>.world.obstacles.json (frozen E040 build_envs /
                # build_routea_train_pool). The old default f"{uid}.json"
                # recorded a phantom file.
                obstacle_json=e.get(
                    "obstacle_json", f"{e['world_uid']}.world.obstacles.json"
                ),
                role=e["role"],
                split=e["split"],
                admission_id=e.get(
                    "admission_id", f"{e['world_uid']}__{e['role']}__{e['split']}"
                ),
                generator_version=e.get("generator_version", "routeA_rl_v1.3"),
                manifest_entry_hash=e.get("manifest_entry_hash", ""),
            )
            for e in entries
        ]
        for s in self._specs:
            if not s.manifest_entry_hash:
                raise ValueError(
                    f"registry entry {s.world_uid}/{s.role} has empty"
                    " manifest_entry_hash (ASSUMPTION-6)"
                )

    def train_specs(self) -> List[EpisodeTaskSpec]:
        return [s for s in self._specs if s.split == "rl_train"]

    def eval_specs(self) -> List[EpisodeTaskSpec]:
        # R7-8 ruling 2026-08-24: the eval side is the two literal
        # eval-only splits (rl_eval = rl_val ∪ rl_test, conventional).
        return [s for s in self._specs if s.split in ("rl_val", "rl_test")]


class RoleSampler:
    """Samples roles at target ratios; gates on +/-2pp over 10k resets (§4.8)."""

    TOLERANCE_PP = 0.02

    def __init__(
        self,
        targets: Dict[str, float],
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        assert abs(sum(targets.values()) - 1.0) < 1e-6, (
            f"role targets must sum to 1, got {sum(targets.values())}"
        )
        assert set(targets) == set(ROLE_NAMES), "targets must cover all roles"
        self.targets = targets
        self._rng = rng if rng is not None else np.random.RandomState()
        self._roles = list(ROLE_NAMES)
        self._probs = np.array([targets[r] for r in self._roles], dtype=np.float64)

    def sample(self) -> str:
        return self._roles[int(self._rng.choice(len(self._roles), p=self._probs))]

    @staticmethod
    def within_tolerance(actual: float, target: float) -> bool:
        return abs(actual - target) <= RoleSampler.TOLERANCE_PP

    @staticmethod
    def proportion_report(counts: Dict[str, int]) -> Dict[str, float]:
        """Actual role proportions from episode counts.

        The single implementation; the caller is train_loop (per-batch
        role statistics). Raises on empty counts — callers must guard.
        """
        total = sum(counts.values())
        assert total > 0, "empty counts"
        return {r: counts[r] / total for r in ROLE_NAMES}


@dataclass(frozen=True)
class SampledEpisodeTask:
    """One sampler decision: task + deterministic episode seed + ids.

    ``global_episode_index`` is the episode-start counter (advances on every
    attempt — each attempt begins a new episode, so ep_ids never repeat);
    ``global_attempt_index`` is the raw attempt counter. Both continue across
    batches and checkpoint/restore (SEED-RESUME).
    """

    task_spec: EpisodeTaskSpec
    episode_seed: int
    global_attempt_index: int
    global_episode_index: int


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Canonical SHA-256 over a JSON payload (the shared hash primitive).

    Key-order independent (sort_keys), compact separators, utf-8, and NaN
    hard-failing (allow_nan=False). NO ``default=str`` — a non-JSON field
    must raise instead of being silently swallowed.
    """
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_record_key(world_uid: str, role: str, admission_id: str) -> str:
    """Composite manifest-record key: two roles sharing a world_uid
    must not overwrite each other's record (rule 6). Fixed-order JSON-array
    form is collision-free — no separator can appear literally inside the
    encoding, and the order is NOT sorted (sorting would alias permutations
    of the three fields)."""
    return json.dumps(
        [world_uid, role, admission_id],
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _spec_set_hash(specs: Sequence[EpisodeTaskSpec]) -> str:
    """Canonical SHA-256 over the (sorted) task-spec set.

    Delegates to the shared canonical_sha256. The hash value changed
    once at the split (payload wrapped as {"specs": [...]}) — no test or
    persisted checkpoint asserts the old value, and the sampler only ever
    compares it against itself (state_dict -> load_state_dict).
    """
    payload = sorted(
        [
            {
                "world_uid": s.world_uid,
                "obstacle_json": s.obstacle_json,
                "role": s.role,
                "split": s.split,
                "admission_id": s.admission_id,
                "generator_version": s.generator_version,
                "manifest_entry_hash": s.manifest_entry_hash,
            }
            for s in specs
        ],
        key=lambda d: (d["world_uid"], d["role"], d["admission_id"]),
    )
    return canonical_sha256({"specs": payload})


class EpisodeTaskSampler:
    """The single stateful task sampler (SAMPLER-LIVE/SEED-RESUME).

    One instance is shared by the production train loop and the G3 gate
    audit: role via RoleSampler.sample(), in-bucket round-robin over the
    role's task specs with a recoverable per-role cursor, and episode seeds
    derived deterministically from ``(base_seed, global_attempt_index)`` —
    never restarted per batch. state_dict captures the full resume state
    (schema, base seed, RoleSampler RNG, cursors, attempt/episode/completed
    counters, task-spec set hash); load_state_dict hard-fails when the
    schema, base seed, or spec set does not match.

    Construction hard-fails on an empty role bucket (config error — no
    silent cross-role fallback) and on any eval-only spec (no training
    leakage).
    """

    STATE_SCHEMA = "episode_task_sampler/v1"

    def __init__(
        self,
        task_specs: Sequence[EpisodeTaskSpec],
        role_targets: Dict[str, float],
        base_seed: int,
    ) -> None:
        for s in task_specs:
            if s.split != "rl_train":
                raise ValueError(
                    f"sampler refuses {s.world_uid}/{s.role} split={s.split!r}"
                    " (eval-only specs must never enter training)"
                )
        buckets: Dict[str, List[EpisodeTaskSpec]] = {r: [] for r in ROLE_NAMES}
        for s in task_specs:
            buckets[s.role].append(s)
        for r in ROLE_NAMES:
            if not buckets[r]:
                raise ValueError(
                    f"no task specs for role {r!r} — sampler hard-fails rather"
                    " than fall back to another role"
                )
        self._buckets = {r: list(b) for r, b in buckets.items()}
        self._role_sampler = RoleSampler(
            targets=role_targets,
            rng=np.random.RandomState(int(base_seed)),
        )
        self._base_seed = int(base_seed)
        self._cursors: Dict[str, int] = {r: 0 for r in ROLE_NAMES}
        self._attempt = 0
        self._episode = 0
        self._completed = 0
        self._spec_set_hash = _spec_set_hash(list(task_specs))

    def next_attempt(self) -> SampledEpisodeTask:
        """Sample the next episode task; ids/seeds are strictly advancing."""
        role = self._role_sampler.sample()
        bucket = self._buckets[role]
        spec = bucket[self._cursors[role] % len(bucket)]
        self._cursors[role] += 1
        attempt = self._attempt
        episode = self._episode
        self._attempt += 1
        self._episode += 1
        return SampledEpisodeTask(
            task_spec=spec,
            episode_seed=self._base_seed + attempt,
            global_attempt_index=attempt,
            global_episode_index=episode,
        )

    def mark_completed(self) -> None:
        """Record one completed episode (voided/aborted episodes never call
        this — completed vs attempted proportions stay truthful)."""
        self._completed += 1

    @property
    def base_seed(self) -> int:
        """The episode seed sequence base — the resume identity axis the
        checkpoint's sampler_state is validated against (P0-2)."""
        return self._base_seed

    @property
    def spec_set_hash(self) -> str:
        """The task-spec set identity — the resume identity axis the
        checkpoint's sampler_state is validated against (P0-2)."""
        return self._spec_set_hash

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.STATE_SCHEMA,
            "base_seed": self._base_seed,
            # base64+pickle (mirror train_loop._rng_states): the raw
            # RandomState tuple embeds a numpy array, which
            # torch.load(weights_only=True) rejects — the sampler state is
            # persisted inside the checkpoint, so the payload must
            # be weights_only-safe.
            "role_sampler_rng": base64.b64encode(
                pickle.dumps(self._role_sampler._rng.get_state())
            ).decode("ascii"),
            "cursors": dict(self._cursors),
            "global_attempt_index": self._attempt,
            "global_episode_index": self._episode,
            "global_completed_index": self._completed,
            "task_spec_set_hash": self._spec_set_hash,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("schema") != self.STATE_SCHEMA:
            raise ValueError(
                f"sampler state schema mismatch: {state.get('schema')!r}"
                f" != {self.STATE_SCHEMA!r}"
            )
        if state.get("base_seed") != self._base_seed:
            raise ValueError(
                f"sampler state base_seed mismatch: {state.get('base_seed')!r}"
                f" != {self._base_seed!r}"
            )
        if state.get("task_spec_set_hash") != self._spec_set_hash:
            raise ValueError(
                "sampler state task-spec set mismatch (resume over a different"
                " manifest must not reuse cursors/seeds)"
            )
        self._role_sampler._rng.set_state(
            pickle.loads(
                base64.b64decode(state["role_sampler_rng"].encode("ascii"))
            )
        )
        self._cursors = dict(state["cursors"])
        self._attempt = int(state["global_attempt_index"])
        self._episode = int(state["global_episode_index"])
        self._completed = int(state["global_completed_index"])


class AdmissionChecker:
    """The 7 start/goal admission rules (§4.9).

    A* reachability is injected as ``planner(start, goal) -> (reachable,
    path_len)``. Planner source: port the pre-v2
    ``kinematic_uav_env.py:558 _astar_3d`` (occupancy-grid A*, 6-tuple
    return; the E040-frozen v2 env has no A* symbols) into
    ``dppo_uav/planner.py``, with backup and docs. Signature note: the
    pre-v2 method takes ``(occ_grid, sx, sy, sz, gx, gy, gz)``, so an
    adapter lambda is required at the wiring point.
    """

    def __init__(
        self,
        contract: RLSystemContract,
        role_params: Dict[str, Dict[str, object]],
        obstacles: Dict[str, object],
        planner: Callable,
        manifest_records: Optional[Dict[str, str]] = None,
    ) -> None:
        self.contract = contract
        self.role_params = role_params
        self.obstacles = obstacles
        self.planner = planner
        self.manifest_records = manifest_records or {}

    def check(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        role: str,
        spec: Optional[EpisodeTaskSpec] = None,
    ) -> AdmissionResult:
        rp = self.role_params.get(role, self.role_params["anticipatory"])
        d_xy_min, d_xy_max = rp["d_xy"]
        start_cl_req = float(rp["start_cl"])
        goal_cl_req = float(rp["goal_cl"])
        reasons: List[str] = []

        # 1-2. start/goal clearance
        start_cl, _, _ = compute_obstacle_clearance(
            float(start[0]), float(start[1]), float(start[2]), self.obstacles
        )
        if start_cl < start_cl_req:
            reasons.append(f"start_clearance {start_cl:.2f} < {start_cl_req}")
        goal_cl, _, _ = compute_obstacle_clearance(
            float(goal[0]), float(goal[1]), float(goal[2]), self.obstacles
        )
        if goal_cl < goal_cl_req:
            reasons.append(f"goal_clearance {goal_cl:.2f} < {goal_cl_req}")

        # 3. 3D distance in role range
        d_xy = math_hypot(goal[0] - start[0], goal[1] - start[1])
        d_3d = math_hypot(d_xy, goal[2] - start[2])
        if d_xy < d_xy_min or d_xy > d_xy_max:
            reasons.append(f"d_xy {d_xy:.2f} outside role range ({d_xy_min}, {d_xy_max})")
        if d_3d < D_3D_FLOOR:
            reasons.append(f"d_3d {d_3d:.2f} < floor {D_3D_FLOOR}")

        # 4. A* / geometric planner reachability
        reachable, path_len = self.planner(
            np.asarray(start), np.asarray(goal)
        )
        if not reachable:
            reasons.append("planner unreachable")

        # 5. path stretch (dense_recovery only, env semantics)
        stretch = None
        if role == "dense_recovery" and reachable:
            stretch = path_len / max(d_3d, 0.1)
            if stretch < STRETCH_RANGE[0] or stretch > STRETCH_RANGE[1]:
                reasons.append(f"path_stretch {stretch:.2f} outside {STRETCH_RANGE}")

        # 6. manifest source consistency: rule 6 validates the spec's SOURCE
        # hash against the composite-key manifest record — two roles sharing
        # a world_uid never overwrite each other. The sampled start/goal/yaw
        # are covered by episode_config_hash (built at reset), not by this
        # hash.
        if spec is not None and self.manifest_records:
            expected = self.manifest_records.get(
                manifest_record_key(spec.world_uid, spec.role, spec.admission_id)
            )
            if expected is None or expected != spec.manifest_entry_hash:
                reasons.append(
                    f"manifest mismatch for {spec.world_uid}/{spec.role}:"
                    f" {spec.manifest_entry_hash!r} vs {expected!r}"
                )

        return AdmissionResult(
            pass_=not reasons,
            reasons=tuple(reasons),
            start_clearance=float(start_cl),
            goal_clearance=float(goal_cl),
            reachability_pass=bool(reachable),
            path_length=float(path_len),
            path_stretch=float(stretch) if stretch is not None else 0.0,
        )

    def raise_if_exhausted(self, attempts: int) -> None:
        """Rule 7: no fixed-coordinate fallback — fail structurally instead."""
        if attempts >= MAX_ADMISSION_ATTEMPTS:
            raise AdmissionExhausted(
                f"admission failed after {attempts} attempts;"
                " sampler must switch world/task"
            )


def build_episode_config_record(
    spec: EpisodeTaskSpec,
    seed: Optional[int],
    start: np.ndarray,
    goal: np.ndarray,
    start_yaw: float,
    start_clearance: float = 0.0,
    goal_clearance: float = 0.0,
    reachability_pass: bool = True,
    path_length: float = 0.0,
    path_stretch: float = 0.0,
) -> EpisodeConfigRecord:
    """Bind one admitted runtime episode to both hashes.

    Shared by the real adapter (metrics from the admission check) and the
    mock env (planner-less defaults) so both produce the SAME
    episode_config_hash for identical geometry — the hash covers
    {identity, episode_seed, start, goal, start_yaw} only; admission
    metrics are recorded but never hashed.
    """
    episode_seed = int(seed) if seed is not None else 0
    payload = {
        "world_uid": spec.world_uid,
        "role": spec.role,
        "split": spec.split,
        "admission_id": spec.admission_id,
        "episode_seed": episode_seed,
        "start": [float(v) for v in start],
        "goal": [float(v) for v in goal],
        "start_yaw": float(start_yaw),
    }
    return EpisodeConfigRecord(
        episode_id=f"{spec.world_uid}:{spec.role}:{episode_seed:09d}",
        world_uid=spec.world_uid,
        role=spec.role,
        split=spec.split,
        admission_id=spec.admission_id,
        episode_seed=episode_seed,
        start=tuple(float(v) for v in start),
        goal=tuple(float(v) for v in goal),
        start_yaw=float(start_yaw),
        start_clearance=float(start_clearance),
        goal_clearance=float(goal_clearance),
        reachability_pass=bool(reachability_pass),
        path_length=float(path_length),
        path_stretch=float(path_stretch),
        manifest_entry_hash=spec.manifest_entry_hash,
        episode_config_hash=canonical_sha256(payload),
    )


def math_hypot(dx: float, dy: float) -> float:
    return float(np.hypot(dx, dy))


class UAVVectorEnvAdapter:
    """Environment boundary: accepts only (n_envs, exec_steps, 11) actions.

    The (n_envs, full_horizon, 11) full-trajectory mis-route is blocked at the
    type/shape level (§4.10 [PROVEN]). Wires the duck-typed train_loop
    contract (MockUAVEnv: reset(task_specs, seeds) -> (batched obs, info),
    step((1, A_e, 11)) -> 5-tuple, cam_far, current_episode_config) to the
    frozen KinematicUAVEnv v2 (episode_config reset path, 4-step execution).
    The v2 env hardcodes n_obs_steps=8 (v2:128), so the adapter batch-prefixes
    the returned window directly — no history reassembly needed.

    Real synchronous vector semantics over n_envs slots. Each slot owns
    an independent KinematicUAVEnv + episode record/spec/done flag (static
    obstacle geometry — synth cache, checker cache — is read-only shared).
      - reset(task_specs, seeds) requires len == len == n_envs (full reset);
      - reset_at(env_ids, task_specs, seeds) resets ONLY the named slots and
        returns results in the env_ids INPUT order;
      - step(actions) always returns results in env_id 0..n_envs-1 order;
      - a slot that terminated/truncated must be reset_at() before it can be
        stepped again — stepping it raises RuntimeError (no auto-reset that
        hides the boundary).
    """

    def __init__(
        self,
        contract: RLSystemContract,
        world_pool: List[Tuple[str, str]],
        *,
        device: str = "cpu",
        max_episode_steps: int = 60,
        curriculum_stage: str = "C2",
        reward_version: str = "v16",
        disturb_enable: bool = True,

        z_min: float = 1.0,
        z_max: float = 5.0,
        manifest_records: Optional[Dict[str, str]] = None,
        n_envs: int = 1,
    ) -> None:
        """Mirror the E040 env construction (frozen v2 __init__) + eval
        discipline knobs. ``world_pool`` is [(world_uid, obstacles_json), ...]
        as returned by the frozen scan_world_pool. ``n_envs`` slots each own
        a KinematicUAVEnv; identical static geometry is read-only
        shared through the synth/checker caches."""
        self.contract = contract
        self._world_pool = list(world_pool)
        self._device = device
        self._manifest_records = manifest_records or {}
        self._cam_far = float(DepthCameraIntrinsics().far)
        self._synth_cache: Dict[str, SynthPointCloud] = {}
        self._validated_cache: Dict[str, ValidatedSynthAdapter] = {}
        self._checker_cache: Dict[str, AdmissionChecker] = {}
        self.n_envs = int(n_envs)
        assert self.n_envs >= 1, "n_envs must be >= 1"
        kwargs = dict(
            world_pool=list(world_pool),
            synth_pc_factory=self._cached_validated,
            max_episode_steps=max_episode_steps,
            z_min=z_min,
            z_max=z_max,
            disturb_enable=disturb_enable,
            reward_version=reward_version,
            curriculum_stage=curriculum_stage,
        )
        self._envs: List[KinematicUAVEnv] = [
            KinematicUAVEnv(**kwargs) for _ in range(self.n_envs)
        ]
        # per-slot episode state: done=True until the slot is reset
        self._records: List[Optional[EpisodeConfigRecord]] = (
            [None] * self.n_envs
        )
        self._specs: List[Optional[EpisodeTaskSpec]] = [None] * self.n_envs
        # slot lifecycle: INITIAL (done=True,
        # never reset) / ACTIVE (done=False) / DONE (done=True, episode
        # ended normally) / VOIDED (done=True AND voided — episode closed
        # on an observation fault via mark_voided). reset_at accepts only
        # INITIAL/DONE/VOIDED; stepping accepts only ACTIVE.
        self._done: List[bool] = [True] * self.n_envs
        self._voided: List[bool] = [False] * self.n_envs
        # Dense_recovery admission attempts that EXHAUSTED on
        # rule 5 (path stretch) — exposed to the G3 audit as a real gate
        # counter (per-attempt rejections that re-sample are NOT counted).
        self._dense_stretch_fail = 0
        # R7-5: per-slot runtime episode hash, captured at reset_at Phase 2
        # from the env's ACTUAL post-reset state ("" until first reset) —
        # the second independent hash next to the source manifest hash.
        self._runtime_episode_hashes: List[str] = [""] * self.n_envs

    @property
    def _env(self) -> KinematicUAVEnv:
        """Slot-0 env (legacy single-env accessor kept for eval/audit paths
        and tests that introspect the inner env)."""
        return self._envs[0]

    @property
    def cam_far(self) -> float:
        """Far-plane placeholder distance (synth D435i clip far)."""
        return self._cam_far

    @property
    def _collision_detected(self) -> bool:
        """Forward slot-0's collision flag (the paired-eval path always runs
        the single-slot configuration; the training loop reads this only via
        _eval_episode)."""
        return bool(getattr(self._envs[0], "_collision_detected", False))

    # ── synth / admission plumbing ───────────────────────────────

    def _cached_synth(self, json_path: str) -> SynthPointCloud:
        """synth_pc_factory for the frozen env: one SynthPointCloud per world."""
        if json_path not in self._synth_cache:
            self._synth_cache[json_path] = SynthPointCloud(
                json_path, device=self._device
            )
        return self._synth_cache[json_path]

    def _cached_validated(self, json_path: str) -> ValidatedSynthAdapter:
        """Inject the VALIDATED wrapper instead of the raw synth — the
        frozen env is untouched, the wrapper replaces the degenerate no-ray
        fallback with a physical far-endpoint encoding. One wrapper per
        world path; slots are processed sequentially, so last_report() is
        read before another slot re-enters the same world's wrapper."""
        if json_path not in self._validated_cache:
            self._validated_cache[json_path] = ValidatedSynthAdapter(
                self._cached_synth(json_path)
            )
        return self._validated_cache[json_path]

    def _pool_path(self, world_uid: str) -> str:
        for uid, json_path in self._world_pool:
            if uid == world_uid:
                return json_path
        raise ValueError(f"world {world_uid!r} not in pool")

    def _checker_for(self, world_uid: str) -> AdmissionChecker:
        """Per-world AdmissionChecker with the adapter's own occupancy grid
        (curriculum ranges from the frozen env; A* ported in planner.py)."""
        if world_uid not in self._checker_cache:
            json_path = self._pool_path(world_uid)
            obstacles = load_obstacles(json_path)
            x_min, x_max, y_min, y_max = self._env._curriculum_xy_range()
            z_min, z_max = self._env._curriculum_z_range()
            grid, meta = build_occupancy_grid(
                obstacles,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_min=z_min,
                z_max=z_max,
            )

            def _plan(start: np.ndarray, goal: np.ndarray) -> Tuple[bool, float]:
                result = astar_3d(grid, meta, *start, *goal)
                return result[0], result[1]

            self._checker_cache[world_uid] = AdmissionChecker(
                self.contract,
                ROLE_ADMISSION_PARAMS,
                obstacles,
                _plan,
                manifest_records=self._manifest_records,
            )
        return self._checker_cache[world_uid]

    def _sample_admitted(
        self, spec: EpisodeTaskSpec, seed: Optional[int]
    ) -> Tuple[np.ndarray, np.ndarray, float, AdmissionResult]:
        """Sample start/goal against the 7 admission rules, mirroring the
        frozen env's own ranges; 50 failures raise AdmissionExhausted (rule
        7 — no fixed-coordinate fallback). Returns the admitting check's
        metrics too (they land on the EpisodeConfigRecord)."""
        rng = np.random.RandomState(int(seed) if seed is not None else 0)
        x_min, x_max, y_min, y_max = self._envs[0]._curriculum_xy_range()
        z_low, z_high = self._envs[0]._curriculum_z_range()
        checker = self._checker_for(spec.world_uid)
        last_result: Optional[AdmissionResult] = None
        for _ in range(MAX_ADMISSION_ATTEMPTS):
            start = np.array(
                [
                    rng.uniform(x_min, x_max),
                    rng.uniform(y_min, y_max),
                    rng.uniform(z_low + 0.3, z_high - 0.3),
                ],
                dtype=np.float32,
            )
            goal = np.array(
                [
                    rng.uniform(x_min, x_max),
                    rng.uniform(y_min, y_max),
                    rng.uniform(z_low + 0.5, z_high - 0.5),
                ],
                dtype=np.float32,
            )
            result = checker.check(start, goal, spec.role, spec=spec)
            last_result = result
            if result.pass_:
                yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
                yaw += rng.uniform(-0.3, 0.3)
                return start, goal, normalize_angle(yaw), result
        # Count dense_recovery attempts that EXHAUSTED on rule-5
        # path stretch (the 50-attempt window used up). Per-attempt stretch
        # rejections are the admission mechanism's normal re-sampling
        # behavior — only exhaustion is a structural failure the G3 audit
        # must gate (the reason string is only ever emitted for
        # dense_recovery, rule 5).
        if spec.role == "dense_recovery" and last_result is not None and any(
            r.startswith("path_stretch") for r in last_result.reasons
        ):
            self._dense_stretch_fail += 1
        checker.raise_if_exhausted(MAX_ADMISSION_ATTEMPTS)

    def stretch_fail_count(self) -> int:
        """Cumulative dense_recovery path-stretch admission rejections
        (rule 5); the G3 audit gates this counter."""
        return self._dense_stretch_fail

    # ── vector-env contract (train_loop / MockUAVEnv duck type) ──

    def reset(
        self,
        task_specs: List[EpisodeTaskSpec],
        seeds: List[int],
    ) -> Tuple[dict, List[dict]]:
        """Full reset of ALL slots: len(task_specs) == len(seeds) ==
        n_envs. Returns batched obs (n_envs, ...) and one info entry per
        slot, in env_id order."""
        assert len(task_specs) == self.n_envs, (
            f"reset needs {self.n_envs} task_specs, got {len(task_specs)}"
        )
        assert len(seeds) == self.n_envs, (
            f"reset needs {self.n_envs} seeds, got {len(seeds)}"
        )
        return self.reset_at(list(range(self.n_envs)), task_specs, seeds)

    def reset_at(
        self,
        env_ids: List[int],
        task_specs: List[EpisodeTaskSpec],
        seeds: List[int],
    ) -> Tuple[dict, List[dict]]:
        """Reset ONLY the named slots; results come back in the
        env_ids INPUT order — the caller maps result row k to env_ids[k].

        Per slot: admit start/goal against the 7 rules, build the
        EpisodeConfigRecord (both hashes), reset that slot's KinematicUAVEnv
        with the sampled episode_config, and clear the done flag. The frozen
        env never resets ``_collision_detected`` — a stale True from a
        previous episode would poison the next episode's SR.

        reset_at accepts ONLY INITIAL / DONE / VOIDED slots. An ACTIVE slot
        (in-flight episode) hard-fails BEFORE any mutation — a reset here
        would silently discard the episode and its already-collected
        transitions.

        TWO-PHASE atomic reset. Phase 1 admits and prepares EVERY named slot
        (sample start/goal against the 7 rules, build the record + episode
        config) WITHOUT touching any slot state — an AdmissionExhausted on
        ANY slot raises before the first env mutation, so a mid-batch
        admission failure leaves ZERO partial resets. Phase 2 applies the
        prepared episodes (record/spec slots, env.reset, done flags) in
        env_ids order.
        """
        ids = [int(i) for i in env_ids]
        assert len(ids) == len(task_specs) == len(seeds), (
            "reset_at needs len(env_ids) == len(task_specs) == len(seeds)"
        )
        assert len(set(ids)) == len(ids), f"duplicate env_ids in reset_at: {ids}"
        # atomic preflight: reject any ACTIVE slot before touching a single
        # slot (the check must happen before the FIRST mutation, not per
        # slot mid-loop — a partial reset is worse than none).
        active = [i for i in ids if not self._done[i]]
        if active:
            raise RuntimeError(
                f"env slots {active} are ACTIVE — reset_at() only resets"
                " done/voided slots (guide R2.4); an in-flight episode must"
                " never be silently replaced"
            )
        # ── Phase 1 (read-only): admission + preparation for ALL slots.
        #    _sample_admitted raises AdmissionExhausted here — before any
        #    record/spec/env mutation (zero partial reset by construction).
        prepared: List[Tuple[int, EpisodeTaskSpec, int,
                             EpisodeConfigRecord, dict]] = []
        for env_id, spec, seed in zip(ids, task_specs, seeds):
            seed = int(seed)
            start, goal, yaw, admission = self._sample_admitted(spec, seed)
            record = build_episode_config_record(
                spec, seed, start, goal, yaw,
                start_clearance=admission.start_clearance,
                goal_clearance=admission.goal_clearance,
                reachability_pass=admission.reachability_pass,
                path_length=admission.path_length,
                path_stretch=admission.path_stretch,
            )
            cfg = {
                "world_name": spec.world_uid,
                "goal_pos": [float(v) for v in goal],
                "start_pos": [float(v) for v in start],
                "start_yaw": float(yaw),
                "disturb_events": [],
                # The unique per-episode id — admission_id
                # is shared across resets of the same spec, not an episode id.
                "episode_id": record.episode_id,
            }
            prepared.append((env_id, spec, seed, record, cfg))
        # ── Phase 2 (mutation): apply the prepared episodes.
        obs_list: List[dict] = []
        info_list: List[dict] = []
        for env_id, spec, seed, record, cfg in prepared:
            self._records[env_id] = record
            env = self._envs[env_id]
            obs, env_info = env.reset(seed=seed, options={"episode_config": cfg})
            env._collision_detected = False
            self._specs[env_id] = spec
            # R7-5: capture the runtime hash from the env's ACTUAL
            # post-reset state — never recomputed at step time (post-step
            # state legitimately differs and would break the equality).
            self._runtime_episode_hashes[env_id] = (
                self._runtime_episode_hash_of(env_id, spec, seed)
            )
            self._done[env_id] = False
            self._voided[env_id] = False
            obs_list.append(obs)
            info_list.append(self._wrap_info(env_id, obs, env_info))
        return self._stack(obs_list), info_list

    def mark_voided(self, env_ids: List[int]) -> None:
        """Explicit ACTIVE -> VOIDED transition:
        the loop calls this when a gate SAFE_END closes an episode on an
        observation fault. A VOIDED slot cannot step (same boundary as done)
        but reset_at() may re-episode it — the void boundary is explicit,
        never silent."""
        ids = [int(i) for i in env_ids]
        not_active = [i for i in ids if self._done[i]]
        if not_active:
            raise RuntimeError(
                f"env slots {not_active} are not ACTIVE — mark_voided()"
                " closes observation-faulted episodes only"
            )
        for env_id in ids:
            self._voided[env_id] = True
            self._done[env_id] = True

    def abort_episode(self, env_ids: List[int]) -> None:
        """Explicit ACTIVE -> DONE discard: the
        gate-audit loop resets the SAME slot for every configuration probe
        and never steps, so with reset_at() refusing ACTIVE slots it must
        close the previous probe episode explicitly. DONE but NOT VOIDED —
        the caller accepts the loss without a fault verdict. An aborted
        slot is a legal reset_at() target and stays unsteppable (the
        boundary is never hidden). Calling on a non-ACTIVE slot raises —
        same atomicity as reset_at/mark_voided."""
        ids = [int(i) for i in env_ids]
        not_active = [i for i in ids if self._done[i]]
        if not_active:
            raise RuntimeError(
                f"env slots {not_active} are not ACTIVE — abort_episode()"
                " discards in-flight episodes only"
            )
        for env_id in ids:
            self._done[env_id] = True

    def reject_episode(self, env_ids: List[int]) -> None:
        """Explicit ACTIVE -> DONE: the loop
        closes a slot whose RESET frame failed the point-cloud gate
        (REJECT_EPISODE) — the reset was consumed but the episode never
        ran, and the slot must be reset-able for the next sampler task.
        DONE but NOT VOIDED (no fault verdict — the same boundary as
        abort_episode, with the reset-reject semantics spelled out).
        Calling on a non-ACTIVE slot raises — same atomicity as
        reset_at/abort_episode/mark_voided."""
        ids = [int(i) for i in env_ids]
        not_active = [i for i in ids if self._done[i]]
        if not_active:
            raise RuntimeError(
                f"env slots {not_active} are not ACTIVE — reject_episode()"
                " closes reset-rejected episodes only"
            )
        for env_id in ids:
            self._done[env_id] = True

    def mark_truncated(self, env_ids: List[int]) -> None:
        """Explicit ACTIVE -> DONE: the loop
        closes a slot when ITS OWN max_ep_len truncates the episode — the
        env never signaled terminated/truncated, so without this explicit
        close the slot would stay ACTIVE and the next episode's reset_at
        would hard-fail. DONE but NOT VOIDED (loop-level truncation is not
        an observation fault). Calling on a non-ACTIVE slot raises — same
        atomicity as reset_at/abort_episode/mark_voided."""
        ids = [int(i) for i in env_ids]
        not_active = [i for i in ids if self._done[i]]
        if not_active:
            raise RuntimeError(
                f"env slots {not_active} are not ACTIVE — mark_truncated()"
                " closes loop-truncated episodes only"
            )
        for env_id in ids:
            self._done[env_id] = True

    def slot_state(self, env_id: int) -> str:
        """Read-only slot lifecycle query: INITIAL (never
        reset) / ACTIVE (in-flight) / DONE (ended, reset legal) / VOIDED
        (closed on an observation fault, reset legal)."""
        env_id = int(env_id)
        if not self._done[env_id]:
            return "ACTIVE"
        if self._voided[env_id]:
            return "VOIDED"
        if self._records[env_id] is None:
            return "INITIAL"
        return "DONE"

    def is_active(self, env_id: int) -> bool:
        """Read-only ACTIVE query — the gate audit uses it to
        skip the abort-close when no slot is ACTIVE instead of swallowing
        RuntimeError."""
        return not self._done[int(env_id)]

    def step(self, executed_actions: Any) -> StepBatch:
        """Step ALL slots synchronously with physical actions.

        Actions strictly (n_envs, exec_steps, 11); results always come
        back in env_id 0..n_envs-1 order. A slot that terminated/truncated
        must be reset_at() before it can be stepped again — stepping it
        raises (no auto-reset hiding the boundary). Delegates to step_at
        over every slot.

        G5 (design §5.7): the loop hands over whatever the E040 normalizer
        returns — a CUDA torch tensor — so the boundary converts tensor ->
        cpu numpy (the env physics is CPU-numpy) before shape checks.
        """
        if torch.is_tensor(executed_actions):
            executed_actions = executed_actions.detach().cpu().numpy()
        executed_actions = np.asarray(executed_actions)
        assert executed_actions.shape[0] == self.n_envs, (
            f"actions must be ({self.n_envs}, exec_steps, 11),"
            f" got {executed_actions.shape[0]} envs"
        )
        return self.step_at(list(range(self.n_envs)), executed_actions)

    def step_at(self, env_ids: List[int], executed_actions: Any) -> StepBatch:
        """Step ONLY the named ACTIVE slots — the
        loop's final partial batch (< n_envs active slots) never touches the
        others. Results come back in the env_ids INPUT order. Same atomic
        preflight as step: a done/voided slot raises BEFORE any slot steps
        (no auto-reset hiding the boundary)."""
        ids = [int(i) for i in env_ids]
        assert len(set(ids)) == len(ids), f"duplicate env_ids in step_at: {ids}"
        if torch.is_tensor(executed_actions):
            executed_actions = executed_actions.detach().cpu().numpy()
        executed_actions = np.asarray(executed_actions)
        n_envs, n_steps, n_dim = executed_actions.shape
        assert n_envs == len(ids), (
            f"actions rows must match env_ids ({len(ids)}), got {n_envs}"
        )
        assert n_steps == self.contract.exec_steps, (
            f"executed_actions must be (n_envs, exec_steps={self.contract.exec_steps},"
            f" 11), got steps={n_steps}"
        )
        assert n_dim == self.contract.action_dim, (
            f"action dim must be {self.contract.action_dim}, got {n_dim}"
        )
        obs_list: List[dict] = []
        rewards: List[float] = []
        terminated_list: List[bool] = []
        truncated_list: List[bool] = []
        info_list: List[dict] = []
        # Atomic preflight — reject when ANY named slot is
        # done/voided BEFORE stepping a single slot (the old per-slot
        # in-loop check advanced slots 0..k before discovering a done slot
        # at k+1).
        done_slots = [i for i in ids if self._done[i]]
        if done_slots:
            raise RuntimeError(
                f"env slots {done_slots} are done —"
                " reset_at() them before stepping"
            )
        for k, env_id in enumerate(ids):
            obs, reward, terminated, truncated, env_info = self._envs[env_id].step(
                np.asarray(executed_actions[k], dtype=np.float32)
            )
            self._done[env_id] = bool(terminated or truncated)
            obs_list.append(obs)
            rewards.append(float(reward))
            terminated_list.append(bool(terminated))
            truncated_list.append(bool(truncated))
            info_list.append(self._wrap_info(env_id, obs, env_info))
        return StepBatch(
            obs=self._stack(obs_list),
            reward=np.asarray(rewards, dtype=np.float32),
            terminated=np.asarray(terminated_list, dtype=bool),
            truncated=np.asarray(truncated_list, dtype=bool),
            info=info_list,
        )

    def current_episode_config(self, env_id: int = 0) -> dict:
        """MockUAVEnv contract keys + the full EpisodeConfigRecord (both
        hashes reach checkpoint and the eval artifact through this dict).
        Per-slot: the checkpoint binds slot 0 unless another env_id is
        requested. Raises if that slot was never admitted."""
        r = self._records[env_id]
        if r is None:
            raise RuntimeError(
                f"current_episode_config before any reset (slot {env_id})"
            )
        return {
            "world": r.world_uid,
            "role": r.role,
            "seed": r.episode_seed,
            "start": list(r.start),
            "goal": list(r.goal),
            "episode_id": r.episode_id,
            "split": r.split,
            "admission_id": r.admission_id,
            "start_yaw": r.start_yaw,
            "start_clearance": r.start_clearance,
            "goal_clearance": r.goal_clearance,
            "reachability_pass": r.reachability_pass,
            "path_length": r.path_length,
            "path_stretch": r.path_stretch,
            "manifest_entry_hash": r.manifest_entry_hash,
            "episode_config_hash": r.episode_config_hash,
        }

    def slot_obs(self, env_id: int) -> dict:
        """Env-level obs window of one slot (the restore path rebuilds
        the selected obs from it)."""
        return self._envs[int(env_id)]._get_obs_dict()

    def _runtime_episode_hash_of(
        self, env_id: int, spec: EpisodeTaskSpec, seed: int
    ) -> str:
        """R7-5: independent runtime episode hash over the env's ACTUAL
        post-reset state.

        Same 8-field payload as ``build_episode_config_record``, but the
        world/start/goal/yaw values are read from the env's attributes
        after reset (the admitted values round-trip float32 exactly
        through the cfg branch), NOT from the admit-time record — so the
        runtime hash and the episode_config_hash agree on the honest path
        and the buffer / gate audit hard-fail on any mismatch. Captured at
        reset time only: a post-step state legitimately differs.
        """
        env = self._envs[int(env_id)]
        payload = {
            "world_uid": str(
                getattr(env, "world_id", spec.world_uid)
            ),
            "role": spec.role,
            "split": spec.split,
            "admission_id": spec.admission_id,
            "episode_seed": int(seed),
            "start": [float(v) for v in env._drone_pos],
            "goal": [float(v) for v in env.goal_w],
            "start_yaw": float(env._drone_yaw),
        }
        return canonical_sha256(payload)

    def observation_window_mapping(self, slot_obs: dict, env_id: int) -> dict:
        """R7-5 window provenance of ONE slot — info-carried, NEVER the
        policy obs dict (model-input pollution rule).

        The window's last ``contract.obs_steps`` frames + the recorded
        per-frame valid flags / sha256 hashes / ray reports + BOTH
        independent hashes. The observation-window gate re-validates
        every frame through its own recompute and cross-checks the
        recorded values; any drift fails closed.
        """
        env = self._envs[int(env_id)]
        obs_steps = self.contract.obs_steps
        valid_hist = np.asarray(
            getattr(env, "_point_cloud_valid_history", np.zeros(8, dtype=bool))
        )
        hash_hist = list(
            getattr(env, "_point_cloud_frame_hash_history", [""] * 8)
        )
        report_hist = list(
            getattr(env, "_ray_report_history", [None] * 8)
        )
        return {
            "points": np.asarray(slot_obs["point_cloud"])[-obs_steps:],
            "cam_far": self._cam_far,
            "point_cloud_valid": [bool(v) for v in valid_hist[-obs_steps:]],
            "point_cloud_frame_hash": hash_hist[-obs_steps:],
            "ray_report_history": report_hist[-obs_steps:],
            "source_manifest_hash": self._specs[env_id].manifest_entry_hash,
            "runtime_episode_hash": self._runtime_episode_hashes[env_id],
        }

    def slot_record(self, env_id: int) -> Optional[EpisodeConfigRecord]:
        """Per-slot EpisodeConfigRecord (None = never admitted)."""
        return self._records[int(env_id)]

    def slot_spec(self, env_id: int) -> Optional[EpisodeTaskSpec]:
        """Per-slot EpisodeTaskSpec (None = never admitted)."""
        return self._specs[int(env_id)]

    # ── per-slot full state (R5.3: active-slot continuation across
    #    rollout boundaries + exact resume) ─────────────────────────

    def state_dict(self, env_id: int) -> Dict[str, Any]:
        """Curated full state of ONE slot: the frozen env's kinematic
        episode state + the adapter's slot bookkeeping. Arrays are kept
        numpy (the checkpoint serializes them). Raises on an invalid
        env_id; works for ACTIVE and DONE/VOIDED/INITIAL slots."""
        env_id = int(env_id)
        assert 0 <= env_id < self.n_envs, f"bad env_id {env_id}"
        env = self._envs[env_id]
        record = self._records[env_id]
        state = {
            "env_id": env_id,
            "adapter": {
                "done": bool(self._done[env_id]),
                "voided": bool(self._voided[env_id]),
                "record": None if record is None else asdict(record),
                "spec": None if self._specs[env_id] is None
                else asdict(self._specs[env_id]),
                # R7-5: runtime episode hash of the slot's current episode.
                "runtime_episode_hash": self._runtime_episode_hashes[env_id],
            },
            "env": {
                "world_id": str(getattr(env, "world_id", "")),
                "obstacles_json": str(getattr(env, "obstacles_json", "")),
                "drone_pos": np.asarray(
                    getattr(env, "_drone_pos", np.zeros(3)),
                    dtype=np.float32,
                ).copy(),
                "drone_yaw": float(getattr(env, "_drone_yaw", 0.0)),
                "drone_vel": np.asarray(
                    getattr(env, "_drone_vel", np.zeros(3)),
                    dtype=np.float32,
                ).copy(),
                "goal_w": np.asarray(
                    getattr(env, "goal_w", np.zeros(3)), dtype=np.float32,
                ).copy(),
                "episode_step": int(getattr(env, "episode_step", 0)),
                "episode_config": getattr(env, "_episode_config", None),
                "disturb_schedule": getattr(env, "_disturb_schedule", None),
                "disturb_idx": int(getattr(env, "_disturb_idx", 0)),
                "count_stuck": int(getattr(env, "_count_stuck", 0)),
                "prev_dist": None
                if getattr(env, "_prev_dist", None) is None
                else float(env._prev_dist),
                "collision_detected": bool(
                    getattr(env, "_collision_detected", False)
                ),
                "done_reason": str(getattr(env, "_done_reason", "none")),
                "pc_history": np.asarray(
                    getattr(env, "_pc_history",
                            np.zeros((8, 4096, 3), np.float32)),
                    dtype=np.float32,
                ).copy(),
                "state_history": np.asarray(
                    getattr(env, "_state_history",
                            np.zeros((8, 9), np.float32)),
                    dtype=np.float32,
                ).copy(),
                # R7-5: provenance histories (pickle-safe: numpy bool
                # array + lists of str / RayGeometryReport).
                "point_cloud_valid_history": np.asarray(
                    getattr(env, "_point_cloud_valid_history",
                            np.zeros(8, dtype=bool)),
                    dtype=bool,
                ).copy(),
                "ray_report_history": list(
                    getattr(env, "_ray_report_history", [None] * 8)
                ),
                "point_cloud_frame_hash_history": list(
                    getattr(env, "_point_cloud_frame_hash_history", [""] * 8)
                ),
                "prev_phi": float(getattr(env, "_prev_phi", 0.0)),
                "episode_rng_seed": int(
                    getattr(env, "_episode_rng_seed", 0)
                ),
                "rng_state": getattr(
                    getattr(env, "_rng", None), "__getstate__", lambda: None
                )(),
            },
        }
        return state

    def load_state_dict(self, env_id: int, state: Dict[str, Any]) -> None:
        """Restore one slot's full state (the inverse of state_dict).
        The slot must be DONE/VOIDED/INITIAL (a live ACTIVE slot is
        never silently replaced); restore makes it ACTIVE."""
        env_id = int(env_id)
        assert 0 <= env_id < self.n_envs, f"bad env_id {env_id}"
        if not self._done[env_id]:
            raise RuntimeError(
                f"env slot {env_id} is ACTIVE — load_state_dict() only "
                "restores closed slots"
            )
        env = self._envs[env_id]
        e = state["env"]
        a = state["adapter"]
        env.world_id = e["world_id"]
        env.obstacles_json = e["obstacles_json"]
        env._set_world(e["world_id"], e["obstacles_json"])
        env._drone_pos = np.asarray(e["drone_pos"], dtype=np.float32).copy()
        env._drone_yaw = float(e["drone_yaw"])
        env._drone_vel = np.asarray(e["drone_vel"], dtype=np.float32).copy()
        env.goal_w = np.asarray(e["goal_w"], dtype=np.float32).copy()
        env.episode_step = int(e["episode_step"])
        env._episode_config = e["episode_config"]
        env._disturb_schedule = e["disturb_schedule"]
        env._disturb_idx = int(e["disturb_idx"])
        env._count_stuck = int(e["count_stuck"])
        env._prev_dist = e["prev_dist"]
        env._collision_detected = bool(e["collision_detected"])
        env._done_reason = str(e["done_reason"])
        env._pc_history = np.asarray(e["pc_history"],
                                     dtype=np.float32).copy()
        env._state_history = np.asarray(e["state_history"],
                                        dtype=np.float32).copy()
        # R7-5: restore the provenance histories row-aligned with
        # _pc_history (defaults for pre-R7-5 checkpoints degrade the
        # gate to recompute-only — never silently wrong).
        env._point_cloud_valid_history = np.asarray(
            e.get(
                "point_cloud_valid_history",
                np.zeros(8, dtype=bool),
            ),
            dtype=bool,
        ).copy()
        env._ray_report_history = list(
            e.get("ray_report_history", [None] * 8)
        )
        env._point_cloud_frame_hash_history = list(
            e.get("point_cloud_frame_hash_history", [""] * 8)
        )
        env._prev_phi = float(e["prev_phi"])
        env._episode_rng_seed = int(e["episode_rng_seed"])
        if e["rng_state"] is not None and getattr(env, "_rng", None) is not None:
            env._rng.__setstate__(e["rng_state"])
        self._done[env_id] = bool(a["done"])
        self._voided[env_id] = bool(a["voided"])
        self._records[env_id] = (
            None if a["record"] is None
            else EpisodeConfigRecord(**a["record"])
        )
        self._specs[env_id] = (
            None if a["spec"] is None else EpisodeTaskSpec(**a["spec"])
        )
        self._runtime_episode_hashes[env_id] = str(
            a.get("runtime_episode_hash", "")
        )

    # ── helpers ──────────────────────────────────────────────────

    def _wrap_info(self, env_id: int, obs: dict, env_info: dict) -> dict:
        """Adapter info entries: env_id + world_uid/role/hit_count + BOTH
        hashes (the buffer trace reads them from here) + R7-5 window
        provenance + forwarding."""
        info = {
            "env_id": int(env_id),
            "world_uid": self._specs[env_id].world_uid,
            "role": self._specs[env_id].role,
            "hit_count": self._hit_count(obs["point_cloud"][-1]),
            "done_reason": env_info.get("done_reason", "none"),
            "manifest_entry_hash": self._specs[env_id].manifest_entry_hash,
            "episode_config_hash": self._records[env_id].episode_config_hash,
            # R7-5 dual hashes: source_manifest_hash (manifest identity,
            # same value as manifest_entry_hash) + runtime_episode_hash
            # (env's actual post-reset state) — the buffer hard-fails on
            # empty/malformed/mismatched values.
            "source_manifest_hash": self._specs[env_id].manifest_entry_hash,
            "runtime_episode_hash": self._runtime_episode_hashes[env_id],
            # R7-5: full-window provenance rides the info dict (never the
            # policy obs dict); the window gate re-validates from it.
            "observation_window_mapping": self.observation_window_mapping(
                obs, env_id
            ),
            # R5.9-C1 (chain-E review P1-1): forward the env's EpisodeOutcome
            # dict — the buffer/artifact consumers read it from here
            # (train_loop reads info[k].get("outcome", {})), and the old
            # drop made eval artifact success/collision always False.
            "outcome": env_info.get("outcome"),
        }
        # The unique EpisodeConfigRecord id (world:role:episode_seed),
        # NOT the env's echo of the old admission_id.
        info["episode_id"] = self._records[env_id].episode_id
        # Analytic ray-geometry evidence for no-hit frames (the gate
        # accepts hit_count=0 only with it). The slot's env calls the
        # wrapper synchronously, so last_report() matches the newest frame.
        wrapper = self._validated_cache.get(
            self._pool_path(self._specs[env_id].world_uid)
        )
        if wrapper is not None:
            rpt = wrapper.last_report()
            if rpt is not None:
                info["ray_report"] = rpt
        return info

    def _hit_count(self, points: np.ndarray) -> int:
        """Rows with a REAL return: distance from the camera mount strictly
        below the far clip (0.1 m band).

        The old heuristic ("rows not equal to (far, far, far)") broke
        under the physical far-endpoint encoding — far-dome rows sit at
        distance == far in DIFFERENT directions, so they are not the
        coordinate triple (10, 10, 10) and would read as hits. Distance-
        based counting reads 0 for both the far-dome (dist ≈ 10.0) and the
        legacy all-far fallback (dist ≈ 17.3), and the real hit count for
        genuine frames (any return within 9.9 m counts).

        R7-5: delegates to the shared env-side helper so the recorded and
        recomputed counts always use the same formula.
        """
        return frame_hit_count(points, self._cam_far)

    @staticmethod
    def _stack(obs_list: List[dict]) -> dict:
        """Stack per-slot (8, 4096, 3)/(8, 9) frame windows along a new
        leading axis: (n_envs, 8, 4096, 3)/(n_envs, 8, 9)."""
        return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
