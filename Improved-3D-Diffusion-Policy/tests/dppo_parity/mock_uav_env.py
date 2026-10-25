"""Deterministic mock of the T7-target env interface for the T5 e2e loop.

The frozen KinematicUAVEnv exposes a 3-frame observation window
(kinematic_uav_env_v2.py:182); the T7 adapter must build the T_h=8 history
buffer. This mock implements that adapter-level interface (rolling T_h-frame
history, reset -> (obs, info), step(executed_actions (1, A_e, 11)) ->
(obs, reward, terminated, truncated, info)) so T5 can wire and close the
loop; real-env wiring stays T7 (environment_adapter.py:274/285).

History slides by one frame per step (like the frozen env's _push_obs), so
next_obs[t] == obs[t+1] holds and RolloutBuffer invariant 3 passes.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from dataclasses import asdict

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    AdmissionExhausted,
    EpisodeConfigRecord,
    EpisodeTaskSpec,
    SampledEpisodeTask,
    build_episode_config_record,
)


class MockUAVEnv:
    """Deterministic scripted env; reach at reach_step (reward +100), then
    truncated at max_steps. Optionally emits far-plane garbage frames at
    invalid_episode_steps (hit_count 0) to exercise gate routing, and
    reset_invalid=True makes every reset frame far-plane (reset-phase
    REJECT_EPISODE path).

    R2.2: ``episode_scripts`` overrides per-episode behavior by world_uid,
    e.g. {"world_comp": {"reach_step": 5}, "world_void": {
    "invalid_episode_steps": [2]}} — so one run can mix completed and
    voided episodes for the truthful role-accounting tests.
    """

    def __init__(
        self,
        contract: RLSystemContract,
        reach_step: int = 5,
        max_steps: int = 12,
        cam_far: float = 50.0,
        invalid_episode_steps: Optional[object] = None,
        reset_invalid: bool = False,
        seed: int = 0,
        episode_scripts: Optional[Dict[str, dict]] = None,
    ) -> None:
        self.contract = contract
        self._episode_scripts = dict(episode_scripts) if episode_scripts else {}
        if isinstance(invalid_episode_steps, int):
            invalid_episode_steps = [invalid_episode_steps]
        self._default_invalid_steps = (
            set(invalid_episode_steps) if invalid_episode_steps else set()
        )
        self._default_reset_invalid = reset_invalid
        self._default_reach_step = reach_step
        self._default_max_steps = max_steps
        self._cam_far = float(cam_far)
        self._reach_step = reach_step
        self._max_steps = max_steps
        self._invalid_steps: set = set()
        self._reset_invalid = False
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._step = 0
        self._spec: Optional[EpisodeTaskSpec] = None
        self._record: Optional[EpisodeConfigRecord] = None
        self._hist_pc: deque = deque()
        self._hist_s: deque = deque()
        self._start: np.ndarray = np.zeros(3, dtype=np.float32)
        self._goal: np.ndarray = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self._done = True  # Must reset before stepping
        # VOIDED mirror of the real adapter (slot_state returns VOIDED
        # only after mark_voided; reset clears it)
        self._voided = False

    def _resolve_episode_script(
        self, world_uid: str,
    ) -> Tuple[dict, int, int, set, bool]:
        """Per-world episode script, read-only (rework #4): returns
        (script, reach_step, max_steps, invalid_steps, reset_invalid) —
        the old version MUTATED the slot's scratch fields, which broke the
        two-phase zero-partial-reset semantics (a scripted admission
        failure must not leave ANY observable state behind)."""
        scr = self._episode_scripts.get(world_uid, {})
        inv = scr.get("invalid_episode_steps", self._default_invalid_steps)
        invalid_steps = (
            {inv} if isinstance(inv, int) else set(inv) if inv else set()
        )
        return (
            scr,
            scr.get("reach_step", self._default_reach_step),
            scr.get("max_steps", self._default_max_steps),
            invalid_steps,
            bool(scr.get("reset_invalid", self._default_reset_invalid)),
        )

    @property
    def cam_far(self) -> float:
        return self._cam_far

    # Single-slot vector contract mirror — the real adapter exposes n_envs
    # slots; the mock is the n_envs=1 case with the same call surface
    # (reset/reset_at/step + done-guard + (1,) array returns).
    n_envs = 1

    def is_active(self, env_id: int) -> bool:
        assert int(env_id) == 0
        return not self._done

    # ── R5.3 duck-typed slot surface (mirrors the real adapter) ─────

    def slot_obs(self, env_id: int) -> dict:
        assert int(env_id) == 0, f"mock is single-slot, got {env_id}"
        return {
            "point_cloud": np.stack(list(self._hist_pc)).astype(np.float32),
            "agent_pos": np.stack(list(self._hist_s)).astype(np.float32),
        }

    def slot_record(self, env_id: int) -> Optional[EpisodeConfigRecord]:
        assert int(env_id) == 0
        return self._record

    def slot_spec(self, env_id: int) -> Optional[EpisodeTaskSpec]:
        assert int(env_id) == 0
        return self._spec

    def state_dict(self, env_id: int) -> Dict[str, Any]:
        assert int(env_id) == 0
        return {
            "adapter": {
                "done": bool(self._done),
                "voided": bool(self._voided),
                "record": None if self._record is None
                else asdict(self._record),
                "spec": None if self._spec is None
                else asdict(self._spec),
            },
            "env": {
                "episode_step": int(self._step),
                "start": np.asarray(self._start,
                                    dtype=np.float32).copy(),
                "goal": np.asarray(self._goal, dtype=np.float32).copy(),
                "reach_step": self._reach_step,
                "max_steps": self._max_steps,
                "invalid_steps": sorted(self._invalid_steps),
                "reset_invalid": bool(self._reset_invalid),
                "rng_state": self._rng.get_state(),
                "pc_history": np.stack(
                    list(self._hist_pc)
                ).astype(np.float32) if self._hist_pc else np.zeros(
                    (0, 4096, 3), np.float32
                ),
                "state_history": np.stack(
                    list(self._hist_s)
                ).astype(np.float32) if self._hist_s else np.zeros(
                    (0, 9), np.float32
                ),
            },
        }

    def load_state_dict(self, env_id: int, state: Dict[str, Any]) -> None:
        assert int(env_id) == 0
        if not self._done:
            raise RuntimeError(
                "mock slot 0 is ACTIVE — load_state_dict() only restores "
                "closed slots"
            )
        a = state["adapter"]
        e = state["env"]
        self._done = bool(a["done"])
        self._voided = bool(a["voided"])
        self._record = (
            None if a["record"] is None
            else EpisodeConfigRecord(**a["record"])
        )
        self._spec = (
            None if a["spec"] is None else EpisodeTaskSpec(**a["spec"])
        )
        self._step = int(e["episode_step"])
        self._start = np.asarray(e["start"], dtype=np.float32).copy()
        self._goal = np.asarray(e["goal"], dtype=np.float32).copy()
        self._reach_step = e["reach_step"]
        self._max_steps = e["max_steps"]
        self._invalid_steps = set(e["invalid_steps"])
        self._reset_invalid = bool(e["reset_invalid"])
        self._rng.set_state(e["rng_state"])
        self._hist_pc = deque(
            np.asarray(e["pc_history"], dtype=np.float32)
        )
        self._hist_s = deque(
            np.asarray(e["state_history"], dtype=np.float32)
        )

    # ── end R5.3 slot surface ───────────────────────────────────────


    def reset(
        self, task_specs: List[EpisodeTaskSpec], seeds: List[int]
    ) -> Tuple[Dict[str, np.ndarray], List[dict]]:
        assert len(task_specs) == self.n_envs and len(seeds) == self.n_envs
        return self.reset_at([0], task_specs, seeds)

    def reset_at(
        self, env_ids: List[int], task_specs: List[EpisodeTaskSpec],
        seeds: List[int],
    ) -> Tuple[Dict[str, np.ndarray], List[dict]]:
        assert list(env_ids) == [0] and len(task_specs) == 1, (
            "mock is the n_envs=1 case"
        )
        # ACTIVE guard — the single-slot mock previously re-reset ACTIVE
        # slots SILENTLY, hiding the loop's leak from the whole suite.
        # Mirror the real adapter exactly: an in-flight episode must never
        # be replaced.
        if not self._done:
            raise RuntimeError(
                "env slot 0 is ACTIVE — reset_at() only resets"
                " done/voided slots (guide R2.4); an in-flight episode must"
                " never be silently replaced"
            )
        # Two-phase atomic reset: prepare is read-only (a scripted
        # admission_exhausted raises here, before any observable mutation);
        # apply is the single mutation step.
        prepared = self._prepare_reset(task_specs[0], int(seeds[0]))
        return self._apply_reset(task_specs[0], int(seeds[0]), prepared)

    def _prepare_reset(
        self, spec: EpisodeTaskSpec, seed: int,
    ) -> Tuple[dict, int, int, set, bool, np.ndarray, np.ndarray,
               EpisodeConfigRecord, np.random.RandomState]:
        """Phase 1 of reset (rework #4): script resolution + sampled
        geometry + record — READ-ONLY, no observable slot state is touched.
        A scripted ``admission_exhausted`` world raises AdmissionExhausted
        here, so a multi-slot caller can prepare every slot before applying
        any of them: zero partial resets, exactly like the real adapter.
        The RandomState is created here and consumed by _apply_reset for
        the reset frames, so the RNG stream is byte-identical to the
        pre-split implementation."""
        scr, reach_step, max_steps, invalid_steps, reset_invalid = (
            self._resolve_episode_script(spec.world_uid)
        )
        if scr.get("admission_exhausted", False):
            raise AdmissionExhausted(
                f"admission failed for {spec.world_uid} (scripted) —"
                " sampler must switch world/task"
            )
        rng = np.random.RandomState(seed)
        start = rng.uniform(-5.0, 5.0, size=3).astype(np.float32)
        goal = start + np.array([6.0, 4.0, 1.0], dtype=np.float32)
        # Mirror the real adapter's EpisodeConfigRecord via the shared
        # builder (planner-less defaults for the metrics) — identical
        # geometry yields the IDENTICAL episode_config_hash on both sides.
        record = build_episode_config_record(spec, seed, start, goal, 0.0)
        return (scr, reach_step, max_steps, invalid_steps, reset_invalid,
                start, goal, record, rng)

    def _apply_reset(
        self, spec: EpisodeTaskSpec, seed: int, prepared: Tuple[dict, int,
        int, set, bool, np.ndarray, np.ndarray, EpisodeConfigRecord,
        np.random.RandomState],
    ) -> Tuple[Dict[str, np.ndarray], List[dict]]:
        """Phase 2 of reset (rework #4): apply a prepared episode — the
        only step that mutates observable slot state. Returns the same
        (obs, info) as the pre-split reset_at."""
        (_, reach_step, max_steps, invalid_steps, reset_invalid,
         start, goal, record, rng) = prepared
        self._spec = spec
        self._step = 0
        self._reach_step = reach_step
        self._max_steps = max_steps
        self._invalid_steps = invalid_steps
        self._reset_invalid = reset_invalid
        self._seed = seed
        self._rng = rng
        self._start = start
        self._goal = goal
        self._record = record
        self._hist_pc.clear()
        self._hist_s.clear()
        self._done = False
        self._voided = False
        # reset history holds steps -(T_h)..-1 (negative): reset_invalid
        # makes every reset frame (including the newest, index -1) far-plane
        # -> reset-phase REJECT path
        for k in range(-self.contract.history_steps, 0):
            self._hist_pc.append(self._new_frame(k))
            self._hist_s.append(self._state_frame(k))
        return self._obs(), [self._info()]

    def step(self, executed_actions: np.ndarray) -> Tuple[Dict[str, np.ndarray],
                                                          np.ndarray, np.ndarray,
                                                          np.ndarray, List[dict]]:
        # executed_actions (1, A_e, 11) physical (mock ignores content)
        # G5 mirror (design §5.7): the loop hands over whatever the E040
        # normalizer returns — a CUDA torch tensor — so the boundary
        # converts tensor -> cpu numpy exactly like RealUAVEnv.step.
        if torch.is_tensor(executed_actions):
            executed_actions = executed_actions.detach().cpu().numpy()
        arr = np.asarray(executed_actions)
        assert arr.shape[0] == self.n_envs, "mock is the n_envs=1 case"
        if self._done:
            raise RuntimeError(
                "env slot 0 is done — reset_at() it before stepping"
            )
        return self._step_internal(arr[0])

    # The loop drives the ACTIVE subset through step_at / mark_voided —
    # the n_envs=1 mock mirrors the real adapter's vector surface with the
    # [0]-only specialization.
    def step_at(self, env_ids: List[int], executed_actions: Any):
        ids = [int(i) for i in env_ids]
        assert ids == [0], "mock is the n_envs=1 case"
        return self.step(np.asarray(executed_actions))

    def mark_voided(self, env_ids: List[int]) -> None:
        ids = [int(i) for i in env_ids]
        assert ids == [0], "mock is the n_envs=1 case"
        if self._done:
            raise RuntimeError(
                "env slot 0 is not ACTIVE — mark_voided() closes"
                " observation-faulted episodes only"
            )
        self._voided = True
        self._done = True

    def abort_episode(self, env_ids: List[int]) -> None:
        """Explicit ACTIVE -> DONE discard: mirror of the real
        adapter — freeze_paired_bank closes its probe episode this way.
        DONE but NOT VOIDED: the caller accepts the loss without a fault
        verdict. Same boundary: raises when the slot is not ACTIVE."""
        ids = [int(i) for i in env_ids]
        if ids != [0]:
            raise ValueError("mock is the n_envs=1 case — only slot 0 exists")
        if self._done:
            raise RuntimeError(
                "env slot 0 is not ACTIVE — abort_episode() discards"
                " in-flight episodes only"
            )
        self._done = True

    def reject_episode(self, env_ids: List[int]) -> None:
        """Explicit ACTIVE -> DONE (rework #4, 审核裁决 P1-1): mirror of
        the real adapter — closes a slot whose reset frame failed the gate.
        Same boundary: raises when the slot is not ACTIVE."""
        ids = [int(i) for i in env_ids]
        assert ids == [0], "mock is the n_envs=1 case"
        if self._done:
            raise RuntimeError(
                "env slot 0 is not ACTIVE — reject_episode() closes"
                " reset-rejected episodes only"
            )
        self._done = True

    def mark_truncated(self, env_ids: List[int]) -> None:
        """Explicit ACTIVE -> DONE (rework #4, 审核裁决 P1-3): mirror of
        the real adapter — closes a slot the LOOP truncated at max_ep_len.
        Same boundary: raises when the slot is not ACTIVE."""
        ids = [int(i) for i in env_ids]
        assert ids == [0], "mock is the n_envs=1 case"
        if self._done:
            raise RuntimeError(
                "env slot 0 is not ACTIVE — mark_truncated() closes"
                " loop-truncated episodes only"
            )
        self._done = True

    def slot_state(self, env_id: int = 0) -> str:
        """Read-only slot lifecycle query (rework #4), mirroring the real
        adapter: INITIAL / ACTIVE / DONE / VOIDED."""
        assert int(env_id) == 0, "mock is the n_envs=1 case"
        if not self._done:
            return "ACTIVE"
        if self._voided:
            return "VOIDED"
        if self._record is None:
            return "INITIAL"
        return "DONE"

    def is_active(self, env_id: int = 0) -> bool:
        """Read-only ACTIVE query (rework #4), mirroring the real adapter."""
        assert int(env_id) == 0, "mock is the n_envs=1 case"
        return not self._done

    def _step_internal(self, action_row: np.ndarray) -> Tuple[Dict[str, np.ndarray],
                                                              np.ndarray, np.ndarray,
                                                              np.ndarray, List[dict]]:
        """One single-slot step; step() wraps it for the (1, A_e, 11) API."""
        self._step += 1
        # arriving frame during loop-step s is frame(s) (self._step-1 after
        # the increment), so invalid_episode_steps names the episode_step of
        # the dropped transition (test_first_fault_safe_ends_episode)
        self._hist_pc.append(self._new_frame(self._step - 1))
        self._hist_s.append(self._state_frame(self._step - 1))
        self._hist_pc.popleft()
        self._hist_s.popleft()
        terminated = self._step >= self._reach_step
        truncated = self._step >= self._max_steps
        self._done = bool(terminated or truncated)
        reward = 100.0 if terminated else 0.0
        return (
            self._obs(),
            np.asarray([float(reward)]),
            np.asarray([terminated]),
            np.asarray([truncated]),
            [self._info()],
        )

    def current_episode_config(self) -> Dict[str, object]:
        """Episode-level metadata for the paired eval record (§4.17); R2.3:
        mirrors the real adapter — both hashes ride the dict."""
        r = self._record
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

    # -- internals --------------------------------------------------------
    def _new_frame(self, step: int) -> np.ndarray:
        frame = self._rng.uniform(
            -8.0, 8.0, size=(self.contract.point_count, 3)
        ).astype(np.float32)
        if step in self._invalid_steps or (self._reset_invalid and step < 0):
            frame[...] = self._cam_far  # far-plane garbage -> invalid frame
        return frame

    def _state_frame(self, step: int) -> np.ndarray:
        """Mirror the real env's 9D state [pos(3), goal_body(3), dist,
        sin_yaw, cos_yaw] (kinematic_uav_env_v2._build_state) — the mock
        flies along +x at the frozen start_yaw, so the eval trace reads
        REAL goal distance (s9[6]) and yaw (atan2(s9[7], s9[8])) from the
        frozen geometry, not placeholder zeros."""
        r = self._record
        pos = np.array([0.01 * max(step, 0), 0.0, 0.0], dtype=np.float32)
        goal = np.asarray(r.goal, dtype=np.float32)
        yaw = float(r.start_yaw)
        c, s = np.cos(yaw), np.sin(yaw)
        dx, dy, dz = goal - pos
        gbx, gby, gbz = c * dx + s * dy, -s * dx + c * dy, dz
        gd = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        return np.array(
            [pos[0], pos[1], pos[2], gbx, gby, gbz, gd,
             np.sin(yaw), np.cos(yaw)], dtype=np.float32,
        )

    def _info(self) -> dict:
        hit = 0 if (self._step - 1) in self._invalid_steps else 4096
        # R6-E §11.6.4 (F-10): mirror the real adapter's outcome
        # forwarding (environment_adapter._wrap_info forwards the env's
        # EpisodeOutcome dict) — the eval record builder's terminal-
        # category mapper consumes it, and a terminated step without a
        # done_reason would hard-fail as an unmappable state. The mock
        # never collides: reach -> reached, max_steps -> timeout.
        return {
            "env_id": 0,  # Info entries carry env_id (adapter mirror)
            "world_uid": self._spec.world_uid,
            "role": self._spec.role,
            "hit_count": hit,  # T7 requirement: env exposes per-frame hit count
            # Both hashes mirror the real adapter's reset/step info
            # (train_loop reads episode_config_hash from info[0]).
            "manifest_entry_hash": self._spec.manifest_entry_hash,
            "episode_config_hash": self._record.episode_config_hash,
            "outcome": {
                "success": self._step >= self._reach_step,
                "collision": False,
                "time_limit": self._step >= self._max_steps,
                "out_of_bounds": False,
                "altitude_exceed": False,
                "terminated": self._step >= self._reach_step,
                "truncated": self._step >= self._max_steps,
                "done_reason": (
                    "reached" if self._step >= self._reach_step
                    else "timeout" if self._step >= self._max_steps
                    else "none"
                ),
            },
        }

    def _obs(self) -> Dict[str, np.ndarray]:
        return {
            "point_cloud": np.stack(list(self._hist_pc))[None].astype(np.float32),
            "agent_pos": np.stack(list(self._hist_s))[None].astype(np.float32),
        }


class ScriptedTaskSampler:
    """Deterministic duck-typed EpisodeTaskSampler for loop-side tests.

    Returns scripted SampledEpisodeTask objects in order (cycling when the
    list is exhausted); attempt/episode indices and seeds advance strictly
    per call so consecutive run_e2e_batch calls continue (never restarting
    at 1). Used where a test needs a fixed task sequence (role-accounting,
    fault routing) instead of the production random RoleSampler.
    """

    def __init__(self, tasks: List[EpisodeTaskSpec], base_seed: int = 1000) -> None:
        assert tasks, "scripted sampler needs at least one task"
        self._tasks = list(tasks)
        self._base_seed = int(base_seed)
        self._calls = 0
        self._completed = 0

    def next_attempt(self) -> SampledEpisodeTask:
        task = self._tasks[self._calls % len(self._tasks)]
        idx = self._calls
        self._calls += 1
        return SampledEpisodeTask(
            task_spec=task,
            episode_seed=self._base_seed + idx,
            global_attempt_index=idx,
            global_episode_index=idx,
        )

    def mark_completed(self) -> None:
        self._completed += 1

    # The scripted sampler is resumable too — its cursor (calls) and
    # completed counter restore from the checkpoint, so a resumed run keeps
    # the seed sequence strictly advancing (never restarts at base_seed).
    # The payload mirrors EpisodeTaskSampler's schema shape.
    STATE_SCHEMA = "scripted_task_sampler/v1"

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.STATE_SCHEMA,
            "base_seed": self._base_seed,
            "calls": self._calls,
            "completed": self._completed,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("schema") != self.STATE_SCHEMA:
            raise ValueError(
                f"scripted sampler state schema mismatch: {state.get('schema')!r}"
                f" != {self.STATE_SCHEMA!r}"
            )
        if state.get("base_seed") != self._base_seed:
            raise ValueError(
                f"scripted sampler state base_seed mismatch:"
                f" {state.get('base_seed')!r} != {self._base_seed!r}"
            )
        self._calls = int(state["calls"])
        self._completed = int(state["completed"])

    # -- spy surface ----------------------------------------------------
    @property
    def calls(self) -> int:
        return self._calls

    @property
    def completed(self) -> int:
        return self._completed
