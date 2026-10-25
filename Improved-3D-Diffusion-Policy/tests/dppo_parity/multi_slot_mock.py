"""Multi-slot deterministic mock of the R2.4+ vector env contract.

Wraps independent single-slot MockUAVEnv instances so the train loop tests
exercise the full vector surface the REAL UAVVectorEnvAdapter exposes
(rework #3, 审核裁决 P1-1/P1-2):

  - reset_at(env_ids, task_specs, seeds): resets ONLY the named slots; an
    ACTIVE slot (not done, not voided) hard-fails BEFORE any mutation
  - step_at(env_ids, actions): steps only the named ACTIVE slots, results
    come back in env_ids INPUT order
  - step(actions): full-slot step with atomic preflight
  - mark_voided(env_ids): ACTIVE -> VOIDED (the observation-fault path);
    a VOIDED slot cannot step but can be reset_at

Per-slot episode scripts are keyed by world_uid, so one adapter runs
heterogeneous episode lengths (the precise-budget tail case).
"""

from typing import Any, Dict, List, Optional

import numpy as np

from mock_uav_env import MockUAVEnv


class MultiSlotMockUAVEnv:
    """n_envs-slot wrapper around independent MockUAVEnv instances."""

    def __init__(
        self,
        contract: Any,
        n_envs: int = 4,
        cam_far: float = 50.0,
        episode_scripts: Optional[Dict[str, dict]] = None,
        reach_step: int = 3,
        max_steps: int = 12,
    ) -> None:
        self.contract = contract
        self.n_envs = int(n_envs)
        self._cam_far = float(cam_far)
        self._envs = [
            MockUAVEnv(
                contract, reach_step=reach_step, max_steps=max_steps,
                cam_far=self._cam_far,
                episode_scripts=episode_scripts, seed=i,
            )
            for i in range(self.n_envs)
        ]
        # slot lifecycle: _done True == INITIAL (never reset) or
        # DONE/VOIDED (closed, reset_at legal); False == ACTIVE
        self._done: List[bool] = [True] * self.n_envs
        self._voided: List[bool] = [False] * self.n_envs
        self._records: List[Optional[dict]] = [None] * self.n_envs

    @property
    def cam_far(self) -> float:
        return self._cam_far

    # ── vector contract ────────────────────────────────────────────

    def reset(self, task_specs: List[Any], seeds: List[int]):
        assert len(task_specs) == self.n_envs and len(seeds) == self.n_envs
        return self.reset_at(list(range(self.n_envs)), task_specs, seeds)

    def _assert_not_active(self, env_ids: List[int]) -> None:
        active = [i for i in env_ids if not self._done[i]]
        if active:
            raise RuntimeError(
                f"env slots {active} are ACTIVE — reset_at() only resets"
                " done/voided slots; a reset here would silently discard an"
                " in-flight episode (rework #3)"
            )

    def reset_at(self, env_ids: List[int], task_specs: List[Any],
                 seeds: List[int]):
        ids = [int(i) for i in env_ids]
        assert len(ids) == len(task_specs) == len(seeds), (
            "reset_at needs len(env_ids) == len(task_specs) == len(seeds)"
        )
        assert len(set(ids)) == len(ids), f"duplicate env_ids: {ids}"
        self._assert_not_active(ids)  # atomic preflight BEFORE any mutation
        # Two-phase — prepare EVERY slot first (read-only; a scripted
        # AdmissionExhausted raises here before any mutation), then apply
        # all. A mid-batch admission failure leaves ZERO partial resets —
        # the exact semantics of the real adapter.
        prepared = [
            self._envs[env_id]._prepare_reset(spec, int(seed))
            for env_id, spec, seed in zip(ids, task_specs, seeds)
        ]
        obs_list: List[Dict[str, np.ndarray]] = []
        info_list: List[dict] = []
        for env_id, spec, seed, prep in zip(ids, task_specs, seeds, prepared):
            obs, info = self._envs[env_id]._apply_reset(spec, int(seed), prep)
            info[0]["env_id"] = env_id
            self._done[env_id] = False
            self._voided[env_id] = False
            self._records[env_id] = self._envs[env_id].current_episode_config()
            obs_list.append(obs)
            info_list.append(info[0])
        return self._stack(obs_list), info_list

    def step_at(self, env_ids: List[int], actions: Any):
        ids = [int(i) for i in env_ids]
        arr = np.asarray(actions, dtype=np.float32)
        n_envs, n_steps, n_dim = arr.shape
        assert n_envs == len(ids), "actions rows must match env_ids"
        assert n_steps == self.contract.exec_steps
        assert n_dim == self.contract.action_dim
        blocked = [i for i in ids if self._done[i]]
        if blocked:
            raise RuntimeError(
                f"env slots {blocked} are done — reset_at() them before"
                " stepping (no auto-reset hiding the boundary)"
            )
        obs_list: List[Dict[str, np.ndarray]] = []
        rewards: List[float] = []
        terminated_list: List[bool] = []
        truncated_list: List[bool] = []
        info_list: List[dict] = []
        for k, env_id in enumerate(ids):
            obs, reward, terminated, truncated, info = self._envs[env_id].step(
                arr[k][None])
            self._done[env_id] = bool(terminated[0] or truncated[0])
            info[0]["env_id"] = env_id
            obs_list.append(obs)
            rewards.append(float(reward[0]))
            terminated_list.append(bool(terminated[0]))
            truncated_list.append(bool(truncated[0]))
            info_list.append(info[0])
        return (
            self._stack(obs_list),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated_list, dtype=bool),
            np.asarray(truncated_list, dtype=bool),
            info_list,
        )

    def step(self, actions: Any):
        arr = np.asarray(actions)
        assert arr.shape[0] == self.n_envs, "step needs actions for ALL slots"
        return self.step_at(list(range(self.n_envs)), arr)

    def mark_voided(self, env_ids: List[int]) -> None:
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

    def reject_episode(self, env_ids: List[int]) -> None:
        """Explicit ACTIVE -> DONE (rework #4, 审核裁决 P1-1) — the loop
        closes a slot whose RESET frame failed the gate (REJECT_EPISODE).
        Same boundary as the real adapter: raises when a named slot is not
        ACTIVE."""
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
        """Explicit ACTIVE -> DONE (rework #4, 审核裁决 P1-3) — the loop
        closes a slot it truncated at max_ep_len (the env never signaled
        terminated/truncated). Same boundary as the real adapter."""
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
        """Read-only slot lifecycle query (rework #4), mirroring the real
        adapter: INITIAL / ACTIVE / DONE / VOIDED."""
        env_id = int(env_id)
        if not self._done[env_id]:
            return "ACTIVE"
        if self._voided[env_id]:
            return "VOIDED"
        if self._records[env_id] is None:
            return "INITIAL"
        return "DONE"

    def is_active(self, env_id: int) -> bool:
        """Read-only ACTIVE query (rework #4), mirroring the real adapter."""
        return not self._done[int(env_id)]

    def current_episode_config(self, env_id: int = 0) -> dict:
        if self._records[env_id] is None:
            raise RuntimeError(f"current_episode_config before reset (slot {env_id})")
        return self._records[env_id]

    # ── helpers ────────────────────────────────────────────────────

    def _stack(self, obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        return {
            "point_cloud": np.concatenate(
                [o["point_cloud"] for o in obs_list], axis=0),
            "agent_pos": np.concatenate(
                [o["agent_pos"] for o in obs_list], axis=0),
        }
