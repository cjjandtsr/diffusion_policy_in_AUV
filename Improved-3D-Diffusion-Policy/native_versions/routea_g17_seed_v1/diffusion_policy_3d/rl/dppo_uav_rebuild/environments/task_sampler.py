"""Role-first deterministic episode-ticket sampler with exact resume state."""

from __future__ import annotations

import copy
import hashlib

import numpy as np

from .task_spec import EpisodeTaskSpec, TRAINABLE_ROLES


class RoleBalancedTaskSampler:
    SCHEMA_VERSION = "routea_role_sampler_v2"

    def __init__(self, tasks, seed: int):
        supplied = tuple(tasks)
        if not supplied or any(not task.training_eligible for task in supplied):
            raise ValueError("sampler accepts only non-empty training-eligible tasks")
        by_role = {role: [] for role in TRAINABLE_ROLES}
        for task in supplied:
            if task.role not in by_role:
                raise ValueError("unknown task role: {}".format(task.role))
            by_role[task.role].append(task)
        if any(not by_role[role] for role in TRAINABLE_ROLES):
            raise ValueError("every trainable role must have at least one task")
        self.tasks_by_role = {
            role: tuple(sorted(by_role[role], key=lambda task: task.sampling_identity))
            for role in TRAINABLE_ROLES
        }
        self.tasks = tuple(
            task for role in TRAINABLE_ROLES for task in self.tasks_by_role[role]
        )
        self.seed = int(seed)
        self._draw_index = 0
        self._counts = {role: 0 for role in TRAINABLE_ROLES}
        self._rngs = {}
        for role in TRAINABLE_ROLES:
            digest = hashlib.sha256(
                str(self.seed).encode("ascii") + b"\x00" + role.encode("ascii")
            ).digest()
            self._rngs[role] = np.random.default_rng(
                int.from_bytes(digest[:8], "big")
            )

    def sample(self) -> EpisodeTaskSpec:
        role = TRAINABLE_ROLES[self._draw_index % len(TRAINABLE_ROLES)]
        tasks = self.tasks_by_role[role]
        index = int(self._rngs[role].integers(0, len(tasks)))
        self._draw_index += 1
        self._counts[role] += 1
        return tasks[index]

    def state_dict(self):
        return {
            "schema_version": self.SCHEMA_VERSION,
            "roles": list(TRAINABLE_ROLES),
            "task_hashes_by_role": {
                role: [task.sampling_identity for task in self.tasks_by_role[role]]
                for role in TRAINABLE_ROLES
            },
            "draw_index": self._draw_index,
            "deficits": {
                role: self._draw_index / len(TRAINABLE_ROLES) - self._counts[role]
                for role in TRAINABLE_ROLES
            },
            "counts": dict(self._counts),
            "rng_states": {
                role: copy.deepcopy(self._rngs[role].bit_generator.state)
                for role in TRAINABLE_ROLES
            },
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        if state.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("sampler schema identity mismatch")
        if state.get("roles") != list(TRAINABLE_ROLES):
            raise ValueError("sampler role identity mismatch")
        if state.get("task_hashes_by_role") != expected["task_hashes_by_role"]:
            raise ValueError("sampler task identity mismatch")
        draw_index = state.get("draw_index")
        counts = state.get("counts")
        if isinstance(draw_index, bool) or int(draw_index) < 0:
            raise ValueError("sampler draw_index is invalid")
        if set(counts or {}) != set(TRAINABLE_ROLES):
            raise ValueError("sampler count identity mismatch")
        if sum(int(value) for value in counts.values()) != int(draw_index):
            raise ValueError("sampler counts disagree with draw_index")
        expected_deficits = {
            role: int(draw_index) / len(TRAINABLE_ROLES) - int(counts[role])
            for role in TRAINABLE_ROLES
        }
        if state.get("deficits") != expected_deficits:
            raise ValueError("sampler deficits disagree with counts")
        if set(state.get("rng_states") or {}) != set(TRAINABLE_ROLES):
            raise ValueError("sampler RNG role identity mismatch")
        self._draw_index = int(draw_index)
        self._counts = {role: int(counts[role]) for role in TRAINABLE_ROLES}
        for role in TRAINABLE_ROLES:
            self._rngs[role].bit_generator.state = copy.deepcopy(
                state["rng_states"][role]
            )


# Backward-compatible import name; semantics are role-first by construction.
TaskSampler = RoleBalancedTaskSampler


__all__ = ("RoleBalancedTaskSampler", "TaskSampler")
