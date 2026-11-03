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



class TaskBankMixtureSampler:
    """Episode mixture; returns factory tasks and exposes the selected reset ticket."""

    SCHEMA_VERSION = "routea_task_bank_mixture_v1"
    SOURCES = ("original", "L0", "L1", "L23_random", "L23_near", "L4")
    MIXTURE = (.15, .10, .15, .30, .20, .10)

    def __init__(self, tasks, bank_path, bank_sha256, seed, mixture=MIXTURE):
        import json
        from pathlib import Path

        self.original = RoleBalancedTaskSampler(tasks, seed)
        self.tasks = self.original.tasks
        self.seed = int(seed)
        values = [mixture[k] for k in self.SOURCES] if isinstance(mixture, dict) else mixture
        self.mixture = tuple(float(v) for v in values)
        if self.mixture != self.MIXTURE:
            raise ValueError("F5 mixture must be the registered fixed proportions")
        raw = Path(bank_path).read_bytes()
        self.bank_sha256 = hashlib.sha256(raw).hexdigest()
        if self.bank_sha256 != bank_sha256:
            raise ValueError("task bank hash identity mismatch")
        bank = json.loads(raw.decode("utf-8"))
        if bank.get("schema") != "routea_training_task_bank_v1":
            raise ValueError("task bank schema identity mismatch")
        self._base_by_world = {t.world_id: t for t in self.tasks}
        if len(self._base_by_world) != len(self.tasks) or set(bank["worlds"]) != set(self._base_by_world):
            raise ValueError("task bank world set differs from admitted training tasks")
        for world, base in self._base_by_world.items():
            record = bank["worlds"][world]
            if record["role"] != base.role or record["obstacle_sha256"] != hashlib.sha256(
                    Path(base.obstacle_json).read_bytes()).hexdigest():
                raise ValueError("task bank world geometry/role identity mismatch")
        self._rows = {s: {"dense": {}, "sparse": {}} for s in self.SOURCES[1:]}
        seen = set()
        for row in bank["tasks"]:
            world = row["world"]
            if world not in self._base_by_world or row["role"] != self._base_by_world[world].role:
                raise ValueError("task bank row references unknown world/role")
            if row["task_id"] in seen:
                raise ValueError("duplicate task bank task id")
            seen.add(row["task_id"])
            if (np.asarray(row["start"]).shape != (3,) or np.asarray(row["goal"]).shape != (3,)
                    or not np.isfinite(row["start"] + row["goal"] + [row["start_yaw"]]).all()):
                raise ValueError("task bank reset ticket must contain finite positions/yaw")
            tier, kind = row["tier"], row["kind"]
            if tier not in ("L0", "L1", "L2", "L3", "L4") or kind not in ("random", "near_obstacle_start"):
                raise ValueError("unknown task bank tier/kind")
            source = tier if tier in ("L0", "L1", "L4") else (
                "L23_random" if kind == "random" else "L23_near")
            n = int(bank["worlds"][world]["n_obstacles"])
            if row["n_world_obstacles"] != n:
                raise ValueError("task bank obstacle count identity mismatch")
            category = "dense" if n > 8 else "sparse"
            self._rows[source][category].setdefault(world, []).append(row)
        self.worlds_by_source = {}
        for source, classes in self._rows.items():
            self.worlds_by_source[source] = {}
            for category, worlds in classes.items():
                if not worlds:
                    raise ValueError("task bank mixture has empty source/class: "+source+"/"+category)
                for rows in worlds.values():
                    rows.sort(key=lambda row: row["task_id"])
                self.worlds_by_source[source][category] = tuple(sorted(worlds))
        self._rngs = {}
        for name in ("source", "world", "task"):
            digest = hashlib.sha256((str(self.seed)+"\x00f5_"+name).encode("ascii")).digest()
            self._rngs[name] = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        self._counts = {source: 0 for source in self.SOURCES}
        self._world_counts = {}
        self.last_reset_options = None
        self.last_metadata = None

    def sample(self):
        source = self.SOURCES[int(self._rngs["source"].choice(len(self.SOURCES), p=self.mixture))]
        self._counts[source] += 1
        if source == "original":
            task = self.original.sample()
            self.last_reset_options = None
            self.last_metadata = {"source": source, "world": task.world_id, "tier": None,
                                  "kind": "original", "bank_task_id": None}
        else:
            dense_probability = .5 if source in ("L0", "L1") else .8
            category = "dense" if self._rngs["world"].random() < dense_probability else "sparse"
            worlds = self.worlds_by_source[source][category]
            world = worlds[int(self._rngs["world"].integers(len(worlds)))]
            rows = self._rows[source][category][world]
            row = rows[int(self._rngs["task"].integers(len(rows)))]
            task = self._base_by_world[world]
            self.last_reset_options = {"fixed_start_goal": {
                "start_pos": list(row["start"]), "goal_pos": list(row["goal"]),
                "start_yaw": row["start_yaw"]}}
            self.last_metadata = {"source": source, "world": world, "tier": row["tier"],
                                  "kind": row["kind"], "bank_task_id": row["task_id"],
                                  "world_class": category}
        key = source + "/" + task.world_id
        self._world_counts[key] = self._world_counts.get(key, 0) + 1
        return task

    def state_dict(self):
        return {"schema_version": self.SCHEMA_VERSION, "bank_sha256": self.bank_sha256,
                "seed": self.seed, "mixture": list(self.mixture),
                "task_hashes": [t.sampling_identity for t in self.tasks],
                "counts": dict(self._counts), "world_counts": dict(self._world_counts),
                "original": self.original.state_dict(),
                "rng_states": {k: copy.deepcopy(v.bit_generator.state) for k,v in self._rngs.items()}}

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ("schema_version", "bank_sha256", "seed", "mixture", "task_hashes"):
            if state.get(key) != expected[key]:
                raise ValueError("task bank sampler identity mismatch: "+key)
        counts, worlds = state["counts"], state["world_counts"]
        if (set(counts) != set(self.SOURCES) or any(type(v) is not int or v < 0 for v in counts.values())
                or any(type(v) is not int or v < 0 for v in worlds.values())
                or sum(worlds.values()) != sum(counts.values())
                or state["original"]["draw_index"] != counts["original"]
                or set(state["rng_states"]) != set(self._rngs)):
            raise ValueError("task bank sampler state counts/RNG mismatch")
        self.original.load_state_dict(state["original"])
        self._counts, self._world_counts = dict(counts), dict(worlds)
        for key, rng in self._rngs.items():
            rng.bit_generator.state = copy.deepcopy(state["rng_states"][key])
        self.last_reset_options = self.last_metadata = None


__all__ = ("RoleBalancedTaskSampler", "TaskSampler", "TaskBankMixtureSampler")
