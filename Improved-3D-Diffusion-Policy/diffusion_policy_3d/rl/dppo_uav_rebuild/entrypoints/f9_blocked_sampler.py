"""Teacher-only blocked-task sampler for F9 stage A2.

Rejection-samples start/goal candidates from the same training task generator and the
same training worlds that PPO uses, keeping only candidates whose straight
start->goal line has a minimum obstacle clearance below 0.3 m (sampled every 0.05 m
with ``collision_geom.compute_obstacle_clearance``, the same oracle the dev/val
blocked split is derived from).  Candidates that reproduce one of the 85 frozen
library tasks (same world and both endpoints within 1e-6 m) are rejected.

PPO rollouts, the keep layer, the guidance setting, calibration, measurement and the
gate never use this sampler; only teacher episodes do.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from ..environments.task_sampler import RoleBalancedTaskSampler

BLOCKED_CLEARANCE_M = 0.3
STRAIGHT_STEP_M = 0.05
MAX_ATTEMPTS = 500
SAMPLER_SEED = 202611231
ENDPOINT_TOLERANCE_M = 1e-6


def library_tasks(plan_path):
    """`(world_id, start, goal)` of the frozen library tasks from the registered bank."""
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    bank = json.loads(Path(plan['banks']['train']['path']).read_text(encoding='utf-8'))
    rows = []
    for task in bank['tasks']:
        payload = task['payload']
        rows.append((str(task['world_id']),
                     np.asarray(payload['start_pos'], np.float64),
                     np.asarray(payload['goal_pos'], np.float64)))
    return tuple(rows)


def straight_min_clearance(start, goal, obstacles):
    """Minimum obstacle clearance along the straight start->goal line (0.05 m grid)."""
    from diffusion_policy_3d.env.collision_geom import compute_obstacle_clearance
    start = np.asarray(start, np.float64)
    goal = np.asarray(goal, np.float64)
    length = float(np.linalg.norm(goal - start))
    count = max(1, int(np.ceil(length / STRAIGHT_STEP_M)))
    best = float('inf')
    for index in range(count + 1):
        point = start + (goal - start) * (index / count)
        value = compute_obstacle_clearance(
            float(point[0]), float(point[1]), float(point[2]), obstacles)[0]
        if value < best:
            best = float(value)
    return best


def is_library_task(world_id, start, goal, library):
    """Same world and both endpoints within 1e-6 m of a frozen library task."""
    start = np.asarray(start, np.float64)
    goal = np.asarray(goal, np.float64)
    for library_world, library_start, library_goal in library:
        if str(library_world) != str(world_id):
            continue
        if (np.max(np.abs(start - library_start)) <= ENDPOINT_TOLERANCE_M
                and np.max(np.abs(goal - library_goal)) <= ENDPOINT_TOLERANCE_M):
            return True
    return False


class BlockedTeacherTaskSampler:
    """Role-balanced draw + rejection sampling on the straight-line clearance."""

    SCHEMA_VERSION = 'f9_blocked_teacher_sampler_v1'

    def __init__(self, tasks, factory, library, seed=SAMPLER_SEED):
        self.base = RoleBalancedTaskSampler(tasks, int(seed))
        self.tasks = self.base.tasks
        self.factory = factory
        self.library = tuple(library)
        self.seed = int(seed)
        digest = hashlib.sha256(
            str(self.seed).encode('ascii') + b'\x00f9_blocked_candidate').digest()
        self.rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))
        self._obstacles = {}
        self.accepted = 0
        self.candidate_attempts = 0
        self.rejection_histogram = dict(clearance=0, library=0, admission=0)
        self.last = None

    # -- candidate geometry ------------------------------------------------
    def obstacles(self, task):
        world = str(task.world_id)
        if world not in self._obstacles:
            from diffusion_policy_3d.env.collision_geom import load_obstacles
            self._obstacles[world] = load_obstacles(task.obstacle_json)
        return self._obstacles[world]

    def draw_candidate(self, task):
        """Draw one start/yaw/goal using the environment's own generator."""
        adapter = self.factory.make(task)
        environment = adapter.environment
        previous = environment._rng
        environment._rng = self.rng
        try:
            start, yaw, goal = environment._draw_start_goal_candidate(task.role)
            reasons = environment._candidate_rejection_reasons(start, goal, task.role)
        finally:
            environment._rng = previous
        return np.asarray(start, np.float64), float(yaw), np.asarray(goal, np.float64), list(reasons)

    # -- sampling ----------------------------------------------------------
    def sample(self):
        rejections = 0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            task = self.base.sample()
            start, yaw, goal, reasons = self.draw_candidate(task)
            self.candidate_attempts += 1
            if reasons:
                self.rejection_histogram['admission'] += 1
                rejections += 1
                continue
            clearance = straight_min_clearance(start, goal, self.obstacles(task))
            if not clearance < BLOCKED_CLEARANCE_M:
                self.rejection_histogram['clearance'] += 1
                rejections += 1
                continue
            if is_library_task(task.world_id, start, goal, self.library):
                self.rejection_histogram['library'] += 1
                rejections += 1
                continue
            self.accepted += 1
            self.last = dict(task=task, start=start, yaw=yaw, goal=goal,
                             clearance=float(clearance), rejections=rejections,
                             attempts=attempt)
            return self.last
        raise RuntimeError(
            'F9 blocked teacher sampler exceeded %d attempts for one task' % MAX_ATTEMPTS)

    def fixed_reset_options(self, picked):
        return {'fixed_start_goal': {
            'start_pos': [float(v) for v in picked['start']],
            'start_yaw': float(picked['yaw']),
            'goal_pos': [float(v) for v in picked['goal']]}}

    def acceptance_rate(self):
        if not self.candidate_attempts:
            return None
        return self.accepted / float(self.candidate_attempts)

    # -- identity / resume -------------------------------------------------
    def state_dict(self):
        return dict(schema_version=self.SCHEMA_VERSION, seed=self.seed,
                    library_sha256=hashlib.sha256(json.dumps(
                        [[w, s.tolist(), g.tolist()] for w, s, g in self.library],
                        sort_keys=True).encode('utf-8')).hexdigest(),
                    accepted=self.accepted, candidate_attempts=self.candidate_attempts,
                    rejection_histogram=dict(self.rejection_histogram),
                    rng=self.rng.bit_generator.state, base=self.base.state_dict())

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ('schema_version', 'seed', 'library_sha256'):
            if state.get(key) != expected[key]:
                raise ValueError('blocked teacher sampler identity mismatch: ' + key)
        self.accepted = int(state['accepted'])
        self.candidate_attempts = int(state['candidate_attempts'])
        self.rejection_histogram = dict(state['rejection_histogram'])
        self.rng.bit_generator.state = state['rng']
        self.base.load_state_dict(state['base'])
        self.last = None
