"""Legacy lightweight paired-bank API; formal endpoint uses the canonical script core."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.task_spec import (
    EVAL_ONLY_WORLD_IDS,
)


@dataclass(frozen=True)
class PairedEpisodeRecord:
    task_id: str
    world_id: str
    role: str
    seed: int
    policy: str
    success: bool
    collision: bool
    episode_return: float
    avoidance_onset_distance: float
    support_cells: int
    steps: int
    evaluator: str = "lightweight_kinematic"
    goal_xyz: tuple = ()
    final_position_xyz: tuple = ()
    final_distance_3d: float = float("nan")
    final_error_xy: float = float("nan")
    final_error_z: float = float("nan")
    terminal_safety_flags: tuple = ()
    observation_validity_summary: tuple = ()
    done_reason: str = "unknown"

    def __post_init__(self):
        if self.policy not in ("bc", "rl"):
            raise ValueError("policy must be bc or rl")
        if self.seed < 0 or self.steps <= 0:
            raise ValueError("invalid seed/steps")
        if self.success and self.collision:
            raise ValueError("episode cannot be both success and collision")
        if self.done_reason == "reached" and self.collision:
            raise ValueError("arrival and collision on the same transition is failure")
        if bool(self.goal_xyz) != bool(self.final_position_xyz):
            raise ValueError("goal and final position must be reported together")
        if self.goal_xyz:
            if len(self.goal_xyz) != 3 or len(self.final_position_xyz) != 3:
                raise ValueError("goal/final position must be three-dimensional")
            delta = tuple(float(a) - float(b) for a, b in zip(self.final_position_xyz, self.goal_xyz))
            distance = math.sqrt(sum(value * value for value in delta))
            xy = math.hypot(delta[0], delta[1])
            z = abs(delta[2])
            for name, observed, expected in (
                ("final_distance_3d", self.final_distance_3d, distance),
                ("final_error_xy", self.final_error_xy, xy),
                ("final_error_z", self.final_error_z, z),
            ):
                if not math.isfinite(observed) or not math.isclose(observed, expected, abs_tol=1e-6):
                    raise ValueError("{} does not match endpoint geometry".format(name))
            unsafe = self.collision or any(bool(value) for value in self.terminal_safety_flags)
            if bool(self.success) != (distance <= 0.5 and not unsafe):
                raise ValueError("success must mean safe three-dimensional arrival within 0.5 m")


class PairedEvaluationBank:
    def __init__(self, records):
        self.records = tuple(records)
        groups = {}
        for record in self.records:
            key = (record.task_id, record.seed)
            groups.setdefault(key, {})[record.policy] = record
        incomplete = [key for key, value in groups.items() if set(value) != {"bc", "rl"}]
        if incomplete:
            raise ValueError("unpaired evaluation records: {}".format(incomplete[:5]))
        self.pairs = tuple((value["bc"], value["rl"]) for _, value in sorted(groups.items()))

    @property
    def content_hash(self):
        payload = [asdict(record) for record in self.records]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def assert_not_training_source(self):
        # These historical worlds are valid paired regression/evaluation data,
        # but this object has no replay export API by design.
        return all(
            record.world_id in EVAL_ONLY_WORLD_IDS or record.evaluator
            for record in self.records
        )
