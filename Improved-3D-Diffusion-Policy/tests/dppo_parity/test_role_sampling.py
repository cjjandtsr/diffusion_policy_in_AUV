"""ROLE-PROP + ADMISSION: episode task/role contract and start/goal admission.

Design §4.8 (EpisodeTaskSpec, role sampling gates) and §4.9 (7 admission
rules, AdmissionExhausted). Role sampling params mirror the frozen env
_ROLE_SAMPLING_PARAMS; A* reachability is injected via a planner callable
(reuse of kinematic_uav_env._astar_3d wiring happens at integration, not here).
"""

import json
import math
from typing import Callable, Dict, Optional

import numpy as np
import pytest

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    AdmissionChecker,
    AdmissionExhausted,
    AdmissionResult,
    DEFAULT_ROLE_TARGETS,
    EpisodeTaskSpec,
    RoleSampler,
    StepBatch,
    TaskSpecRegistry,
    UAVVectorEnvAdapter,
    manifest_record_key,
)
from diffusion_policy_3d.env.collision_geom import (
    ObstacleSpec,
    compute_obstacle_clearance,
)

D_A = 11
ROLE_NAMES = list(DEFAULT_ROLE_TARGETS)


def _obstacles() -> list:
    """One distant cylinder so clearance checks are dominated by the drone."""
    return [
        ObstacleSpec(
            name="far_cyl", shape="cylinder", x=40.0, y=40.0, z=0.5,
            radius=1.0, height=2.0,
        ),
    ]


def _planner_reachable(start, goal):
    """Stub planner: always reachable with stretch ~1.0."""
    return True, 10.0


# ---------- EpisodeTaskSpec ----------

def test_episode_task_spec_fields():
    spec = EpisodeTaskSpec(
        world_uid="world_001",
        obstacle_json="obstacles_world_001.json",
        role="anticipatory",
        split="rl_train",
        admission_id="adm_001",
        generator_version="routeA_rl_v1.3",
        manifest_entry_hash="deadbeef",
    )
    assert spec.world_uid == "world_001"
    assert spec.admission_id == "adm_001"
    assert spec.split == "rl_train"


def test_episode_task_spec_frozen():
    spec = EpisodeTaskSpec(
        world_uid="w1", obstacle_json="o.json", role="anticipatory",
        split="rl_train", admission_id="a1", generator_version="v1",
        manifest_entry_hash="h",
    )
    with pytest.raises(Exception):
        spec.role = "dense_recovery"  # type: ignore[misc]


def test_episode_task_spec_rejects_unknown_role():
    with pytest.raises(ValueError):
        EpisodeTaskSpec(
            world_uid="w1", obstacle_json="o.json", role="hover",
            split="rl_train", admission_id="a1", generator_version="v1",
            manifest_entry_hash="h",
        )


def test_episode_task_spec_rejects_unknown_split():
    with pytest.raises(ValueError):
        EpisodeTaskSpec(
            world_uid="w1", obstacle_json="o.json", role="anticipatory",
            split="test_split", admission_id="a1", generator_version="v1",
            manifest_entry_hash="h",
        )


def test_same_world_multiple_roles_make_distinct_specs():
    """§4.8 [JUDGMENT]: role is not 1:1 with world; one world may serve
    multiple roles as distinct EpisodeTaskSpec entries."""
    entries = [
        {"world_uid": "wA", "role": "anticipatory", "split": "rl_train",
         "manifest_entry_hash": "hashA"},
        {"world_uid": "wA", "role": "dense_recovery", "split": "rl_train",
         "manifest_entry_hash": "hashB"},
    ]
    specs = TaskSpecRegistry(entries).train_specs()
    assert len(specs) == 2
    assert specs[0].world_uid == specs[1].world_uid == "wA"
    assert specs[0].role != specs[1].role
    assert specs[0].admission_id != specs[1].admission_id


# ---------- role sampling gates ----------

def test_role_sampler_10k_resets_within_2pp():
    """§4.8: 10k dry resets, each main-role ratio within target +/- 2pp."""
    rng = np.random.RandomState(42)
    sampler = RoleSampler(targets=DEFAULT_ROLE_TARGETS, rng=rng)
    counts = {r: 0 for r in ROLE_NAMES}
    for _ in range(10000):
        counts[sampler.sample()] += 1
    total = sum(counts.values())
    assert total == 10000
    for role, target in DEFAULT_ROLE_TARGETS.items():
        actual = counts[role] / total
        assert abs(actual - target) <= 0.02, (
            f"{role}: actual {actual:.4f} vs target {target:.4f}"
        )


def test_role_sampler_proportion_report_and_tolerance():
    rng = np.random.RandomState(7)
    sampler = RoleSampler(targets=DEFAULT_ROLE_TARGETS, rng=rng)
    counts = {r: 0 for r in ROLE_NAMES}
    for _ in range(10000):
        counts[sampler.sample()] += 1
    report = sampler.proportion_report(counts)
    assert RoleSampler.within_tolerance(
        report["anticipatory"], DEFAULT_ROLE_TARGETS["anticipatory"]
    )
    assert not RoleSampler.within_tolerance(0.80, DEFAULT_ROLE_TARGETS["anticipatory"])


def test_eval_only_worlds_never_in_train_manifest():
    """§4.8: eval-only worlds never appear in the train manifest. R7-8
    ruling 2026-08-24: the manifest carries three literal splits
    (rl_train / rl_val / rl_test); the eval side is rl_val ∪ rl_test."""
    entries = [
        {"world_uid": "train_A", "role": "anticipatory", "split": "rl_train",
         "manifest_entry_hash": "h1"},
        {"world_uid": "eval_B", "role": "anticipatory", "split": "rl_val",
         "manifest_entry_hash": "h2"},
        {"world_uid": "eval_C", "role": "support_expansion", "split": "rl_test",
         "manifest_entry_hash": "h3"},
    ]
    reg = TaskSpecRegistry(entries)
    train = reg.train_specs()
    assert {s.world_uid for s in train} == {"train_A"}
    assert reg.eval_specs() and {s.world_uid for s in reg.eval_specs()} == {
        "eval_B",
        "eval_C",
    }


# ---------- admission 7 rules ----------

def _make_checker(
    planner: Optional[Callable] = None,
    manifest_records: Optional[Dict[str, str]] = None,
) -> AdmissionChecker:
    return AdmissionChecker(
        contract=RLSystemContract(),
        role_params={
            "anticipatory": {"d_xy": (10, 24), "start_cl": 0.60, "goal_cl": 2.50},
            "dense_recovery": {"d_xy": (5, 18), "start_cl": 0.60, "goal_cl": 1.50},
            "support_expansion": {"d_xy": (8, 26), "start_cl": 0.80, "goal_cl": 2.00},
            "yaw_lateral_recovery": {"d_xy": (6, 18), "start_cl": 0.80, "goal_cl": 2.00},
        },
        obstacles=_obstacles(),
        planner=planner or _planner_reachable,
        manifest_records=manifest_records,
    )


def _start_goal(d_xy=15.0, dz=0.0):
    start = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    goal = np.array([d_xy, 0.0, 2.0 + dz], dtype=np.float32)
    return start, goal


def test_admission_rule1_start_clearance():
    checker = _make_checker()
    start = np.array([0.0, 0.0, 2.0], dtype=np.float32)  # far from obstacles
    goal = np.array([15.0, 0.0, 2.0], dtype=np.float32)
    res = checker.check(start, goal, role="anticipatory")
    assert res.pass_ is True


def test_admission_rule1_2_clearance_thresholds_reject():
    checker = _make_checker()
    # start/goal adjacent to the cylinder at (40,40): clearance ~= 0
    start = np.array([39.0, 40.0, 2.0], dtype=np.float32)
    goal = np.array([41.0, 40.0, 2.0], dtype=np.float32)
    res = checker.check(start, goal, role="anticipatory")
    assert res.pass_ is False
    assert any("clearance" in r for r in res.reasons)


def test_admission_rule3_3d_distance_range():
    checker = _make_checker()
    start, goal = _start_goal(d_xy=5.0)  # below anticipatory d_xy min=10
    res = checker.check(start, goal, role="anticipatory")
    assert res.pass_ is False
    assert any("d_xy" in r for r in res.reasons)


def test_admission_rule4_astar_unreachable_rejects():
    def planner_never(start, goal):
        return False, 0.0

    checker = _make_checker(planner=planner_never)
    res = checker.check(*_start_goal(), role="anticipatory")
    assert res.pass_ is False
    assert any("reachable" in r for r in res.reasons)


def test_admission_rule5_path_stretch_only_for_dense_recovery():
    """v1 env semantics: stretch in [1.1, 1.8] checked for dense_recovery only."""

    def planner_stretch(start, goal, stretch=4.0):
        return True, stretch * 15.0

    checker = _make_checker(planner=planner_stretch)
    res = checker.check(*_start_goal(), role="dense_recovery")
    assert res.pass_ is False
    assert any("stretch" in r for r in res.reasons)
    # same planner, anticipatory: stretch not checked
    res2 = checker.check(*_start_goal(), role="anticipatory")
    assert res2.pass_ is True


def test_admission_rule6_manifest_consistency():
    # Rule-6 records are keyed by the composite (world_uid, role,
    # admission_id) — a world_uid-only record no longer matches anything.
    checker = _make_checker(manifest_records={
        manifest_record_key("world_001", "anticipatory", "a1"): "deadbeef",
    })
    spec = EpisodeTaskSpec(
        world_uid="world_001", obstacle_json="o.json", role="anticipatory",
        split="rl_train", admission_id="a1", generator_version="v1",
        manifest_entry_hash="cafebabe",  # mismatch
    )
    res = checker.check(*_start_goal(), role="anticipatory", spec=spec)
    assert res.pass_ is False
    assert any("manifest" in r for r in res.reasons)
    good = EpisodeTaskSpec(
        world_uid="world_001", obstacle_json="o.json", role="anticipatory",
        split="rl_train", admission_id="a1", generator_version="v1",
        manifest_entry_hash="deadbeef",
    )
    res2 = checker.check(*_start_goal(), role="anticipatory", spec=good)
    assert res2.pass_ is True


def test_admission_rule7_fifty_failures_raise_exhausted():
    """§4.9 [JUDGMENT]: 50 failed samples -> AdmissionExhausted, no fixed
    unverified-coordinate fallback."""

    def planner_never(start, goal):
        return False, 0.0

    checker = _make_checker(planner=planner_never)
    with pytest.raises(AdmissionExhausted):
        for _ in range(50):
            res = checker.check(*_start_goal(), role="anticipatory")
            if res.pass_:
                break
        else:
            checker.raise_if_exhausted(50)


def test_registry_requires_manifest_entry_hash():
    """ASSUMPTION-6 pin: registry entries carry a non-empty
    manifest_entry_hash (the MANIFEST SOURCE hash — R2.3 split: the sampled
    start/goal/role are covered by episode_config_hash at reset; an empty
    source hash would spuriously fail rule 6 against a populated manifest)."""
    with pytest.raises(ValueError):
        TaskSpecRegistry([
            {"world_uid": "w1", "role": "anticipatory", "split": "rl_train"},
        ])


def test_fallback_uses_same_admission_no_bypass():
    """§4.9 rule 7: fallback path must run the identical checker (no bypass).
    API contract: there is no check-free admission entry point."""
    import inspect

    checker = _make_checker()
    method_names = [n for n in dir(checker) if not n.startswith("_")]
    assert "admit" not in method_names  # no unchecked bypass entry
    assert "check" in method_names


# ---------- env adapter shape guards (§4.10) ----------

def _write_world(tmp_path, obstacles: list) -> str:
    """Minimal obstacles.json in the real schema (load_obstacles)."""
    path = tmp_path / "guard_world.obstacles.json"
    path.write_text(json.dumps({"world_id": "guard_world",
                                "obstacles": obstacles}))
    return str(path)


def test_step_actions_shape_guard(tmp_path):
    """(n_envs, exec_steps, 11) enforced at the adapter boundary — the
    (n_envs, 16, 11) full-trajectory mis-route is type-shape blocked.
    (The valid-shape path is covered end-to-end by test_real_adapter.py —
    this test anchors the guard only.)"""
    contract = RLSystemContract()
    adapter = UAVVectorEnvAdapter(
        contract, world_pool=[("guard_world", _write_world(tmp_path, []))]
    )
    with pytest.raises(AssertionError):
        adapter.step(np.zeros((2, contract.full_horizon, D_A), dtype=np.float32))
    with pytest.raises(AssertionError):
        adapter.step(np.zeros((2, contract.exec_steps, 10), dtype=np.float32))


def test_step_batch_shapes():
    sb = StepBatch(
        obs={"point_cloud": np.zeros((2, 3, 4096, 3), dtype=np.float32)},
        reward=np.zeros(2, dtype=np.float32),
        terminated=np.zeros(2, dtype=bool),
        truncated=np.zeros(2, dtype=bool),
        info=[{} for _ in range(2)],
    )
    assert sb.reward.shape == (2,)
    assert sb.terminated.dtype == bool
