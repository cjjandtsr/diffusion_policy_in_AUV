"""P8 (B3, T2-ext): UAVVectorEnvAdapter real-env wiring — RED tests.

The adapter is the T2-ext boundary between the A-Official-Core train_loop
(duck-typed MockUAVEnv contract: reset -> (batched obs, info list),
step((1, A_e, 11)) -> 5-tuple, cam_far property, current_episode_config())
and the frozen kinematic_uav_env_v2.py gym env (episode_config reset path,
4-step execution, 3-frame->8-frame obs window).

Contract anchors:
- obs is batch-prefixed (1, T_h=8, 4096, 3) / (1, 8, 9) — train_loop indexes
  obs["point_cloud"][0, -1];
- step returns the designed StepBatch, iterable in the train_loop 5-tuple
  order (obs, reward, terminated, truncated, info);
- hit_count is derived per frame (points NOT at the camera far plane —
  synth_pointcloud.py:489-500 placeholder semantics);
- role-aware start/goal admission with A* reachability (AdmissionChecker,
  rules 1-6); 50 failed attempts raise AdmissionExhausted (rule 7 — no
  fixed-coordinate fallback);
- _collision_detected forwarded from the inner env and CLEARED on reset
  (the frozen env never resets the flag — stale True would poison every
  subsequent episode's SR).
"""

import json

import numpy as np
import pytest

from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
    AdmissionExhausted,
    EpisodeTaskSpec,
    StepBatch,
    UAVVectorEnvAdapter,
)
from diffusion_policy_3d.rl.dppo_uav.pointcloud_gate import PointCloudValidityGate

CONTRACT = RLSystemContract()
CAM_FAR = 10.0  # synth_pointcloud DepthCameraIntrinsics.far (D435i clip far)


def _write_world(tmp_path, obstacles: list, name: str = "test_world") -> str:
    path = tmp_path / f"{name}.obstacles.json"
    path.write_text(json.dumps({"world_id": name, "obstacles": obstacles}))
    return str(path)


def _sparse_world(tmp_path) -> str:
    """Two small cylinders — admission has plenty of free pairs."""
    return _write_world(tmp_path, [
        {"name": "c1", "shape": "cylinder", "x": 4.0, "y": 0.0,
         "radius": 1.2, "height": 4.0},
        {"name": "c2", "shape": "cylinder", "x": -6.0, "y": 5.0,
         "radius": 0.9, "height": 3.0},
    ])


def _blocked_world(tmp_path) -> str:
    """Radius-30 cylinder covering the whole C2 box: every start/goal pair
    fails clearance + planner -> admission exhausts."""
    return _write_world(tmp_path, [
        {"name": "mega", "shape": "cylinder", "x": 0.0, "y": 0.0,
         "radius": 30.0, "height": 10.0},
    ])


def _adapter(tmp_path, world_json: str, **kwargs) -> UAVVectorEnvAdapter:
    return UAVVectorEnvAdapter(
        CONTRACT,
        world_pool=[("test_world", world_json)],
        device="cpu",
        max_episode_steps=30,
        **kwargs,
    )


def _spec(role: str = "anticipatory", world: str = "test_world") -> EpisodeTaskSpec:
    return EpisodeTaskSpec(
        world_uid=world,
        obstacle_json="unused.json",
        role=role,
        split="rl_train",
        admission_id=f"{world}__{role}__rl_train",
        generator_version="routeA_rl_v1.3",
        manifest_entry_hash="unit-test-hash",
    )


# ═══════════════════════════════════════════════════════════════

def test_cam_far_matches_synth_far_plane(tmp_path):
    """The gate's cam_far must equal the synth camera far plane (10.0)."""
    assert _adapter(tmp_path, _sparse_world(tmp_path)).cam_far == CAM_FAR


def test_reset_returns_batched_obs_and_info(tmp_path):
    """reset -> batched (1, 8, 4096, 3)/(1, 8, 9) obs + info list carrying
    world_uid/role/hit_count; the reset frame passes the validity gate."""
    adapter = _adapter(tmp_path, _sparse_world(tmp_path))
    obs, info = adapter.reset([_spec()], seeds=[42])
    assert obs["point_cloud"].shape == (1, 8, 4096, 3)
    assert obs["agent_pos"].shape == (1, 8, 9)
    assert isinstance(info, list) and len(info) == 1
    assert info[0]["world_uid"] == "test_world"
    assert info[0]["role"] == "anticipatory"
    assert info[0]["hit_count"] > 0
    gate = PointCloudValidityGate(CONTRACT, phase="formal")
    frame = gate.validate(obs["point_cloud"][0, -1], info[0]["hit_count"], CAM_FAR)
    assert frame.valid, frame.invalid_reason


def test_step_returns_iterable_stepbatch(tmp_path):
    """step((1, 4, 11)) -> StepBatch, unpackable in the train_loop 5-tuple
    order (obs, reward, terminated, truncated, info)."""
    adapter = _adapter(tmp_path, _sparse_world(tmp_path))
    adapter.reset([_spec()], seeds=[42])
    batch = adapter.step(np.zeros((1, 4, 11), dtype=np.float32))
    assert isinstance(batch, StepBatch)
    obs, reward, terminated, truncated, info = batch
    assert obs["point_cloud"].shape == (1, 8, 4096, 3)
    # StepBatch fields are (n_envs,) arrays, info carries env_id
    assert reward.shape == (1,) and terminated.shape == (1,)
    assert truncated.shape == (1,)
    assert not bool(terminated[0]) and not bool(truncated[0])
    assert info[0]["env_id"] == 0
    assert info[0]["world_uid"] == "test_world"
    assert info[0]["hit_count"] > 0


def test_step_asserts_action_shape(tmp_path):
    """exec_steps and action_dim are shape-blocked at the boundary (§4.10)."""
    adapter = _adapter(tmp_path, _sparse_world(tmp_path))
    adapter.reset([_spec()], seeds=[42])
    with pytest.raises(AssertionError, match="exec_steps"):
        adapter.step(np.zeros((1, 5, 11), dtype=np.float32))
    with pytest.raises(AssertionError, match="action dim"):
        adapter.step(np.zeros((1, 4, 10), dtype=np.float32))


def test_admission_rules_satisfied(tmp_path):
    """Sampled start/goal honor the role's rules: d_xy in (10, 24), clearances
    >= 0.60/2.50, A* reachable on the adapter's own occupancy grid."""
    adapter = _adapter(tmp_path, _sparse_world(tmp_path))
    adapter.reset([_spec()], seeds=[42])
    cfg = adapter.current_episode_config()
    start = np.asarray(cfg["start"], dtype=np.float32)
    goal = np.asarray(cfg["goal"], dtype=np.float32)
    d_xy = float(np.hypot(goal[0] - start[0], goal[1] - start[1]))
    assert 10.0 <= d_xy <= 24.0

    from diffusion_policy_3d.env.collision_geom import compute_obstacle_clearance
    from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
        ROLE_ADMISSION_PARAMS,
    )
    from diffusion_policy_3d.rl.dppo_uav.planner import (
        astar_3d,
        build_occupancy_grid,
    )

    obstacles = adapter._env.obstacles  # world switched to test_world
    grid, meta = build_occupancy_grid(obstacles)
    for pt, req in ((start, ROLE_ADMISSION_PARAMS["anticipatory"]["start_cl"]),
                    (goal, ROLE_ADMISSION_PARAMS["anticipatory"]["goal_cl"])):
        cl, _, _ = compute_obstacle_clearance(float(pt[0]), float(pt[1]),
                                              float(pt[2]), obstacles)
        assert cl >= req, f"clearance {cl:.2f} < {req}"
    ok, _, _, _, _, _ = astar_3d(grid, meta, *start, *goal)
    assert ok, "admitted pair must be A* reachable"


def test_admission_exhausted_raises(tmp_path):
    """Fully blocked world -> 50 failed attempts raise AdmissionExhausted
    (rule 7: fail structurally, never fall back to fixed coordinates)."""
    adapter = _adapter(tmp_path, _blocked_world(tmp_path))
    with pytest.raises(AdmissionExhausted):
        adapter.reset([_spec()], seeds=[42])


def test_unknown_world_raises(tmp_path):
    """A spec whose world is not in the adapter's pool is a hard error."""
    adapter = _adapter(tmp_path, _sparse_world(tmp_path))
    with pytest.raises(ValueError, match="not in pool"):
        adapter.reset([_spec(world="ghost_world")], seeds=[42])


def test_collision_flag_cleared_on_reset(tmp_path):
    """Drive into the cylinder -> terminated + _collision_detected True; the
    NEXT reset clears the flag (the frozen env never resets it)."""
    adapter = _adapter(tmp_path, _sparse_world(tmp_path))
    adapter.reset([_spec()], seeds=[42])
    env = adapter._env
    # place the drone 0.15 m from the cylinder surface, pushing into it
    env._drone_pos = np.array([2.65, 0.0, 2.0], dtype=np.float32)
    env._drone_yaw = 0.0
    env._drone_vel = np.zeros(3, dtype=np.float32)
    _, reward, terminated, truncated, info = adapter.step(
        np.full((1, 4, 11), 0.5, dtype=np.float32)
    )
    assert bool(terminated[0]), "pushing into the cylinder must terminate"
    assert adapter._collision_detected is True
    assert info[0]["done_reason"] != "none"

    adapter.reset([_spec()], seeds=[43])
    assert adapter._collision_detected is False


def test_current_episode_config_roundtrip(tmp_path):
    """current_episode_config returns {world, role, seed, start, goal}; the
    start matches the inner env's actual drone position after reset."""
    adapter = _adapter(tmp_path, _sparse_world(tmp_path))
    adapter.reset([_spec()], seeds=[7])
    cfg = adapter.current_episode_config()
    assert cfg["world"] == "test_world"
    assert cfg["role"] == "anticipatory"
    assert cfg["seed"] == 7
    inner_pos = adapter._env._drone_pos
    assert np.allclose(cfg["start"], inner_pos, atol=1e-5)


def test_eval_discipline_params_forwarded(tmp_path):
    """Constructor passes eval discipline (B4) through to the inner env:
    disturb_enable=False. R5.8-G (F-4): the mid_state_init_ratio knob is
    REMOVED from the adapter — it was dead (the env only consumed it in
    the non-episode_config branch, which production never runs)."""
    adapter = _adapter(tmp_path, _sparse_world(tmp_path),
                       disturb_enable=False)
    assert adapter._env.disturb_enable is False


def test_hit_count_derivation(tmp_path):
    """hit_count = POINTS (rows) NOT at the far-plane placeholder — the
    placeholder fills missed rays per row, so the count is per-point, never
    per-coordinate (3x): all-far -> 0; half-far -> 2048; a normal frame -> > 0."""
    adapter = _adapter(tmp_path, _sparse_world(tmp_path))
    far = np.full((4096, 3), CAM_FAR, dtype=np.float32)
    assert adapter._hit_count(far) == 0
    half = np.full((4096, 3), 0.5, dtype=np.float32)
    half[:2048] = CAM_FAR
    assert adapter._hit_count(half) == 2048, (
        "hit_count must count rows (points), not coordinates"
    )
    adapter.reset([_spec()], seeds=[42])
    adapter.abort_episode([0])  # reset_at refuses ACTIVE slots
    obs, info = adapter.reset([_spec()], seeds=[44])
    assert adapter._hit_count(obs["point_cloud"][0, -1]) == info[0]["hit_count"]
    assert info[0]["hit_count"] > 0


# ═══════════════════════════════════════════════════════════════
# Dual hash split — manifest source hash vs sampled episode hash
# ═══════════════════════════════════════════════════════════════

def test_canonical_sha256_key_order_and_float_fixed():
    """R2.3: canonical serialization is key-order independent, float-exact,
    and NaN-hard-failing (no default=str swallowing un-JSON data)."""
    from diffusion_policy_3d.rl.dppo_uav.environment_adapter import canonical_sha256

    a = canonical_sha256({"b": 1.5, "a": [1.0, 2.0]})
    b = canonical_sha256({"a": [1.0, 2.0], "b": 1.5})  # key order swapped
    assert a == b
    assert canonical_sha256({"x": 1.0}) != canonical_sha256({"x": 1}), (
        "float/int serialization must be fixed (json.dumps: 1.0 vs 1)"
    )
    with pytest.raises(ValueError):
        canonical_sha256({"x": float("nan")})


def test_episode_config_hash_sensitive_to_sampled_fields():
    """R2.3: under one manifest entry, changing seed/start/goal/yaw changes
    episode_config_hash; manifest_entry_hash stays bound to the entry.
    Derived admission metrics are NOT part of the hash (they are
    deterministic from start/goal/world)."""
    from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
        build_episode_config_record,
    )

    spec = _spec()
    start = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    goal = np.array([15.0, 2.0, 3.0], dtype=np.float32)

    base = build_episode_config_record(
        spec, seed=7, start=start, goal=goal, start_yaw=0.5,
        start_clearance=2.0, goal_clearance=3.0, reachability_pass=True,
        path_length=15.5, path_stretch=1.1,
    )
    assert base.manifest_entry_hash == "unit-test-hash"
    assert len(base.episode_config_hash) == 64

    moved = build_episode_config_record(  # start changed
        spec, seed=7, start=np.array([2.0, 2.0, 3.0], dtype=np.float32),
        goal=goal, start_yaw=0.5,
    )
    assert moved.episode_config_hash != base.episode_config_hash
    rerolled = build_episode_config_record(  # goal changed
        spec, seed=7, start=start, goal=np.array([16.0, 2.0, 3.0], dtype=np.float32),
        start_yaw=0.5,
    )
    assert rerolled.episode_config_hash != base.episode_config_hash
    rotated = build_episode_config_record(  # yaw changed
        spec, seed=7, start=start, goal=goal, start_yaw=-1.2,
    )
    assert rotated.episode_config_hash != base.episode_config_hash
    reseeded = build_episode_config_record(  # seed changed
        spec, seed=8, start=start, goal=goal, start_yaw=0.5,
    )
    assert reseeded.episode_config_hash != base.episode_config_hash
    # deterministic: same geometry, metrics irrelevant to the hash
    assert build_episode_config_record(
        spec, seed=7, start=start, goal=goal, start_yaw=0.5,
    ).episode_config_hash == base.episode_config_hash


def test_reset_info_and_config_carry_both_hashes(tmp_path):
    """R2.3: reset info + current_episode_config carry manifest_entry_hash
    AND episode_config_hash; the record's admission metrics come from the
    check that admitted the episode."""
    adapter = _adapter(tmp_path, _sparse_world(tmp_path))
    spec = _spec()
    obs, info = adapter.reset([spec], seeds=[7])
    assert info[0]["manifest_entry_hash"] == "unit-test-hash"
    cfg = adapter.current_episode_config()
    assert cfg["manifest_entry_hash"] == "unit-test-hash"
    assert cfg["episode_config_hash"] == info[0]["episode_config_hash"]
    assert isinstance(cfg["episode_config_hash"], str)
    assert cfg["split"] == "rl_train"
    assert cfg["admission_id"] == spec.admission_id
    # admission metrics from the real checker (sparse world admits)
    assert cfg["reachability_pass"] is True
    assert cfg["start_clearance"] > 0.0
    assert cfg["goal_clearance"] > 0.0
    assert cfg["path_length"] > 0.0
    # step info keeps carrying the same episode hashes
    _, _, _, _, sinfo = adapter.step(np.zeros((1, 4, 11), dtype=np.float32))
    assert sinfo[0]["episode_config_hash"] == cfg["episode_config_hash"]


def test_step_atomically_rejects_any_done_slot(tmp_path):
    """P2-7 (审核裁决): step() must reject BEFORE advancing ANY slot — a
    done slot at index 1 must not be discovered only after slot 0 already
    stepped (per-slot in-loop check was non-atomic).

    RED contract (before the rework): slot 0 stepped, THEN slot 1's done
    check raised — partial progress on a rejected call."""
    adapter = _adapter(tmp_path, _sparse_world(tmp_path), n_envs=2)
    adapter.reset_at([0], [_spec()], [42])  # slot 0 active; slot 1 done
    stepped = []
    original_step = adapter._envs[0].step
    adapter._envs[0].step = lambda a: (stepped.append(1), original_step(a))[1]
    with pytest.raises(RuntimeError, match="done"):
        adapter.step(np.zeros((2, 4, 11), dtype=np.float32))
    assert stepped == [], "slot 0's env must not step when any slot is done"


def test_reset_info_episode_id_is_unique_record_id(tmp_path):
    """P2-9 (审核裁决): reset info episode_id must be the unique
    EpisodeConfigRecord id (world:role:episode_seed) — the shared
    admission_id is not an episode id: two resets of the same spec
    previously returned the SAME episode_id (collision).

    RED contract (before the rework): info episode_id echoed the env's
    cfg entry, which was spec.admission_id."""
    adapter = _adapter(tmp_path, _sparse_world(tmp_path))
    spec = _spec()
    _, info1 = adapter.reset([spec], seeds=[100])
    record1 = adapter._records[0].episode_id
    adapter.abort_episode([0])  # reset_at refuses ACTIVE slots
    _, info2 = adapter.reset([spec], seeds=[101])
    assert info1[0]["episode_id"] == record1
    assert info1[0]["episode_id"] != spec.admission_id
    assert info2[0]["episode_id"] != info1[0]["episode_id"]
    assert info2[0]["episode_id"] == adapter._records[0].episode_id


def test_manifest_records_composite_key_no_overwrite():
    """R2.3: two roles sharing a world_uid keep separate manifest records —
    key = (world_uid, role, admission_id); role A's hash never admits role B."""
    from diffusion_policy_3d.rl.dppo_uav.environment_adapter import (
        AdmissionChecker,
        ROLE_ADMISSION_PARAMS,
        manifest_record_key,
    )

    def _planner(start, goal):
        return True, 15.0

    checker = AdmissionChecker(
        CONTRACT, ROLE_ADMISSION_PARAMS, [], _planner,
        manifest_records={
            manifest_record_key("shared_world", "anticipatory", "id_a"): "hash-a",
            manifest_record_key("shared_world", "dense_recovery", "id_b"): "hash-b",
        },
    )
    spec_a = EpisodeTaskSpec(
        world_uid="shared_world", obstacle_json="o.json", role="anticipatory",
        split="rl_train", admission_id="id_a", generator_version="v1",
        manifest_entry_hash="hash-a",
    )
    res = checker.check(
        np.array([0.0, 0.0, 2.0]), np.array([15.0, 0.0, 2.0]),
        role="anticipatory", spec=spec_a,
    )
    assert res.pass_ is True
    # role B under role A's record: rule 6 rejects (no cross-overwrite)
    spec_b_wrong = EpisodeTaskSpec(
        world_uid="shared_world", obstacle_json="o.json", role="dense_recovery",
        split="rl_train", admission_id="id_b", generator_version="v1",
        manifest_entry_hash="hash-a",
    )
    res = checker.check(
        np.array([0.0, 0.0, 2.0]), np.array([10.0, 0.0, 2.0]),
        role="dense_recovery", spec=spec_b_wrong,
    )
    assert res.pass_ is False
    assert any("manifest" in r for r in res.reasons)
    # role B under its own record: passes
    spec_b_ok = EpisodeTaskSpec(
        world_uid="shared_world", obstacle_json="o.json", role="dense_recovery",
        split="rl_train", admission_id="id_b", generator_version="v1",
        manifest_entry_hash="hash-b",
    )
    res = checker.check(
        np.array([0.0, 0.0, 2.0]), np.array([10.0, 0.0, 2.0]),
        role="dense_recovery", spec=spec_b_ok,
    )
    assert res.pass_ is True


@pytest.mark.gpu
def test_step_accepts_cuda_tensor_actions(tmp_path):
    """G5 RED: the loop hands the adapter whatever the E040 normalizer
    returns — a CUDA torch tensor — and np.asarray(cuda) raises TypeError
    (environment_adapter.py:477). The boundary must convert tensor -> cpu
    numpy; the kinematic env physics is CPU-numpy."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("gpu marker: CUDA unavailable")
    adapter = _adapter(tmp_path, _sparse_world(tmp_path))
    adapter.reset([_spec()], seeds=[0])
    actions = torch.zeros((1, 4, 11), device="cuda:0")
    batch = adapter.step(actions)
    assert batch.obs["point_cloud"].shape[-2:] == (4096, 3)
