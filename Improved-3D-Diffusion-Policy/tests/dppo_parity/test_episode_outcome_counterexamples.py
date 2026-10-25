"""R5.8-C C8 + R6-E §11.6.4 (F-10): EpisodeOutcome counterexamples on
the REAL kinematic env.

The four ruling-mandated counterexamples (N5/P1-2/F-2/F-3) driven on the
REAL production env (kinematic_uav_env_v2 + real world obstacles +
real SynthPointCloud) with the paired-eval episode_config reset path —
the same objects formal training and eval use, no mock env, no
placeholder geometry. White-box state control (vel/pos) is used only to
reach the exact edge conditions a learned policy cannot reliably hit.

R6-E adds: env-level terminated/truncated CO-FIRE counterexamples
(reached+truncated / collision+truncated at the max_steps=1 boundary,
pure timeout) plus the mutually exclusive terminal_category mapper unit
tests — the category re-derives from the raw flags, and any state that
cannot be mapped uniquely hard-fails (§11.6.4, no priority-silent
reclassification). reached_collision exists ONLY at the mapper level:
the EpisodeOutcome invariant forbids success∧collision in the env.

Run: pytest -q tests/dppo_parity/test_episode_outcome_counterexamples.py
"""

import pytest
import numpy as np

from diffusion_policy_3d.env.episode_outcome import (
    EpisodeOutcome,
    terminal_category,
)
from diffusion_policy_3d.env.kinematic_uav_env_v2 import KinematicUAVEnv
from diffusion_policy_3d.model.common.synth_pointcloud import SynthPointCloud

WORLDS_DIR = "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train"
WORLD = ("routeA_rl_v1__anticipatory__rl_train__"
         "double_gate_progressive__000001")
# gate1_l from the real obstacles json (world frame, no spawn offset)
GATE1_L_CENTER = (-3.0, -1.75, 2.5)
GATE1_L_HALF = 0.25  # size_x/2 = size_y/2
DRONE_RADIUS = 0.35  # collision_thresh


def _make_env(goal, start, max_steps=6):
    """Real env: world_pool + episode_config reset (the paired-eval path)."""
    env = KinematicUAVEnv(
        world_id=0,
        goal_world=np.asarray(goal, dtype=np.float32),
        reward_version="v16",
        disturb_enable=False,
        exec_noise_scale=0.0,
        max_episode_steps=max_steps,
        mid_state_init_ratio=0.0,
        curriculum_stage="C2",
        world_pool=[(WORLD, f"{WORLDS_DIR}/{WORLD}.world.obstacles.json")],
        synth_pc_factory=lambda jp: SynthPointCloud(jp, device="cpu"),
    )
    env.reset(seed=0, options={
        "episode_config": {
            "world_name": WORLD,
            "goal_pos": [float(v) for v in goal],
            "start_pos": [float(v) for v in start],
            "start_yaw": 0.0,
            "disturb_events": [],
        },
    })
    return env


def _action(dx=0.0, vx=0.0):
    """(8, 11) zero action with optional x-delta and x-velocity."""
    a = np.zeros((8, 11), dtype=np.float32)
    a[:, 0] = dx   # DPOS_W x
    a[:, 3] = vx   # VEL_W x
    return a


def _outcome(info):
    return EpisodeOutcome(**info["outcome"])


def test_high_speed_pass_through_goal_is_not_success():
    """N5 counterexample 1: a high-speed pass through the goal sphere
    must NOT terminate, must NOT earn the reach bonus, must NOT be
    success (the env's velocity gate v_xy < 1.0 is authoritative)."""
    goal = np.asarray([8.0, 0.0, 2.0], dtype=np.float32)
    start = goal - np.asarray([0.4, 0.0, 0.0], dtype=np.float32)  # dist 0.4
    env = _make_env(goal, start)
    _, reward, terminated, truncated, info = env.step(_action(vx=10.0))
    assert not terminated, "high-speed pass must not terminate"
    assert not truncated
    out = _outcome(info)
    assert out.done_reason == "none", out.done_reason
    assert not out.success, "no velocity gate -> false success"
    assert reward < 50.0, (
        f"high-speed pass must not earn the +100 reach bonus, got {reward}"
    )


def test_slow_reach_is_success_with_terminal_bonus():
    """N5 counterexample 2: a slow legal arrival terminates as success
    with exactly one reach bonus (terminal-gated by construction)."""
    goal = np.asarray([8.0, 0.0, 2.0], dtype=np.float32)
    start = goal - np.asarray([0.3, 0.0, 0.0], dtype=np.float32)  # dist 0.3
    env = _make_env(goal, start)
    _, reward, terminated, truncated, info = env.step(_action())
    assert terminated
    out = _outcome(info)
    assert out.done_reason == "reached", out.done_reason
    assert out.success
    assert abs(reward - 100.0) < 1e-2, (
        f"slow reach = progress 0 + r_reach 100, got {reward}"
    )


def test_collision_and_goal_cofire_is_collision_no_reach_bonus():
    """N5 counterexample 3: arriving AT the goal that sits on an
    obstacle (real gate1_l geometry) co-fires collision+reached — the
    outcome must be collision, success False, and the reach bonus must
    NOT be paid."""
    goal = np.asarray(GATE1_L_CENTER, dtype=np.float32)
    start = goal - np.asarray([0.6, 0.0, 0.0], dtype=np.float32)
    env = _make_env(goal, start)
    # four x-deltas of 0.15 land the drone exactly at the gate center
    _, reward, terminated, truncated, info = env.step(_action(dx=0.15))
    assert terminated, "landing on an obstacle must terminate"
    out = _outcome(info)
    assert out.collision, out
    assert not out.success, "collision+goal must never be success"
    assert out.done_reason == "collision", out.done_reason
    assert reward < -15.0, (
        f"collision+goal pays r_collision without r_reach, got {reward}"
    )


def test_timeout_outcome_and_no_stale_done_reason():
    """F-3 + timeout semantics: a timeout episode reports done_reason
    "timeout" — and a timeout AFTER a collision episode must never
    report the previous episode's stale "collision"."""
    goal = np.asarray([8.0, 0.0, 2.0], dtype=np.float32)
    env = _make_env(goal, np.asarray([-3.0, -1.75, 2.5]), max_steps=3)

    # episode 1: collision at the goal-on-obstacle point
    env.reset(seed=1, options={
        "episode_config": {
            "world_name": WORLD,
            "goal_pos": list(GATE1_L_CENTER),
            "start_pos": [GATE1_L_CENTER[0] - 0.6, GATE1_L_CENTER[1],
                          GATE1_L_CENTER[2]],
            "start_yaw": 0.0,
            "disturb_events": [],
        },
    })
    _, _, terminated, _, info = env.step(_action(dx=0.15))
    assert terminated
    assert _outcome(info).done_reason == "collision"

    # episode 2 (same env, reset): far start, runs to the time limit
    env.reset(seed=2, options={
        "episode_config": {
            "world_name": WORLD,
            "goal_pos": [8.0, 0.0, 2.0],
            "start_pos": [0.0, 0.0, 2.0],
            "start_yaw": 0.0,
            "disturb_events": [],
        },
    })
    last = None
    for _ in range(3):
        _, _, terminated, truncated, info = env.step(_action())
        last = info
        if terminated:
            break
    out = _outcome(last)
    assert out.time_limit, out
    assert out.done_reason == "timeout", (
        f"stale done_reason leaked across episodes: {out.done_reason}"
    )
    assert not out.success and not out.collision


# ═══════════════════════════════════════════════════════════════════
# R6-E §11.6.4 (F-10): env 级 terminated/truncated co-fire 反例
# — raw 标志同帧共发时互斥 terminal_category 仍唯一映射
# ═══════════════════════════════════════════════════════════════════

def test_reached_plus_truncated_cofire_outcome():
    """§11.6.4: reached 与 time limit 在同一帧共发（max_steps=1 边界）
    时 outcome 同时携带 terminated=True 与 truncated=True，done_reason
    保持 "reached"（env :542-543 只在 reason=="none" 时改写为 timeout）
    — terminal_category 唯一映射 "reached"，绝不是 "timeout"。"""
    goal = np.asarray([8.0, 0.0, 2.0], dtype=np.float32)
    start = goal - np.asarray([0.3, 0.0, 0.0], dtype=np.float32)
    env = _make_env(goal, start, max_steps=1)
    _, _, terminated, truncated, info = env.step(_action())
    assert terminated and truncated, "reached 与 time limit 必须同帧共发"
    out = _outcome(info)
    assert out.done_reason == "reached", out.done_reason
    assert out.success
    assert out.terminal_category == "reached"


def test_collision_plus_truncated_cofire_outcome():
    """collision 与 time limit 同帧共发（max_steps=1，一步落到 gate
    中心）→ done_reason="collision"（collision 折叠优先），category
    "collision" — 共发不产生 timeout 类别。"""
    goal = np.asarray(GATE1_L_CENTER, dtype=np.float32)
    start = goal - np.asarray([0.15, 0.0, 0.0], dtype=np.float32)
    env = _make_env(goal, start, max_steps=1)
    _, _, terminated, truncated, info = env.step(_action(dx=0.15))
    assert terminated and truncated, "collision 与 time limit 必须同帧共发"
    out = _outcome(info)
    assert out.done_reason == "collision", out.done_reason
    assert out.collision and not out.success
    assert out.terminal_category == "collision"


def test_pure_timeout_env_outcome_category():
    """纯 timeout（远起点跑满 max_steps）→ truncated=True、
    terminated=False、done_reason="timeout" → category "timeout"。"""
    goal = np.asarray([8.0, 0.0, 2.0], dtype=np.float32)
    env = _make_env(goal, np.asarray([0.0, 0.0, 2.0], dtype=np.float32),
                    max_steps=1)
    _, _, terminated, truncated, info = env.step(_action())
    assert not terminated and truncated
    out = _outcome(info)
    assert out.time_limit
    assert out.done_reason == "timeout", out.done_reason
    assert out.terminal_category == "timeout"


# ───────────────────────────────────────────────────────────────────
# §11.6.4 terminal_category mapper — 互斥唯一映射 + 矛盾 hard-fail
# ───────────────────────────────────────────────────────────────────

def test_mapper_reached_collision_only_exists_at_mapper_level():
    """success+collision → "reached_collision"。EpisodeOutcome 不变量
    禁止 success∧collision 在 env 层构造，因此该类别只能存在于 mapper
    （artifact/报告）层 — raw success/collision 计数两者都保留。"""
    assert terminal_category(
        terminated=True, truncated=False, success=True, collision=True,
        done_reason="collision", out_of_bounds=False, altitude_exceed=False,
    ) == "reached_collision"


@pytest.mark.parametrize("kwargs, match", [
    # collision=True 但 done_reason ∉ {collision, crash_ground}
    (dict(terminated=True, truncated=False, success=False, collision=True,
          done_reason="reached", out_of_bounds=False, altitude_exceed=False),
     "collision"),
    # success=True 且无 collision 但 done_reason != "reached"
    (dict(terminated=True, truncated=False, success=True, collision=False,
          done_reason="collision", out_of_bounds=False, altitude_exceed=False),
     "success"),
    # terminated=True 但 done_reason == "none"
    (dict(terminated=True, truncated=False, success=False, collision=False,
          done_reason="none", out_of_bounds=False, altitude_exceed=False),
     "terminated"),
    # truncated=True 但 done_reason == "none"
    (dict(terminated=False, truncated=True, success=False, collision=False,
          done_reason="none", out_of_bounds=False, altitude_exceed=False),
     "truncated"),
])
def test_mapper_contradictions_hard_fail(kwargs, match):
    """§11.6.4: 四类矛盾状态（collision/reached/terminated/truncated 与
    done_reason 冲突）→ ValueError — 无优先级静默重分类。"""
    with pytest.raises(ValueError, match=match):
        terminal_category(**kwargs)


def test_mapper_unmappable_state_hard_fails():
    """terminated=True 但 success=False ∧ collision=False ∧
    done_reason="reached" — 无环境能产出该状态（reached ⇒ success），
    无唯一类别可映射 → hard-fail（§11.6.4 禁优先级静默）。"""
    with pytest.raises(ValueError):
        terminal_category(
            terminated=True, truncated=False, success=False, collision=False,
            done_reason="reached", out_of_bounds=False, altitude_exceed=False,
        )


def test_mapper_timeout_only_from_done_reason():
    """timeout 类别仅在 done_reason=="timeout" 时出现；reached+truncated
    共发映射 "reached"；done_reason=="timeout" 而无 truncated 是语义
    矛盾（env 中 timeout ⇒ truncated）→ hard-fail。"""
    assert terminal_category(
        terminated=False, truncated=True, success=False, collision=False,
        done_reason="timeout", out_of_bounds=False, altitude_exceed=False,
    ) == "timeout"
    assert terminal_category(
        terminated=True, truncated=True, success=True, collision=False,
        done_reason="reached", out_of_bounds=False, altitude_exceed=False,
    ) == "reached"
    with pytest.raises(ValueError, match="timeout"):
        terminal_category(
            terminated=False, truncated=False, success=False, collision=False,
            done_reason="timeout", out_of_bounds=False, altitude_exceed=False,
        )


@pytest.mark.parametrize("other", ["observation_fault", "rollout_cut",
                                   "other_terminated"])
def test_mapper_non_env_categories(other):
    """observation_fault / rollout_cut / other_terminated 非 env
    termination：干净状态（全 False + done_reason="none"）映射到自身；
    与任何 env termination 同现 → 矛盾 hard-fail。"""
    assert terminal_category(
        terminated=False, truncated=False, success=False, collision=False,
        done_reason="none", out_of_bounds=False, altitude_exceed=False,
        other=other,
    ) == other
    with pytest.raises(ValueError, match="other"):
        terminal_category(
            terminated=True, truncated=False, success=False, collision=False,
            done_reason="collision", out_of_bounds=False, altitude_exceed=False,
            other=other,
        )


def test_mapper_invalid_other_hard_fails():
    with pytest.raises(ValueError, match="other"):
        terminal_category(
            terminated=False, truncated=False, success=False, collision=False,
            done_reason="none", out_of_bounds=False, altitude_exceed=False,
            other="bogus",
        )


def test_episode_outcome_terminal_category_property():
    """EpisodeOutcome.terminal_category 走同一个 mapper — env 产出的
    outcome 直接给出互斥类别。reached_collision 在 outcome 层不可构造
    （success∧collision 被不变量拦截）。"""
    assert EpisodeOutcome(
        success=True, collision=False, time_limit=False,
        out_of_bounds=False, altitude_exceed=False,
        terminated=True, truncated=False, done_reason="reached",
    ).terminal_category == "reached"
    assert EpisodeOutcome(
        success=False, collision=True, time_limit=False,
        out_of_bounds=False, altitude_exceed=False,
        terminated=True, truncated=False, done_reason="collision",
    ).terminal_category == "collision"
    assert EpisodeOutcome(
        success=False, collision=False, time_limit=True,
        out_of_bounds=False, altitude_exceed=False,
        terminated=False, truncated=True, done_reason="timeout",
    ).terminal_category == "timeout"
