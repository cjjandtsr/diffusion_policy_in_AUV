"""external-review #3 (RULING-7) 强制正路径要求 — rework #3 positive matrix。

裁决要求用真实生产链重写正路径测试，禁止手写字典模拟 LoRA：
  ① 真实 insert_lora() 链：save → fresh model → 完整 restore_training_state()
     → LoRA 张量 bitwise 恢复 + optimizer state 恢复 + base UNet 不变
     + counters 返回。
  ④ env RNG 修复语义：字符串 world_id 直接进 canonical JSON → BLAKE2b
     （不再经进程随机 hash()）；env v1/v2 双路径各 pin 一个稳定 reference。
  ⑤ 两个 fresh process（PYTHONHASHSEED unset）同一 task 必须产出相同
     初始 obs（point_cloud sha256 一致）。

全部 TDD RED 先行：修复前每条都在 P0-1（key count vs n_adaptors）或
P0-2（hash 随机）处失败，修复后全绿。
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from diffusion_policy_3d.rl.dppo_uav.checkpoint import (
    CheckpointBuilder,
    SCHEMA_VERSION,
    build_lora_identity,
)
from diffusion_policy_3d.rl.dppo_uav.policy_loader import extract_lora_state
from diffusion_policy_3d.rl.dppo_uav.train_loop import (
    _actor_optimizer_param_names,
    restore_training_state,
)

from conftest import DEFAULT_BC_CKPT
from test_checkpoint import _runtime_identity
from test_r3_rework_negative import _good_structured, _restore_ready, _v3_data

REPO_ROOT = Path(__file__).resolve().parents[2]

# Route A RL v1 val world（真实资产，stable-UID 语义）
UID = "routeA_rl_v1__dense_recovery__rl_val__bottleneck_corridor__000100"
OBSTACLES = (
    "/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/val/"
    f"{UID}.world.obstacles.json"
)


def _episode_config() -> dict:
    """Deterministic episode_config for the stable-UID reference tests."""
    return {
        "world_name": UID,
        "goal_pos": [9.0, -9.0, 1.5],
        "start_pos": [-8.0, 4.0, 1.5],
        "start_yaw": 0.0,
        "disturb_events": [],
        "episode_id": "r3_rework3_ref",
    }


def _v2_env():
    from diffusion_policy_3d.env.kinematic_uav_env_v2 import KinematicUAVEnv
    from diffusion_policy_3d.model.common.synth_pointcloud import (
        SynthPointCloud,
    )

    return KinematicUAVEnv(
        world_pool=[(UID, OBSTACLES)],
        synth_pc_factory=lambda p: SynthPointCloud(p, device="cpu"),
        max_episode_steps=8,
    )


# ═══════════════════════════════════════════════════════════════════
# 裁决①: 真实生产链 save → fresh → restore 正路径（非手写 dict 模拟）
# ═══════════════════════════════════════════════════════════════════


def test_real_lora_checkpoint_save_restore_roundtrip(tmp_path, monkeypatch):
    """真实链全流程：BC checkpoint → insert_lora → 真实 optimizer step →
    extract/build identity → save → fresh model + fresh insert_lora →
    restore_training_state → 全部张量 bitwise 断言。

    RED 失败点（修复前）：validate_checkpoint_lora_usable 用 n_adaptors
    （模块数）比较真实 lora_state 的 2n 个 key → CheckpointError（P0-1）。
    """
    from diffusion_policy_3d.policy.lora_adapters import insert_lora
    from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
    from diffusion_policy_3d.rl.dppo_uav.critic import (
        PolicyObsEncoder,
        ValueCritic,
    )
    from diffusion_policy_3d.rl.dppo_uav.policy_loader import load_ddpm_policy

    monkeypatch.setenv("PYTHONHASHSEED", "7")
    ckpt = os.environ.get("DPPO_BC_CKPT", DEFAULT_BC_CKPT)
    if not os.path.isfile(ckpt):
        pytest.skip(f"BC checkpoint not found: {ckpt}")

    # ── 真实生产链（保存侧）──
    policy = load_ddpm_policy(ckpt, device="cpu")
    lora_info = insert_lora(policy.model, r=4, alpha=4, verbose=False)
    n_adaptors = lora_info["report"]["n_adaptors"]
    assert n_adaptors >= 1, "real BC policy must expose LoRA target modules"
    n_tensors = len(lora_info["lora_params"])
    assert n_tensors == 2 * n_adaptors, (
        "production LoRA semantics: n adaptors -> 2n param tensors "
        f"(got {n_tensors} != 2*{n_adaptors})"
    )

    # 真实训练一步：合成 loss 只对 LoRA 参数求梯度 → 真实参数值 +
    # 真实 optimizer state（exp_avg/exp_avg_sq 非零）
    actor_optimizer = torch.optim.AdamW(lora_info["lora_params"], lr=1e-3)
    actor_optimizer.zero_grad()
    loss = sum((p * p).sum() for p in lora_info["lora_params"])
    loss.backward()
    actor_optimizer.step()

    lora_state = extract_lora_state(lora_info)
    ident = build_lora_identity(lora_state, lora_info)
    assert ident["n_lora_param_tensors"] == len(lora_state), (
        "build_lora_identity must record the param-tensor count "
        f"(got {ident.get('n_lora_param_tensors')!r}, state has "
        f"{len(lora_state)} tensors)"
    )
    base_snapshot = {
        k: v.detach().cpu().clone()
        for k, v in policy.model.state_dict().items()
    }

    contract = RLSystemContract()
    critic = ValueCritic(contract, encoder=PolicyObsEncoder(policy))
    critic_optimizer = torch.optim.AdamW(
        critic.value_head_parameters(), lr=1e-3
    )

    data = _restore_ready(
        _v3_data(
            schema_version=SCHEMA_VERSION,
            lora_state=lora_state,
            # R7-6 v6: the save side records the sorted LoRA key order in
            # the checkpoint (restore's self-consistency gate demands
            # lora_name_order == sorted(lora_state)) — the fixture mirrors
            # production wiring.
            lora_name_order=sorted(lora_state),
            # R7-6 v6: the actor optimizer's parameter-name axis through
            # the registered bundle — derived via the SAME production
            # helper the save side uses (restore validates it against the
            # live wiring before any tensor load).
            actor_optimizer_param_names=_actor_optimizer_param_names(
                actor_optimizer, policy),
            lora_identity={
                **ident, "n_lora_param_tensors": len(lora_state),
            },
            environment_rng_state={
                "seed_source": "sampler_state",
                "python_hash_seed": "7",
                "restorable": True,
            },
        ),
        policy, critic, actor_optimizer, critic_optimizer,
    )
    path = tmp_path / "real_lora.pt"
    CheckpointBuilder.save(data, path)

    # ── fresh 世界（恢复侧）──
    fresh_policy = load_ddpm_policy(ckpt, device="cpu")
    fresh_lora = insert_lora(fresh_policy.model, r=4, alpha=4, verbose=False)
    fresh_critic = ValueCritic(contract, encoder=PolicyObsEncoder(fresh_policy))
    fresh_opt = torch.optim.AdamW(fresh_lora["lora_params"], lr=1e-3)
    fresh_opt2 = torch.optim.AdamW(
        fresh_critic.value_head_parameters(), lr=1e-3
    )

    restored = restore_training_state(
        str(path), fresh_policy, fresh_critic, fresh_lora,
        fresh_opt, fresh_opt2,
        runtime_identity=_runtime_identity(data),
        structured_identity=_good_structured(),
    )

    # 1) LoRA 张量 bitwise 恢复（每个参数张量）
    for i, p in enumerate(fresh_lora["lora_params"]):
        assert torch.equal(p.detach().cpu(), lora_state[f"lora_{i}"]), (
            f"LoRA tensor lora_{i} not restored bitwise"
        )
    # 2) base UNet 未改变（restore 只加载 LoRA 适配器 + 保存时的 base）
    fresh_base = {
        k: v.detach().cpu() for k, v in fresh_policy.model.state_dict().items()
    }
    assert set(fresh_base) == set(base_snapshot)
    for k in base_snapshot:
        assert torch.equal(fresh_base[k], base_snapshot[k]), (
            f"base weight {k} changed by restore"
        )
    # 3) optimizer state 恢复（exp_avg/exp_avg_sq bitwise）
    saved_opt = data.actor_optimizer_state["state"]
    restored_opt = fresh_opt.state_dict()["state"]
    assert set(restored_opt) == set(saved_opt), (
        "actor optimizer param indices not restored"
    )
    for idx in saved_opt:
        for k in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(restored_opt[idx][k], saved_opt[idx][k]), (
                f"actor optimizer state [{idx}].{k} not restored bitwise"
            )
    # 4) counters 返回（resume 后从 checkpoint 继续编号）
    assert restored["ppo_update_count"] == data.ppo_update_count
    assert restored["optimizer_step_count"] == data.optimizer_step_count
    assert (
        restored["environment_transition_count"]
        == data.environment_transition_count
    )
    assert restored["completed_episode_count"] == data.completed_episode_count
    assert restored["outer_iteration"] == data.outer_iteration


# ═══════════════════════════════════════════════════════════════════
# 裁决④: env v2 episode_config 路径 — 稳定 blake2b reference
# ═══════════════════════════════════════════════════════════════════


def test_env_v2_episode_config_seed_matches_stable_blake2b_reference():
    """修复语义：字符串 world_id 原样进 canonical JSON → blake2b →
    episode_rng_seed。RED 失败点（修复前）：payload 里的 world_id 是
    abs(hash(str(uid)))（进程随机盐）→ digest 不同 → 断言失败（P0-2）。"""
    if not os.path.isfile(OBSTACLES):
        pytest.skip(f"obstacles json not found: {OBSTACLES}")
    env = _v2_env()
    cfg = _episode_config()
    obs, info = env.reset(options={"episode_config": cfg})
    assert info["episode_id"] == cfg["episode_id"]
    expected = int.from_bytes(hashlib.blake2b(
        json.dumps({
            "world_id": UID,  # 修复后语义：稳定字符串原样
            "goal_w": [round(float(v), 4) for v in env.goal_w],
            "drone_pos": [round(float(v), 4) for v in env._drone_pos],
        }, sort_keys=True).encode("utf-8"),
        digest_size=4,
    ).digest(), "little")
    assert env._episode_rng_seed == expected, (
        f"episode_rng_seed {env._episode_rng_seed} != stable reference "
        f"{expected} — world_id must enter the payload as the stable "
        "string, not hash(str(world_id))"
    )


def test_env_v1_random_path_seed_matches_stable_blake2b_reference():
    """env v1（world_pool random path）同修复：固定 seed 下 start/goal 与
    world 选择确定 → 字符串 world_id 原样进 payload。RED 失败点（修复前）：
    hash(str(world_id)) & 0x7FFFFFFF（P0-2）。"""
    if not os.path.isfile(OBSTACLES):
        pytest.skip(f"obstacles json not found: {OBSTACLES}")
    from diffusion_policy_3d.env.kinematic_uav_env import KinematicUAVEnv
    from diffusion_policy_3d.model.common.synth_pointcloud import (
        SynthPointCloud,
    )

    env = KinematicUAVEnv(
        world_pool=[(UID, OBSTACLES)],
        synth_pc_factory=lambda p: SynthPointCloud(p, device="cpu"),
        max_episode_steps=8,
        seed=42,
    )
    obs, info = env.reset(seed=42)
    assert info["world_id"] == UID
    expected = int.from_bytes(hashlib.blake2b(
        json.dumps({
            "world_id": UID,  # 修复后语义：稳定字符串原样
            "goal_w": [round(float(v), 4) for v in env.goal_w],
            "drone_pos": [round(float(v), 4) for v in env._drone_pos],
            "seed": 42,
        }, sort_keys=True).encode("utf-8"),
        digest_size=4,
    ).digest(), "little")
    assert env._episode_rng_seed == expected, (
        f"episode_rng_seed {env._episode_rng_seed} != stable reference "
        f"{expected} — env v1 must not route world_id through hash()"
    )


# ═══════════════════════════════════════════════════════════════════
# 裁决⑤: 两个 fresh process（PYTHONHASHSEED unset）同一 task 相同初始 obs
# ═══════════════════════════════════════════════════════════════════


def _fresh_process_pc_sha256() -> str:
    """Run a bare interpreter (PYTHONHASHSEED removed) that builds the env
    v2, resets the deterministic episode_config, and prints the sha256 of
    the initial point_cloud frame. Returns (sha256, stderr) — the subprocess
    env deliberately has NO PYTHONHASHSEED so str-hash randomization is
    active unless the fix removes the hash() dependence."""
    script = (
        "import hashlib, sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from diffusion_policy_3d.env.kinematic_uav_env_v2 import "
        "KinematicUAVEnv\n"
        "from diffusion_policy_3d.model.common.synth_pointcloud import "
        "SynthPointCloud\n"
        f"env = KinematicUAVEnv(world_pool=[({UID!r}, {OBSTACLES!r})],\n"
        "    synth_pc_factory=lambda p: SynthPointCloud(p, device='cpu'),\n"
        "    max_episode_steps=8)\n"
        f"obs, info = env.reset(options={{'episode_config': "
        f"{json.dumps(_episode_config())}}})\n"
        "pc = obs['point_cloud'][0]\n"
        "print(hashlib.sha256(pc.tobytes()).hexdigest())\n"
    )
    run_env = dict(os.environ)
    run_env.pop("PYTHONHASHSEED", None)
    proc = subprocess.run(
        [sys.executable, "-c", script], env=run_env,
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, (
        f"fresh-process probe failed:\n{proc.stderr}"
    )
    return proc.stdout.strip()


def test_two_fresh_processes_same_task_same_initial_obs():
    """裁决⑤：两个互不相关的解释器进程（PYTHONHASHSEED unset）执行同一
    task，初始点云必须逐字节一致。RED 失败点（修复前）：hash() 进程随机
    盐 → 两进程 episode_rng_seed 不同 → point_cloud 不同（P0-2）。"""
    if not os.path.isfile(OBSTACLES):
        pytest.skip(f"obstacles json not found: {OBSTACLES}")
    h1 = _fresh_process_pc_sha256()
    h2 = _fresh_process_pc_sha256()
    assert h1 == h2, (
        "two fresh processes produced different initial point clouds:\n"
        f"  process 1: {h1}\n  process 2: {h2}\n"
        "the episode seed must derive from a stable blake2b of the task "
        "payload, never hash(str(world_id))"
    )
