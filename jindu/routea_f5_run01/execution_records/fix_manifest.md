# F5 评价阶段修复清单（2026-11-02）

本文件记录 F5 执行期间**评价阶段**发生的两处偏离与处置。训练侧未做任何修复：四个臂的配置、
checkpoint、reward 源码、seed 全部保持启动时状态。

---

## 修复 1：评价辅助层 v3 门禁放行（裁决 B）

### 触发
`step_04_eval.sh` 在 `eval_dev_P8` 返回 rc=1：

```
File "scripts/run_routea_xk_evaluation.py", line 383, in worker
    adapter=make_adapter(system,config,world,t['role'])
File ".../scripts/routea_paired_support/runtime.py", line 193, in make_adapter
    raise ValueError("common environment is not v2/H4/8")
```

失败发生在**第一个回合之前**，无任何回合记录产生。原因链：`worker` 用 checkpoint 自带
`training_config` 建环境 → P8/P10 是 reward v3 → `env.reward_version='v3'` → 门禁只接受 v2。

### 裁决
用户转达 Codex 裁决「F5评价v3门禁修复与接续 v1.0」：采用 **B**，评价辅助层允许 v2/v3；
H4/8、dt=0.3、60 决策、碰撞余量 0.1、观测/身份/首次事件门禁全部不变；v3 状态推进直接调用
`step_v2`，仅追加 `risk_close`；累计 reward 不得跨 v2/v3 直接解释为策略收益。

### 改动（唯一生产改动）
| 项目 | 值 |
|------|-----|
| 文件 | `scripts/routea_paired_support/runtime.py` |
| 修复前 sha256 | `657d542766b3…` |
| 修复后 sha256 | `f86538e135184f5b16ecda87400e7636d6564e723617fefeb73750690d6158f6` |
| 语义改动 | `env.reward_version != "v2"` → `env.reward_version not in ("v2","v3")`；错误串 `v2/H4/8` → `v2-or-v3/H4/8`；新增一行注释 |
| 未改动 | 同函数其余 5 项判据（`n_exec_steps=4`、`rl_action_steps=8`、`traj_dt=.3`、`max_episode_steps=60`、`collision_thresh=.1`）逐字未变 |

新增测试：`tests/routea_paired/test_common_environment_versions.py`（sha256 `4f1bee22…`）。

### 边界声明（已遵守）
- `scripts/routea_paired_support/runtime.py` **不在** 137 文件执行源身份闭包 `e5f9c80a…` 内，改它不破坏既有 checkpoint 加载。
- `scripts/evaluate_dppo_uav_formal_endpoint.py` **在**闭包内，**未改**。
- 未改原生快照、训练配置、权重、正式 endpoint、seed。
- 已有训练 prepared 闭包未更新；未启动或继续任何训练。

### 证据文件
- `runtime_before.py` / `runtime_before.sha256` — 修复前原件
- `runtime_after.py` / `runtime_after.sha256` — 修复后落地件
- `runtime_delivered_windows.py` / `runtime_delivered_windows.sha256` — 裁决交付件（与 after 字节相同）
- `runtime_before_after.diff` — 逐行差异
- `test_common_environment_versions.sha256`
- `00_tests.log` / `00_tests.rc` — **24 passed**
- 原失败报告：`blocker_eval_v3_gate.md`（交付目录）；原失败现场：`eval_dev/full/P8_failed_before_episode/`

---

## 修复 2：冻结 val bank 的绝对路径失效

### 触发
`val_bc` 返回 rc=1（2026-11-02 06:15）：

```
File ".../native_a2/scripts/routea_paired_support/runtime.py", line 183, in make_adapter
    if file_hash(world["obstacle_path"]) != world["obstacle_sha256"]:
FileNotFoundError: ... '/home/cjj/AUV/program/results/routea_xk_run01/val_bank/obstacles/…000051.json'
```

同样发生在第一个回合之前（`eval_val/full/bc` 只有 `policy_provenance.json`）。

### 原因
冻结 val bank 的 `worlds[uid].obstacle_path` 是 **20 条绝对路径**，指向 2026-11-01 磁盘清理
中被删除的 `program/results/routea_xk_run01/`；`bank.json` 本身与其 `content_hash` 未被触碰。
（同一批清理也删掉了最初定位的 VAL_BANK，已于 2026-11-01 18:30 改指向 jindu 现存副本。）

### 改动（零字节改动）
仅新增符号链接，不复制、不改写、不重新生成：

```
program/results/routea_xk_run01/val_bank -> Improved-3D-Diffusion-Policy-main/jindu/routea_xk_run01/val_bank
```

### 核验
- 20/20 个 `obstacle_path` 经该链接可解析，且 sha256 **等于** bank 记录的 `obstacle_sha256`（`obstacles_bad=0`）
- `bank.json` 文件 sha256 `3dbe2fcb2a237eb7ecb35a4e4c93c59f2a972ce313fdf4fe43eba30c3913ad66`
- `content_hash` 仍为 `a2b28e0cf2ca32ee8a987d0f1a03f94d2783dea57fee1bd096fdb8e5ca542017`（80 任务 / 20 世界）
- 记录见 `evaluation_gate_fix_v1/val_bank_path_fix.log`
- 复跑前置门在 `fix_eval_resume_val.sh` 中固化为硬检查（`obstacles_bad != 0` 即 rc=3 停步）

### 失败现场保留
`evaluation_gate_fix_v1/val_bc_failed_before_episode/`：`val_bc.rc`=1、`val_bc.log`、
`val_bc.command.txt`、空输出目录、`ARCHIVE_NOTE.txt`。归档前守卫确认 commits=0、traces=0。

### 边界声明
未改 bank 内容、未改 hash、未删困难样本、未换 seed、未关校验、未改 checkpoint。

---

## 未发生的事项（明确记录）

- 没有任何训练被启动、续跑或修改；`prepared/training/*` 只有训练期写下的 metrics。
- `dev/bc` 与 `dev/P6` 未重跑（复用第一轮结果，与 #E2 2026-10-31 数值逐项一致）。
- 没有覆盖任何已存在的分层冻结或分析产物（`test ! -e` 守卫全程生效）。
- 权重 `state.pt` **未进入**回传包（`--exclude '*.pt'` + `--exclude 'state.pt'`），只回传 header/SHA256/COMMITTED 证据。
