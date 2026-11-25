# F9 v1_3 阶段A 回传（g1 裁决 + P28 + measure + gate）

日期：2026-11-22/23（America/Chicago）。交付包 `F9_v1_3_code_v1_2.zip`（`0973447a…64a1`）。
本目录是执行指引 v1_0（g1 裁决版）的第 1–6 步产物。

## 结论先行

| 项 | 值 |
|---|---|
| 选参 | **g1：d_safe 0.6 / eta 0.4 / M 4**（用户裁决，原规则 winner 为 g5，已留档） |
| P28 训练 | 从 P6 u256 到 u264，176 分钟，rc=0 |
| G-0 | 记录为不通过、不阻断（F9 两段式梯度 vs P22 一次 backward，浮点累加顺序不同） |
| 首轮隔离 | **passed**（1024 条转移，开/关教师回合的 PPO 更新前数据逐位相同） |
| 预填 | 3588 转移、可入库 73、λ_g=0.13076972047088628 |
| D_T / D_C / D_P6 | 0.16210 / 0.19515 / 0.21066 |
| 到达数 T/C/P6 | 262 / 256 / 253（各 425 条件） |
| 离线比 r | **未计算**：留出仅 12 个决策（要求 ≥20），`frozen.json adequate=false` |
| **gate 判定** | **`pending_r`** |

`pending_r` 按指引作停止处理：不开阶段 B、不重开、不延长 8 轮预算、未做任何补测。

## 目录

| 路径 | 内容 |
|---|---|
| `ruling/` | `ruling_selection_20261122.json`、新旧 `selection*.json`、裁决前后 sha256 |
| `shadow_g1/` | g1 影子探针 on/off 结果与 trace（`passed=true`；D 0.25 vs 0，除 `probe_metrics` 外逐字段相同） |
| `P28/` | 配置、fork 血统、预填、标定、`random_isolation_P28.json`、逐轮 metrics 257–264、`f9_collect/update`、检查点身份（无权重）、`gate.json` |
| `measure/` | `eval_refs.json`、`task_strata.json`（49/36 分层）、`offline/`（冻结留出，`adequate=false`）、三份逐条件 CSV |
| `evaluations/` | P28/P29/P6 各 425 条件的 commit 与原迹 |
| `execution/` | 本轮全部 log、rc、命令快照、裁决 sha 记录 |
| `aborted/` | 首次 g1 shadow-probe 失败说明（日志被重跑覆盖，已如实标注重建） |

## 测量明细（train 库 85×5，未评 dev/val/test12）

| 策略 | D_num/D_den | D | 到达 | collision | time_limit |
|---|---|---|---|---|---|
| P28（T） | 1533/9457 | 0.16210 | 262 | 145 | 18 |
| P29（C） | 1650/8455 | 0.19515 | 256 | 161 | 8 |
| P6@u256 | 1616/7671 | 0.21066 | 253 | 170 | 2 |

49/36 分层（P6 已解 49 / 未解 36，按 P6 无引导 ≥3/5 判已解）：

| 层 | T | C | P6 | T−C 新增/流失 | T−P6 新增/流失 |
|---|---|---|---|---|---|
| 已解 49 | 44 | 44 | 49 | +2 / −2 | +0 / −5 |
| 未解 36 | 9 | 8 | 0 | +5 / −4 | +9 / −0 |

## gate 判读依据

`stage_gate`：`t=1533/9457=0.1621` 大于 `0.7*c=0.7×0.1952=0.1366`，D 未达同一性；`r` 因留出不足为 `None`，
故返回 **`pending_r`**（不自动延长 8 轮预算）。留出不足 20 是原始事实，未补测、未改预算。

## 身份

| 项 | 值 |
|---|---|
| P28 u264 state | `b37dccc250a098b84cd469654c69ddf74fddd54a48928a6c25ac1567ba771bb7` |
| P29 u264 state | `bee8de960058a3bf4a5b0d10c96447e8a8b7f760866623e3014bc8e4244d6fa9` |
| P28 runtime identity | `0674c65e16875741ea7017315ff9ff718de4f1030e46355d4062a842ea5012b2` |
| selection.json（g1） | `07845c041eb963cde85ff3693278e311800c65e668d790c602571e3b0fea6ab6` |
| 权重 | 均留在 WSL `$WEIGHTS/P28`、`$WEIGHTS/P29`，本包只有 header/SHA/COMMITTED |

## 偏离与待办

1. `pending_r` → 停止（指引第 5 步）。
2. P29 的 `config.yaml` 里 `f9.selection_sha256` 仍指向旧 g5 selection（P29 是控制臂、`f9.enabled=false`，
   选参不参与计算；audit 只校验 config.yaml 与 configured.json 一致）。
3. 未做 audit 封存；未开阶段 B；未重启任何臂。
