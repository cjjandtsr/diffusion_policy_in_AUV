# F10 第 0 步：盘点与复用核对（只读回报）

日期：2026-11-25（America/Chicago）。执行端：WSL。依据：《F10 去噪内碰撞引导系统评价 WSL 执行指引 v1_0》。
本文件只报事实，不作判定；不写评价代码（按指引“第 0 步 … 回报后再写代码”）。

## 0. 一句话结论

| 结论 | 内容 |
|---|---|
| 可复用 | S0（P6 #E2）dev420、val20；S3（BC #E2）dev420、val20；S2（G0）dev420、val20（G0 探针原迹）；S5（EGO）论文 186 银行 930 回合 |
| 需新跑 | S1（dev420 / val20 / 186）、S4（dev420 / val20 / 186）、S2（186）、S0（186）、S3（186） |
| 186 银行规模 | 186 任务 × 5 重复 = 930 回合；5 臂 = 4650 回合 |
| 停止条件 | 未触发：已有结果身份均可追溯；无 import 报错；四个银行哈希与以往记录一致 |
| 需 Claude 知情 | 论文 186 银行与 test12 共享 215/240 个初始状态（43/48 任务、11/12 世界）；与 dev420、val20 完全不重叠（0 个初始状态、0 个世界） |

## 1. 盘点表

“回合数”= 评价条件数（条件 = 任务 × 重复）。所有“已有”的 dev420/val20 均满足 #E2 协议。

| 臂 | 策略 | 协议 | dev420（420） | val20（80） | 论文 186 银行（930） | test12 |
|---|---|---|---|---|---|---|
| S0 | P6@u256 | #E2 | **已有** 420（F5） | **已有** 80（F5） | **无** → 新跑 | 不运行（见 §4.4） |
| S1 | P6@u256 | #E4（g1） | 新跑 | 新跑 | 新跑 | 不运行 |
| S2 | P6@u256 | #E4b（G0） | **已有** 420（G0 探针） | **已有** 80（G0 探针） | 新跑 | 不运行 |
| S3 | BC | #E2 | **已有** 420（F5） | **已有** 80（F5） | **无** → 新跑 | 不运行 |
| S4 | BC | #E4（g1） | 新跑 | 新跑 | 新跑 | 不运行 |
| S5 | EGO | 规划器 | **无**（按第 0 步第 3 条不新跑） | **无**（不新跑） | **已有** 930（论文线） | 不运行 |

## 2. 已有结果的身份明细

### 2.1 S0：P6@u256 #E2

| 项 | dev420 | val20 |
|---|---|---|
| 原迹目录（WSL） | `program/results/routea_p6_u256_e2_dev420_run01_restored/full/P6` | `program/results/routea_p6_u256_e2_val20_run01/full/P6` |
| 仓库同源路径 | `jindu/routea_p6_u256_e2_dev420_run01/full/P6`（main 分支）与 `jindu/routea_f5_run01/results/eval_dev/full/P6` | `jindu/routea_f5_run01/results/eval_val/full/P6`（main） |
| 回合数 | 420（84 任务 × 5 重复） | 80（80 任务 × 1） |
| 结局（1 m） | reached 203 / collision 143 / time_limit 74 | reached 42 / collision 14 / time_limit 24 |
| 任务银行 | `…/routea_p6_u256_e2_dev420_run01_restored/bank.json`，文件 sha256 `1a638813…3783`，content_hash `863bf1fb…996a` | `…/routea_p6_u256_e2_val20_run01/bank.json`，文件 sha256 `3dbe2fcb…ad66`，content_hash `a2b28e0c…2017` |
| 评价脚本 | `scripts/run_routea_xk_evaluation.py` sha256 **15bca386**（提交 `15ad24e2`） | 同左 |
| 引导脚本 | 无（#E2） | 无 |
| 权重 | `…/routea_a2_selective_keep_run01/T/checkpoint_update_000256`，`state.pt` sha256 `a91c9f42…dd43`，update 256 | 同左 |
| 冻结原生源 | `program/results/routea_test12_independent_run02/native_a2`，runtime identity `d987235a…b61f` | 同左 |
| LoRA 分支 | `actual_use_base_policy=false`（走 LoRA 分支） | 同左 |

### 2.2 S3：BC #E2

| 项 | dev420 | val20 |
|---|---|---|
| 原迹目录 | `jindu/routea_f5_run01/results/eval_dev/full/bc`（main 分支；本地仅存于 `routea_f5_run01.tar.zst`） | `jindu/routea_f5_run01/results/eval_val/full/bc` |
| 回合数 | 420 | 80 |
| 结局（1 m） | reached 110 / collision 150 / time_limit 160 | reached 29 / collision 16 / time_limit 35 |
| 任务银行 | 同 S0 dev420（`863bf1fb…`） | 同 S0 val20（`a2b28e0c…`） |
| 评价脚本 | `scripts/run_routea_xk_evaluation.py` **15bca386** | 同左 |
| 权重 | 同一 `checkpoint_update_000256`，`actual_use_base_policy=true`（基础分支） | 同左 |

### 2.3 S2：P6@u256 G0（#E4b）—— G0 探针原迹

| 项 | dev420 | val20 |
|---|---|---|
| 原迹目录 | `program/results/routea_probe_e3g_run01{,_delivery}/routea_probe_e3g_run01/runs/G0_dev` | `…/runs/G0_val` |
| 回合数 | 420 | 80 |
| 结局（1 m） | reached 300 / collision 118 / time_limit 2 | reached 59 / collision 14 / time_limit 7 |
| 任务银行 | dev420 银行（`1a638813…`） | val20 银行（`3dbe2fcb…`） |
| 评价入口 | `scripts/run_routea_e3g_probe.py` sha256 **c5a22b98** | 同左 |
| 引导脚本 | `scripts/routea_denoising_guidance.py` sha256 **41c3d47c** | 同左 |
| 采样包装 | `scripts/run_routea_xk_evaluation.py`（b44c16b8，#E2 覆盖：clip 2.0、min_std 0.1、deterministic） | 同左 |
| 引导参数 | `d_safe=0.6, eta=0.2, M=2, consistency=0`（identity.json 记录 eta 0.2 / consistency 0） | 同左 |
| 注入点 | `p_mean_var` 内、x0 clip 之前的 `_probe_guidance.update(x_recon, index)` | 同左 |
| 回代校验 | `regression_pass.json`：E2 dev/val 首条件整条 JSONL SHA256 与 F5 原迹逐位相同 | 同左 |

### 2.4 S5：EGO 论文 186 银行

| 项 | 值 |
|---|---|
| 原迹目录 | `paper_186_v1_0_run01/ego/full/ego`（本地仅存于 `program/results/paper_186_v1_0_run01.tar.zst`） |
| 回合数 | 930（186 任务 × 5 重复） |
| 论文线结局 | reached 726 / collision 32 / time_limit 172（论文判据，见 §5 注意） |
| 任务银行 | `paper_186_v1_0_run01/input/bank.json`，文件 sha256 `90611f88…a62`，content_hash `d34c10d6…742b` |
| 策略身份 | `protocol.json: identity = 8965396b…f8ee`；`binaries`：ego_planner_node `f0ba2456…`、libbspline_opt.so `956f7e5e…`、libpath_searching.so `7741527c…`、libplan_env.so `c2248f9b…` |
| 执行协议 | `common_H8_H4_0.3s`（与 #E2 的 8 步策略 / 4 步执行 / 0.3 s 同构），`repeat_policy=all_five_frozen_reset_seeds` |
| 评价脚本 | ROS 规划器本身，无 Python 评价入口；比较统计在 `ego/comparison_statistics.json` |

## 3. G0 探针的入口与引导脚本核对（指引第 0 步第 1 条）

| 问题 | 回答 |
|---|---|
| 探针评价入口 | `scripts/run_routea_e3g_probe.py`，sha256 **c5a22b98bc621d1ae524916e2f5443bc20edd23f7947dfc5e4a008d7f758938b** |
| 探针引导脚本 | `scripts/routea_denoising_guidance.py`，sha256 **41c3d47cb9d69daaed715ecf9ff50f124ca8a69ffb13f658070ac1e81b4071b8** |
| 与 `f9-code-a2-20261124` 逐字相同？ | 入口：**相同**（c5a22b98）。引导脚本：**不相同**（分支为 **8fa3b263…1347**） |
| 差异内容 | 仅两处参数化：`costs(..., d_safe=.6)` 与 `Guidance(..., d_safe=.6, updates=2)`，并把 `range(2)`/`update_index == 0` 改为 `range(self.updates)`/`update_index + 1 < self.updates`。默认值（0.6 / 2）与探针常量逐字等价，故 G0 在默认参数下语义相同（T5 将按逐位核对） |
| 采样包装 | 探针期间 `run_routea_xk_evaluation.py` = **b44c16b8**；与 F5 期的 15bca386 差异仅 `evaluation_reproduction`（F6 迁移副本）分支，标准 P6/BC 路径不受影响；探针 regression 已用整条 JSONL SHA 验证 |

## 4. 任务银行

### 4.1 哈希

| 银行 | 路径 | 文件 sha256 | content_hash | 规模 |
|---|---|---|---|---|
| dev420 | `program/results/routea_p6_u256_e2_dev420_run01_restored/bank.json` | `1a6388136ea329f680c7055227f915a25ee25417df752a2cb7d85e30e57c3783` | `863bf1fb2a41c991b73cc92493aa7d8732c711b4e48c6a8d063bd2dd8c59996a` | 84 任务 / 21 世界 / 420 回合（schema `routea_train_domain_paired_dev_v1`） |
| val20 | `program/results/routea_p6_u256_e2_val20_run01/bank.json` | `3dbe2fcb2a237eb7ecb35a4e4c93c59f2a972ce313fdf4fe43eba30c3913ad66` | `a2b28e0cf2ca32ee8a987d0f1a03f94d2783dea57fee1bd096fdb8e5ca542017` | 80 任务 / 20 世界 / 80 回合（schema `routea_fixed_split_v1`, repeats=1） |
| 论文 186 | `paper_186_v1_0_run01/input/bank.json`（本地在 `paper_186_v1_0_run01.tar.zst`） | `90611f88987c6859b9aed0eacc942f34e68ba4b77335d32d22a941b2c62f1a62` | `d34c10d660c47fe609ebd83ce271a18e41f2d76918187ced049a583e5aa4742b` | 186 任务 / 40 世界 / 930 回合（schema `paper_186_frozen_v1`） |
| test12（只报位置与哈希） | `program/results/routea_test12_independent_run02/bank/bank.json` | `72c336e98d3e112f490ef8c5927580c430b73504e3cb62926133aa8a2d9d78d7` | `b3a872bbe9683ea952d6767a8da9a20cb12a1ce11d5d24ef2953c7861ec80474` | 48 任务 / 12 世界 / 240 回合（×4 策略 = 960；schema `routea_fixed_split_v1`, repeats=5） |

### 4.2 三者关系（按“世界 + 起终点”与初始状态哈希双口径）

| 对比 | 任务重叠 | 世界重叠 | 初始状态（episode initial_hash）重叠 |
|---|---|---|---|
| 186 银行 vs dev420 | 0 / 84 | 0 / 21 | **0 / 930** |
| 186 银行 vs val20 | 0 / 80 | 0 / 20 | **0 / 930** |
| 186 银行 vs test12 | **43 / 48** | **11 / 12** | **215 / 240** |
| dev420 vs val20 | 0 | 0 | 0 |

186 银行的 `initial_npz` 直接指向 `routea_test12_independent_run02/bank/initial/`，故与 test12 共享 43 个任务的 5 个重置种子（215 条）。

### 4.3 是否进入过任何选参

| 选参环节 | 用的银行 | 186 是否参与 |
|---|---|---|
| F9 第 0 步（g 选择，`selection.json`） | `routea_f8_diag_run03/bank.json`（425 条件 = 85 库任务 × 5） | **否** |
| G0/G1 网格与正式（探针） | dev420（420）+ val20（80）+ f8_diag train（425） | **否** |
| 论文线三方规划器（ego/fast/px4/super） | 186 银行 | 是（但只是被评价对象，不进任何选参） |

结论：186 银行没有进入过 F9 的 g 选择，也没有进入 G0 网格；它只作为论文线规划器评价集出现过。

### 4.4 test12

`program/results/routea_test12_independent_run02/bank/bank.json`（sha256 `72c336e9…78d7`，content_hash `b3a872bb…0474`）。按指引不运行，本步未运行任何 test12 条件。

## 5. 协议参数与 #E2 的差异

| 维度 | #E2（S0/S3/S5 及 S1/S4 的基底） | #E4b G0（S2） | #E4 g1（S1/S4） |
|---|---|---|---|
| 采样 | DDIM 10 步、deterministic、固定初噪 + 每决策固定动作种子 | 同 #E2 | 同 #E2 |
| 截断 | `denoised_clip_value` 2.0（原生 1.0 被覆盖） | 同 | 同 |
| 采样/对数概率下限 | `min_std` = 0.1 × 11 | 同 | 同 |
| 策略分支 | P6：LoRA 分支；BC：基础分支 | P6 LoRA | P6 LoRA / BC 基础 |
| 执行段 | 8 步策略动作取前 4 步执行（`n_exec=4`），`traj_dt=0.3`，deadline 60 决策，`collision_thresh=0.1`，执行噪声 0、扰动关 | 同 | 同 |
| 引导 | 无 | `d_safe 0.6, eta 0.2, M 2, consistency 0`；注入 `p_mean_var`，仅 `index >= 7` 的 3 个去噪步可改 | `d_safe 0.6, eta 0.4, M 4, consistency 0`（来源见下） |
| 引导代价取点 | — | 当前观测 `point_cloud[-1]` 转到世界系后 `z >= 0.1` | 同 |
| 身份字段 | `evaluation_wrapper_sha256` | 另记 `guidance`（注入前后函数 SHA256）与 `sampling_protocol` | 同 G0 |

g1 的 `consistency=0` 依据：`diffusion_policy_3d/rl/dppo_uav_rebuild/engine/f9_online.py:138` 为 `Guidance(unnormalize, setting['eta'], 0, setting['d_safe'], setting['M'])`；F9 `selection.json` 的选中 `setting` 为 `{d_safe: 0.6, eta: 0.4, M: 4}`。即 g1 是 F9 的正式选中设置，不是 G0 探针里的“G1（consistency=1, eta=0.1）”。

**S5 的判据注意**：论文线 EGO 用 `paper_goal_radius_m` 与 `paper_safe_arrival` 记录（`official_paper_episodes.jsonl`），其 `min_goal_distance_m` 与逐决策轨迹均保留，可按 1 m 重判；重判由 Claude 用原迹复算（本步不判）。

## 6. 代价估算（用于排期，不是判定）

| 依据 | 实测 |
|---|---|
| F5 dev420 P6（#E2） | 420 回合 / 52 分钟 ≈ 7.4 s/回合 |
| G0 探针 G0_dev | 420 回合 / 6254 s ≈ 14.9 s/回合（含引导） |
| G0 探针 G0_val | 80 回合 / 1495 s ≈ 18.7 s/回合 |
| 探针 E3_dev | 420 回合 / 9744 s ≈ 23 s/回合（仅参考） |

按单进程 15 s/回合估：dev420 S1+S4 ≈ 3.5 h；val20 S1+S2 ≈ 0.7 h；186 银行 5 臂 4650 回合 ≈ 19 h。执行端会用多 worker 调度（G0 探针 formal 实测 3 路仅约 1.36× 加速）；最终排期在 Step 3 前回报。

## 7. 需要 Claude 知情/确认的点（不阻断）

1. **186 银行与 test12 共享 215 个初始状态**（§4.2）。186 上的数字不能声称独立于 test12；若主表同时用 test12 与 186，两者不是独立样本。
2. **S0/S3 在 186 上没有原迹**。论文线只跑过 ego/fast/px4/super；P6、BC 在 186 上仅有延迟计时（`paper_timing_p6_u256.json` / `paper_timing_bc_u256.json`），没有回合。故 186 上的 S0、S3 需新跑（930 × 2）。
3. **g1 的 consistency 取 0**（§5 依据）。若方案里的 g1 要带 consistency=1，请指正，我按新参数跑。
4. **BC 分支的引导注入点**：探针的 `install()` 只作用于 `system.official_runtime`（P6 的 LoRA runtime）。BC 走 `system.bc_runtime`，Step 1 需要让 `install()` 作用在实际被调用的那个 runtime 上；两种策略的注入点、`d_safe`、`eta`、`M` 完全一致，仅 runtime 对象不同。这是 Step 1 的实现细节，不影响本步结论。
5. **186 上的动作种子规则**：dev420 用 `canonical_hash(['policy-sample-v1', task_id, repeat, decision])`；val20 因 `routea_fixed_split_v1/repeats=1` 改用 `_action_seed(task_id, decision)`；186 银行 schema 为 `paper_186_frozen_v1`，Step 1 将按 dev420 规则（canonical_hash）并在 README 标明。
6. **S5 的 dev420/val20**：本机与仓库均无 EGO 在这两个划分上的原迹，按指引第 0 步第 3 条不新跑，只在 186 上对比。

## 8. 未做/未写

- 未修改任何代码；未创建 F10 结果目录；未运行任何 F10 回合。
- 未把 725 MB 的 F5 归档、998 MB 的 186 归档解包进仓库；本报告只引用路径与哈希，原迹仍在其原有位置。
