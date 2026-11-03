# Task Plan: F5 任务分布 + reward v3 执行（指导 v1.0，2026-11-01）

## Goal
在 WSL 按 `F5任务分布与reward v3执行指导 v1.0` 逐条执行：交付核验 → bank 生成(8400) → prepare(4 配置)
→ 四臂训练 P8/P9/P10/P11（u129→u192，约 53h）→ 六策略统一评价（dev420 + val20，约 6h）
→ 分析 → 回传包。执行侧只报事实，能力判读归 Codex。

## Inputs (已定位，未新建)
| 角色 | 路径 | 证据 |
|------|------|------|
| RECEIPT (P2@u128 parent) | `program/results/routea_a2_selective_keep_run01/parent/receipt.json` | schema=a2_parent_v1, 133 files, actor 94/critic 10, r1_probe |
| REFERENCE (549 样本) | `program/results/routea_a2_selective_keep_run01/reference/` (bank .pt 同目录) | reference_manifest samples=549, sha=8759fc40… |
| DEV_BANK (420 回合) | `…/data/outputs/diagnostics/routea_bc_a0_r1_paired_dev_20261010_py38fix_run02/bank/bank.json` | content_hash=863bf1fb…, initial/ 420 npz |
| VAL_BANK (80 条件) | `program/results/routea_xk_run01/val_bank/bank.json` | content_hash=a2b28e0c…, initial/ 80 npz |
| admission manifest | `program/reports/routeA_rl_env_v1/calibration/admission_manifest.json` | 21 世界 |
| worlds/train | `PX4-Autopilot/Tools/sitl_gazebo/worlds/train` | 180 world + 180 obstacles |
| P6_NATIVE | `program/results/routea_test12_independent_run02/native_a2` | E2 plan.json 同源 |
| P6_CHECKPOINT | `…/data/outputs/dppo_uav_rebuild/routea_a2_selective_keep_run01/T/checkpoint_update_000256` | state.pt sha=a91c9f42…dd43 |

## Phases
- [x] Phase 0: 事实核对（STATUS/TODO/设计定位；23 交付文件双侧 hash）
- [x] Phase 1: 交付复制（备份 12 个被覆盖文件 → 复制 23 → WSL 侧 23/23 复验）
- [x] Phase 2: 预检（00_import；pytest 交付测试 + 回归）
- [x] Phase 3: 01_bank（21 世界 × 400 = 8400 候选）
- [x] Phase 4: 02_prepare（P8/P9/P10/P11 四配置 + prepared.json）
- [x] Phase 5: 训练四臂 u129（冒烟兼首轮门禁）→ u192（约 53h）
- [x] Phase 6: 统一评价 6 策略 × dev(rep5)/val(rep1) = 3000 条件
- [x] Phase 7: freeze-strata × 2 + evaluate 分析
- [x] Phase 8: 回传包（OUT 全目录 + SHA256 清单）

## Key Questions
1. 训练 53h 期间文件冻结是否守得住（不改受保护源码）？
2. 磁盘余量 14G 是否够（预估训练 ~3G + 评价 ~1.5G + 回传 ~5G）？
3. u129 门禁（20 提交 / 1024 转移 / keep 0.01061006 / task_bank 非 original）能否过？

## Decisions Made
- DEV_BANK 使用原始 diagnostics 路径（`initial_npz` 为相对路径，必须与 `initial/` 同目录）
- 计划/记录放 `program/task_distribution_implementation_20261031/`，不污染 `$OUT`（指导要求 `test ! -e "$OUT"`）
- 长跑用 setsid 分离启动（WSL2 终端崩溃史）

## Errors Encountered
- `probe_source_hash.py` 首跑 `build_inventory(str)` → TypeError（需 Path）；修正后通过；RESTORED_OK 复验 23/23。
- `analyze_routea_credit_counterfactual.py --self-check` 第三项在 import 期即 RuntimeError（非测试断言），同样源于固定常量。

## Findings
- **预交付源码闭包 = 135 文件 / `3bcc153c…`；交付后 = 137 文件 / `e5f9c80a…`；测试固定常量 = 111 文件 / `5aed4fb6…`（v1.7 诊断期）**
  ⇒ 3 项 audit 测试失败（test_d05b_cli / test_d2c_cli / test_routea_causal_contracts）在交付前即已失效，与 F5 交付无关。
- 交付后 pytest：交付测试 **31 passed**；回归 `tests/dppo_uav_rebuild tests/routea_paired` = **705 passed / 3 failed / 10 skipped**（3 failed 均为上述历史常量）。

- **01_bank rc=0**：8400 任务 / 21 世界 / 每世界 400；`sha256=a820729f5586f713367a8f4c34929f599aff683310528f7adb5d73fb1b6bc5a5`
  tier: L0 2727 / L1 2090 / L2 438 / L3 2874 / L4 271；kind: random 6943 / near_obstacle_start 1457
  12 个 source×class 组合全部非空（L0 1721/1006、L1 1459/631、L23_random 1596/379、L23_near 1063/274、L4 161/110）
- **02_prepare rc=0**：`F5_PREPARED_P8_P9_P10_P11`；P8(v3+bank,seed 202610311)/P9(v2,seed 202610311)/P10(v3+bank,seed 202610312)/P11(v2,seed 202610312)；
  keep_coefficient=0.01061006、bank seed=run.seed+17、ckpt 序列含 129、prepared.files=143 行
- 世界分类：dense 15（全部 dense_recovery，≥27 障碍）/ sparse 6（其余三角色）

## Status
**Phase 3–4 完成** — Phase 5 训练四臂已分离启动（P9 u129 于 23:57 起跑，runtime_identity=ebcf5013…）。

### Phase 5 运行记录
- 00:01 磁盘由 22G 升至 **38G 可用**（用户自行清理，非本任务产生）；训练/评价/回传预估合计 ~6.5G，余量充足。
- `_execution_last_update` 语义（代码 `pipeline.py:203-219`、`:454`）：`--stop-after-update 129` = fork(u128) 后只跑 **1 个外层更新**（update_index=129），
  写 `metrics_update_000129.json` + `checkpoint_update_000129`；`--stop-after-update 192` = 续跑 130..192 共 63 个外层更新。
  `verify_first_update` 的 `len(metrics['updates'])==20` 指该外层更新内的 20 次已提交优化器更新，`transition_records==1024` 指该次 rollout 审计记录。
- **P9 u129 门禁 PASS**（00:04，耗时 6 分 07 秒；`03_P9_u129.rc=0` + `F5_U129_CONTRACT_PASS`）：
  - `updates`=20 全 committed；`rejected_updates`=0；`hard_stop_reason`=null；`microbatch_evaluations`=400
  - `transition_records`=1024，1024/1024 带 `task_bank`；source 全 `original`（P9 为 v2 对照臂，符合预期）
  - 27 回合 / 1024 步 / `training_usable` 1024/1024；counters 累计 attempted=committed=2580（=128×20+20，与 20 内层步语义一致）
  - `r1_plan_probe`: prefix_l2=0.294724、prefix_mean_abs=0.024292、tail_l2=0.186476、tail_mean_abs=0.018053
  - 权重：`checkpoint_update_000129/` header.json + SHA256.json + COMMITTED.json + state.pt(36.4M)
- **实测速率**：1 个外层更新 ≈ 6 min（rollout 1024 步）⇒ 单臂 u192 = 63 个外层更新 ≈ 6.5 h；四臂合计 ≈ 26 h（低于指导预估 53 h）。
- 00:04 起 P9 u192 续跑（PID 10536，`--resume …/P9/checkpoint_update_000129`）。
- **P9 整臂完成**（05:32，`04_P9_u192.rc=0` + `05_P9_monitor.rc=0`）：64 个外层更新（129–192）、65536 转移；
  ckpt 每 8 更新一个（129/136/144/152/160/168/176/184/192）；`automatic_training_stop=false`。
  8 轮窗口成功率：0.364 / 0.330 / 0.355 / 0.354 / 0.338 / 0.336 / 0.359 / 0.385；整臂 reached 618 / completed 1751 = **0.353**，碰撞 490、truncated 643。
- 05:32 P8 u129 起跑（PID 29012）。**P8 u129 门禁 PASS**（05:39，`F5_U129_CONTRACT_PASS`）：20/20 committed、rejected=0、1024 转移全带 task_bank。
  - reward `schema_version=v3`，terms = newbest/risk/**risk_close**/space/terminal/time；`risk_close` mean −0.05729、min −1.79001（界 [−2,0] 内）、nonzero 161/1024。
  - **按回合**（43 回合）来源比：original 6 / L0 5 / L1 6 / L23_random 10 / L23_near 7 / L4 9，对照目标 15/10/15/30/20/10 在抽样噪声内。
  - **按转移**统计会偏离（original .11 / L0 .27 / L23_random .20 / L23_near .10），原因是回合长度差异大（L0 均 55.2 步 vs L23_near 14.6 步）—— 混合契约是**每回合**抽样，不是每转移。
  - dense_rate：L0 .40、L1 .67、L23_random .90、L23_near .71、L4 .78（n=5~10，与配置 .5/.5/.8/.8/.8 在噪声内）。`original` 记录无 `world_class` 键（113 转移/6 回合的唯一 schema 变体）。
- **P8 整臂完成**（12:03，`04_P8_u192.rc=0` + `05_P8_monitor.rc=0`，`automatic_training_stop=false`）：64 更新、65536 转移。
  - **口径警告**：`statistics.by_source` 在 P8 里按来源切分，若只取 `original` 桶会得到 349 回合（占 15%），与 P9 的 1751（全部都是 original）不可比。全来源合计才是臂级口径。
  - 全来源：completed 2380 / reached 536 / collision 1310 / truncated 534，SR **0.2252**、碰撞率 0.5504、截断率 0.2244。
  - 分来源 SR：L0 .457(243)、L1 .371(345)、**original .352(349)**、L2 .294(180)、L3 .109(1039)、L4 .036(224)。
  - **同 seed 配对事实**：P8 的 original 层 SR .352 / 碰撞 .287 与 P9 全臂 SR .3529 / 碰撞 .2798 几乎相同；两臂差异集中在 L3/L4 硬层（碰撞率 .71/.91）。
  - `risk_close` 全臂 sum −3560.688，非零 9887/65536，最小值触底 −2.0000（= 设计下界）。
  - 每回合来源比在 u129/u160/u192 分别为 original .140/.167/.079，属 n≈38 回合的二项抽样噪声，非抽样器漂移（MIXTURE 为常量）。
- 12:03 P11（v2 对照，seed 202610312）u129 起跑（PID 38292）。
- **P11 整臂完成**（18:29，`04_P11_u192.rc=0` + `05_P11_monitor.rc=0`，`auto_stop=false`）：completed 1749 / reached 593 / coll 499 / trunc 657；
  SR **0.3391**、碰撞率 0.2853、截断率 0.3756。种子级对照：P9（…311）SR .3529 碰撞 .2798，P11（…312）SR .3391 碰撞 .2853。
- 18:29 P10（v3+bank，seed 202610312）u129 起跑。
- **四臂训练全部完成**（11-02 01:03，`TRAIN_MASTER_RC=0`；`04_*_u192.rc=0` 与 `05_*_monitor.rc=0` 各 4 个）：
  每臂 9 个检查点（129/136/144/152/160/168/176/184/192）。
  - P10 整臂：completed 2396 / reached 566 / coll 1301 / trunc 529，SR **0.2362**、碰撞率 .5430；original 层 SR .351 / 碰撞 .310。
  - 两 seed 的 per-source SR 高度一致：L0 .458/.457、L1 .381/.371、L23_near .155/.157、L23_random .143/.123、L4 .056/.036（P10/P8）。
- **Phase 6 启动**：11-02 01:04 分离启动 `step_04_eval.sh`（PID 54473 → 54475 → 54476，首个策略 dev/bc）。
- **bc dev 校验通过**（01:57，`eval_dev_bc.rc=0`，53 min）：420 条记录 / 84 个不同任务 / repeat 1–5，
  全部 `status=valid`、`terminate_at_space_limits=false`；`bank_hash=863bf1fb…`（= DEV_BANK content_hash）。
  - **口径确认**：`dev420` 冻结 bank 本身就是 **21 世界 × 4 任务 = 84 个条件 × 5 重复 = 420 回合**（`bank.episodes` 420 条）。
    故 420 记录是完整分母，不是漏跑。6 策略 ×（420 dev + 80 val）= **3000 条**，与指导口径一致。
  - **冻结初观测配对门**：420/420 条记录的 `initial_hash` 与 bank 逐 (task, repeat) 完全一致，缺失 0、多余 0、错配 0。
  - bc dev 结果：reached 110 / collision 150 / time_limit 160（各 repeat 波动 20–23 / 29–32 / 30–35）。
- **Phase 6 阻塞（11-02 02:43）**：`eval_dev_P8` rc=1 —— `ValueError: common environment is not v2/H4/8`
  （`runtime.py:193`）。原因：`worker` 用 checkpoint 自带 `training_config` 建环境，P8/P10 是 reward v3 → `env.reward_version='v3'`，门禁只接受 v2。
  已按失败协议停步，未改校验/未重跑。详见 `blocker_eval_v3_gate.md`。P9/P11（v2）不受影响；bc dev、P6 dev 已完成且配对门全过。
  - P6 dev 复现核对：reached 203 / collision 143 / time_limit 74 与 2026-10-31 #E2 dev420 完全一致。
  - 闭包事实：`scripts/routea_paired_support/runtime.py` **不在** 137 文件闭包内；`scripts/evaluate_dppo_uav_formal_endpoint.py` **在**闭包内。
- **阻塞解除（11-02 02:57，裁决 B）**：按 `F5评价v3门禁修复与接续 v1.0` 装入交付的 `runtime.py`
  （`657d5427…` → `f86538e1…`，仅把 `reward_version != "v2"` 改为 `not in ("v2","v3")`，其余 5 项与消息外无改动），
  并装入新测试 `tests/routea_paired/test_common_environment_versions.py`（`4f1bee22…`）。
  修复记录目录 `$OUT/evaluation_gate_fix_v1/`：runtime_before/after/delivered + 三份 sha256 + before_after.diff；
  失败现场 `eval_dev/full/P8`（仅 policy_provenance.json，无回合）已归档为 `P8_failed_before_episode/`。
  - `00_tests` rc=0：**24 passed**（含新 v3 门禁测试）。
  - dev/P8 已越过门禁开始跑回合（02:57，PID 58061），旧失败点为 adapter 创建（<5 s 即失败），现已进入 traces 阶段。
  - 接续脚本 `fix_eval_resume.sh`：dev 只跑 P8/P9/P10/P11（bc、P6 复用首轮结果），val 跑六策略，随后 strata_dev/strata_val/analyze。

### 第二次评价中断与接续（2026-11-02 06:15 → 06:31）
- **dev 六策略全部完成**：`00_tests` rc=0（24 passed）；dev_P8/P9/P10/P11 rc=0（03:52/04:37/05:27/06:14）。
  `eval_dev/full/{bc,P6,P8,P9,P10,P11}` 各 **420 commits + 420 traces + summary.json（records=420）**，六策略合计 2520 条。
- **`val_bc` rc=1（06:15）**：`FileNotFoundError` —— 冻结 val bank 内嵌 20 个**绝对** `worlds[uid].obstacle_path`
  指向已被删除的 `program/results/routea_xk_run01/val_bank/obstacles/…`。失败在第一个回合之前（`eval_val/full/bc` 只有 `policy_provenance.json`）。
- **处置（不扩大任何允许版本、不改 checkpoint）**：仅建符号链接
  `program/results/routea_xk_run01/val_bank -> Improved-3D-Diffusion-Policy-main/jindu/routea_xk_run01/val_bank`，
  字节未变；复验 20/20 obstacle 的 sha256 等于 bank 记录的 `obstacle_sha256`，`content_hash=a2b28e0c…` 不变。
- **失败现场归档**：`$OUT/evaluation_gate_fix_v1/val_bc_failed_before_episode/`（rc/log/command.txt/空输出目录/ARCHIVE_NOTE.txt），
  归档前证据守卫确认 commits=0、traces=0。
- **接续脚本 `fix_eval_resume_val.sh`**（06:31 setsid 分离启动，PID 62815/62819）：先做 sha256 校验 + val bank 20/20 障碍物 hash 前置门
  （实测 `obstacles_bad=0`），再跑 val 六策略，随后 `strata_dev`/`strata_val`/`analyze`。
  不重跑 dev（六份 dev 结果已完整），不重跑任何已成功的 val 之外的步骤。
- **进度**（06:32）：`val/bc` 已在产回合（4 commits / 5 traces，80 条目标），此前同一策略在 <5 s 内就失败——说明路径修复生效。

### 评价与分析收官（2026-11-02 07:36）
- **val 六策略全部完成**：`val_bc/P6/P8/P9/P10/P11` 全部 rc=0（06:31 起跑，07:33 前结束），各 **80 commits / 80 traces / summary records=80**。
- `strata_dev` / `strata_val` / `analyze` 全部 rc=0：`dev_strata.json`（07:33）、`val_strata.json`（07:34）、`analysis.json`（07:36，7.3 MB）。
- **总分母 3000 条**：dev 2520（6×420）+ val 480（6×80），与指导预期一致。
- **独立复核（逐 commit 用 `load_commit` 读原始证据）**：dev 与 val 全部 12 个 (split, policy) 组合，
  `bank_hash` 不符 0、`initial_hash` 配对不符 0、`use_base_policy` 分支不符 0、`status!=valid` 0、`terminate_at_space_limits` 0。

| policy | dev reached/collision/time_limit | val reached/collision/time_limit |
|---|---|---|
| bc  | 110 / 150 / 160 | 29 / 16 / 35 |
| P6  | 203 / 143 / 74  | 42 / 14 / 24 |
| P8  | 144 / 143 / 133 | 36 / 21 / 23 |
| P9  | 170 / 159 / 91  | 33 / 18 / 29 |
| P10 | 152 / 155 / 113 | 34 / 20 / 26 |
| P11 | 161 / 158 / 101 | 31 / 20 / 29 |

**analysis.json 事实（执行端只报数，判读归 Codex）**
- `verdict = adoption_criteria_not_met_both_seeds`；`protocol=#E2`。
- `P8_vs_P9`：`criteria_not_met`；`dev_arrival_delta=-26`、`high_arrival_delta=0`、`val_arrival_delta=+3`、
  `zero_band_loss_delta=0`、`dev_collision_delta=-16`、`high_collision_delta=-10`。
- `P10_vs_P11`：`criteria_not_met`；`dev_arrival_delta=-9`、`high_arrival_delta=+7`、`val_arrival_delta=+3`、
  `zero_band_loss_delta=+5`、`dev_collision_delta=-3`、`high_collision_delta=-1`。
- `F3_seed_difference`（P9 为 treatment / P11 为 reference，dev420，420 对，21 世界）：`delta_pp=+2.1429`，
  95% CI `[-3.5714, +7.6190]`，rescued 33 / lost 24，bootstrap 20000 次 seed 2026102601。
- `historical_F2_range='unavailable under common #E2; do not substitute historical #E1'`。

### 回传包（2026-11-02 07:48）
- 位置 `/home/cjj/AUV/wsl_snapshot/routea_f5_run01/`，**3.9 G**。
- `results/` = `$OUT` 全目录（**6424 文件**，与 OUT 去掉 `.pt` 后逐一相等）；`*.pt` 与 `state.pt` **0 个**。
- `weights_evidence/{P8,P9,P10,P11}`：各 9 个 checkpoint 的 `header.json`/`SHA256.json`/`COMMITTED.json`，共 **108 文件**，无 `state.pt`。
- `execution_records/`：`fix_manifest.md`、`blocker_eval_v3_gate.md`、`task_plan.md`、`diagnose_p8_env.py`、
  `fix_eval_resume.sh`、`fix_eval_resume_val.sh`、`step_04_eval.sh`、三份 master log/rc。
- `results/evaluation_gate_fix_v1/`：修复前/后 helper + 三份 sha256 + diff、新测试 sha256、全部 rc/log/command、
  `val_bank_path_fix.log`、`P8_failed_before_episode/`、`val_bc_failed_before_episode/`。
- **自检**：`return_sha256.txt` 6549 行，`sha256sum -c` **6549 OK / 0 FAILED，VERIFY_RC=0**。
  （首次构建时清单把自身写进了清单导致 1 行 FAILED；改为先写 `/tmp` 再 `mv` 后重生成，包内容未变。）

### 环境风险处置：VAL_BANK 路径修复（2026-11-01 18:30）
- 现象：`program/results/routea_xk_run01/` 整个目录在磁盘清理中被删除，`VAL_BANK` 指向的 `val_bank/bank.json` 不存在；`wsl_snapshot/` 也已清空。
- 现存活副本：`Improved-3D-Diffusion-Policy-main/jindu/routea_xk_run01/val_bank/`（xk 回传包副本），含 bank.json + initial/(80 npz) + obstacles/(80 json)。
- 身份核验：`canonical_hash(bank 去掉 content_hash) = a2b28e0cf2ca32ee…` 与记录值一致（MATCH）；file sha256=3dbe2fcb…；80 任务 / 20 世界 / split=rl_val / repeats=1；80 个 `initial/*.npz` 引用全部存在。
- 处置：仅改 `env.sh` 的 `VAL_BANK` 指向现存副本（加注释），未改任何内容。
- **为什么必须提前修**：`scripts/run_routea_xk_evaluation.py:361` 是 `read(bank_path) if bank_path.exists() else prepare_fixed_bank(...)` —— 路径不存在时会**静默重新生成**一份新 bank 而不是报错，违反指导"只用已有输入、不重采集"。已消除该风险。
