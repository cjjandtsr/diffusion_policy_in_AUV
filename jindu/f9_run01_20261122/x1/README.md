# F9 诊断 X1：动作空间吸收量

日期：2026-11-24。依据《F9 诊断 X1 动作空间吸收量 WSL 执行指引 v1_0》与 2026-11-24 的两条裁定
（七.1 按"底层一致"判通过并把 g 改为 `normalize(out['full'])`；七.4 按臂各自 45/58 条记录复算）。

## 结论状态

- §七 七项核对**全部通过**：η=0 逐位相同（max 差 0.0）、同种子重复一致、条件差 0、
  P6 记录损失对 P35/P36 均差 0.0、两树 P6 逐位相同、代数恒等式（最大 1.07e-08）。
- 本包**不作判定**；Q1、Q2 由 Claude 依据原始量复算。

## 内容

| 文件 | 内容 |
|---|---|
| `f9_x1_absorption.py` | 脚本（`af9708c4…57db`） |
| `x1_states.json` | U = 83 状态的清单（序号、sample_id、observation_sha256、来源子集） |
| `x1_raw_run甲.pt`、`x1_raw_run乙.pt` | 逐状态、逐策略的 b、g、delta；P6 的 s_k、h_k 与 5 种目标损失 |
| `x1_summary.json` | §六 全部量：整体 + P28/P35/P36 三个来源子集 |
| `x1_checks.json` | §七 核对结果（含 bootstrap 区间、两树 P6 一致性、逐臂损失复现） |
| `x1_arm_loss_P35.json`、`x1_arm_loss_P36.json` | 按臂各自记录复现 prefix_worker 的 P6 损失 |
| `x1_run甲.log`/`.rc`、`x1_run乙.log`/`.rc`、`x1_arm_loss.log`、`x1_compare.log`/`.rc` | 运行日志与退出码 |
| `x1_summary.log` | 三段耗时与合并摘要 |
| `SHA256SUMS.txt` | 上述文件哈希 |

## 复算须知

- 甲（当前树）里 **P35 的 checkpoint 无法在 A2 后的代码下加载**（IdentityMismatch）：P35 训练时的
  `scripts/run_routea_f9.py` 是 `05cc0d0c…8457`，磁盘上已无该版本。用已上传的 stageA2 回传包
  （git `e4943d2f` 内 `a2_all.patch`）把交付版打回该版本，重建 `tree_P35` 后 216 条源码身份与
  P35 的评测记录逐条相同，才成功加载。该树未随包回传（667M）。
- 乙使用阶段 A 树：`sources()` 215 条与 `evaluations/P29/u000264/train_shadow/result.json` 逐条相同。
- 共享状态（32 个）的冻结目标与噪声按 §三取 P35 那一份；P36 有 2 条共享记录的冻结量与 P35 不同，
  故七.4 改为按臂各自复算（结果差 0.0）。
