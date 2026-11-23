# F9 v1_3 阶段A 回传包（P29 控制臂 u264，2026-11-22）

## 这是什么

`routea_f9_run01` 到目前为止的**原始材料**：第0步六组、select、shadow-probe、P29（控制臂）
u257→u264 的逐轮 metrics、G-0 记录、失败原迹（仅日志/rc/试次 metrics，**不含权重**）、
源码身份与两处兼容补丁的 diff。

对应交付包 `F9_v1_3_code_v1_2.zip`（SHA256 `0973447a3612181bacf7377bc9734466bd88c5ac8016877278277d63cb2464a1`），
code/ 与 v1_1 逐字节相同。

## 目录

| 路径 | 内容 |
|---|---|
| `run/plan.json` | prepare 登记的输入与 215 份源码身份（含两处补丁后的哈希） |
| `run/selection.json` | 六组选择结果：`d_safe=1.2, eta=0.4, M=4`，`passed=true` |
| `run/step0/g0..g5/result.json` | 六组各 425 条件（g0 为旧弱组重放，逐条断言通过） |
| `run/pair_probe/` | 原生配对噪声 11 份逐位相同 + 噪声张量 |
| `run/shadow_probe/` | on/off 各 1 条件；除 `probe_metrics` 外逐字段相同 |
| `run/P29/metrics_update_000257..264.json` | P29 逐轮 metrics（每轮 20 次提交、1024 条转移审计） |
| `run/P29/{configured.json,config.yaml,fork_lineage.json,r1_plan_probe.json}` | 配置与 fork 血统 |
| `run/execution/` | 各阶段命令日志与退出码（rc） |
| `run/aborted/` | 5 次失败的原迹（日志/rc/试次 metrics），不含权重 |
| `run/g0.json` | G-0 记录：`recorded=true, passed=false, blocking=false`，逐项相等与最大差值 |
| `weights_identity/P29/checkpoint_update_000{257,264}/` | 仅 header.json / SHA256.json / COMMITTED.json |
| `code_identity/` | 交付包期望源码集、实际源码集、两处兼容补丁 diff |
| `reports/` | 部署与预检报告、兼容补丁记录、G-0 未通过证据 |

## 关键身份

| 项 | 值 |
|---|---|
| 交付包 | `F9_v1_3_code_v1_2.zip` `0973447a…64a1` |
| P6@u256 state | `a91c9f428c5749b8ca507168ea83f17eeb88cce4b4a03c6aeaea197b214bdd43` |
| P22 u257（G-0 对照） | state `b8137f7978d8d9a90d8160b22cac75d74bc3150af488bf4bd282cb6a7e59f3f8` |
| P29 u264 state.pt | `bee8de960058a3bf4a5b0d10c96447e8a8b7f760866623e3014bc8e4244d6fa9` |
| P29 runtime identity | `1c537b1a5f613070d5e4f8320fffb15ae16b944c59aedd82fb25e42623031410` |
| 兼容补丁 1（a2_fork 白名单） | `scripts/evaluate_dppo_uav_formal_endpoint.py` |
| 兼容补丁 2（f9 下 reward_identity_hash 可变） | v2 reward bundle 被 F9 改造，PPO 项未变 |
| 兼容补丁 3（G-0 记录化） | 只改流程控制，判据未放宽 |

## G-0 判读（如实记录，不放宽）

`recorded=true, passed=false, blocking=false`。critic / critic_optimizer / counters 逐位相同；
lora 最大绝对差 9.2e-07、actor_optimizer 1.15e-05；转移的数据面（观测、动作、old_log_prob、
advantage、provenance）逐位相同，差异集中在更新时重算的诊断字段
（`analytic_bc_reverse_kl_sum` 最大 0.212/50.4、`new_log_prob` 2.8e-04 等）。
成因：F9 的 actor 更新为两段式（PPO backward + keep `autograd.grad` 合并），P22 为一次 backward，
数学等价、浮点累加顺序不同。

## 尚未做

`measure`、`gate`、P28（处理臂）均未执行；P28 等待核对单里 P28 专属 A 级项复审。
权重只留在 WSL 的 `WEIGHTS/P29/checkpoint_update_000{257,264}`，本包只有 header/SHA/COMMITTED。
