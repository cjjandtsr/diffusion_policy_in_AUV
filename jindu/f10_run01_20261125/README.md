# F10 第 1–3 步：评价入口、探针与正式评价回传（不作判定）

日期：2026-11-25（America/Chicago）。执行端：WSL。依据：《F10 去噪内碰撞引导系统评价 WSL 执行指引 v1_0》第 1、2 步。
本目录含：第 1 步代码与探针（§1–§3）、复用三组 1 m/0.5 m 双口径（§4）、
第 3 步正式评价 dev420/val20 的 S1（P6 g1）与 S4（BC g1）（§8–§10）。Q1、Q2 由 Claude 用原迹复算。

## 1. 代码

| 项 | 值 |
|---|---|
| 分支 | `f10-code-20261125`（基于 `f9-code-a2-20261124` = `9801780`） |
| `scripts/run_routea_f10_eval.py` | `f54adbbe91da771b5ea0e236016b2aa7b75af2fa5d968f710f9420fa683f908b` |
| `scripts/f10_chain_probe.py` | `b525f33a17e74cf4ca833c1900608376cbfb19d9b08df652ce89f41244fe6d39` |
| `scripts/f10_compare.py` | `fb3e7e66f85e533aced9ce0925d54745f0baeffe921c46d6d0927dabbef28a19` |
| 引导脚本（未改） | `scripts/routea_denoising_guidance.py` `8fa3b263c6d6ae830405af702c8dbda2448ad4979af55487b999a75937b31347` |
| 权重 | `checkpoint_update_000256`，`state.pt` `a91c9f42…dd43`，runtime identity `d987235a…b61f` |
| g1 / G0 | `d_safe 0.6, eta 0.4, M 4, consistency 0` / `d_safe 0.6, eta 0.2, M 2, consistency 0`（与 `f9_online.py:138` 的调用形状一致：consistency 恒为 0） |

`request.json` 固定了 16 个受控文件的 SHA256、两个银行的 SHA256/content_hash 与 P6 回执；探针期间任何受控文件变动都会让入口直接失败（已由 `load_plan()` 强制）。

## 2. 探针命令与耗时

全部在 dev420 的**前 10 个条件**上跑（`--limit 10`），单进程；每条约 40 s 是严格身份恢复的固定开销。

| 探针 | 命令（节选） | 回合 | 耗时 |
|---|---|---|---|
| T1 基线 | `worker --policy P6 --guidance none --split dev --limit 10 --tag T1_none` | 10 | 90 s |
| T1 η=0 | `worker --policy P6 --guidance g1 --eta 0 … --tag T1_g1_eta0` | 10 | 90 s |
| T4 | `worker --policy BC --guidance none … --tag T4_BC_none` | 10 | 106 s |
| T5 | `worker --policy P6 --guidance G0 … --tag T5_P6_G0` | 10 | 93 s |
| T6 | `worker --policy P6 --guidance g1 … --tag T6_P6_g1` | 10 | 96 s |
| T2 | `f10_chain_probe.py --condition 3 --guidance g1 --decisions 60` | 1 回合 / 55 决策 | 见 `execution/T2_chain.log` |
| T7 | `rejudge --job runs/G0_dev --bank dev420` | 420 | 15 s |

完整 argv 在 `execution/*.command.txt`，日志与退出码在 `execution/*.log`、`execution/*.rc`（全部 rc=0）。

## 3. 探针结果

### T1（正例：η=0 与 none 逐位相同）**通过**

`comparisons/T1.json`：10/10 条件 **每个决策的全部输出逐位相同**（`full_action_16x11`、`policy_action_8x11`、
`executed_action_4x11`、`candidate_h4_path`、`executed_path`、`planned_policy_path`、`post_state`），
动作种子相同，1 m 结局相同（10 reached）。
更强的证据：η=0 这一臂里仍有 **3 个决策进入了更新路径**（`triggered_denoise_steps` 非空，共 36 次更新），
输出仍逐位不变——即“触发但零步长 ⇒ 不动”成立，不是“没触发”。

### T2（反例：只有后 3 步不同）**通过**

`t2_chain.json`（条件 3，55 个决策）：只有 2 个决策触发引导。

| 决策 | 触发的去噪步 | 首个不同的链下标 | 链长 | 代价用点 |
|---|---|---|---|---|
| 22 | 7, 8, 9 | **8** | 11 | 2029 |
| 23 | 8 | **9** | 11 | 2041 |

链条是 `[x_init, x_1, …, x_10]`（`ft_denoising_steps = ddim_steps = 10` 时 11 项）。决策 22 的链 0–7 逐位相同、
8–10 不同 ⇒ **前 7 步 x_t 完全一致，只有后 3 步被改**；决策 23 只在第 8 步触发，链 0–8 相同、9–10 不同，
方向一致。未触发的 53 个决策两链逐位全同。

### T3（代价取点只来自当前观测）**通过**

`t3_cloud.json`：718 个决策中 182 个的代价用点非空；点数中位数 1063.5、p95 2293.3、最小 16、最大 2622。
来源字段固定为 `observation.point_cloud[-1]`，经 `body_to_world(cloud, drone_pos, yaw)` 转到世界系后按 `z >= 0.1` 过滤。
`scripts/routea_denoising_guidance.py` 的 `costs()` 只读 `self.cloud`，该对象只在 `Guidance.begin()` 里由当前观测构造；
入口没有把世界几何、障碍表或任务数据传给引导。

### T4（BC none 复现 #E2 原迹）**通过**

参照：仓库 main 的 `jindu/routea_f5_run01/results/eval_dev/full/bc`（同 10 个条件）。
`comparisons/T4.json`：10/10 条件逐决策输出逐位相同、动作种子相同、1 m 与 raw 结局相同（1 m 5 reached / 5 time_limit）。

### T5（G0 复现探针原迹）**通过（且逐位相同）**

参照：`routea_probe_e3g_run01/runs/G0_dev`（同 10 个条件，引导脚本 `41c3d47c`）。
`comparisons/T5.json`：10/10 条件逐决策输出**逐位相同**、结局相同（10 reached）。
说明：`8fa3b263` 与 `41c3d47c` 的差异只是把 `d_safe`、`updates` 参数化（默认 0.6 / 2），默认参数下两级实现位级等价。

### T6（计时）**通过**

`t6_timing.json`（每个决策的 `denoise_seconds`，`forward` 前后 `torch.cuda.synchronize()`）：

| 臂 | 决策数 | 去噪中位数 | 去噪 p95 | 推理全程中位数 |
|---|---|---|---|---|
| P6 none | 358 | **212.1 ms** | **271.6 ms** | 227.1 ms |
| P6 g1 | 359 | **228.2 ms** | **287.8 ms** | 241.3 ms |
| P6 G0 | 359 | 218.1 ms | 273.0 ms | 231.5 ms |

前 10 个条件里 g1 只触发 2 个决策、G0 也只触发 2 个，所以两臂的差异不是引导开销的主体；
触发决策本体会多跑一次基线链（为了 `delta_exec`），因此**臂间总耗时不能当作引导代价**，
T6 只用于确认“有无引导都在同一量级、没有量级退化”。

### T7（1 m 重判的两个边界）**通过**

`t7_rejudge.json`（对 G0_dev 的 420 个原迹重判，入口未参与）：

| 情形 | 例 | 原迹 raw | 1 m 重判 |
|---|---|---|---|
| 先入 1 m 后碰撞（应记成功） | `cul_de_sac_trap__000036__cfg01` repeat 4 | collision | **reached**（最近 0.531 m） |
| 入 1 m 前碰撞（应记碰撞） | `bottleneck_corridor__000016__cfg03` repeat 1 | collision | **collision**（最近 14.84 m） |

G0_dev 全量重判：reached_1m **300**、raw reached **216**、collision 122 → 其中 4 个先入 1 m 后碰撞；
与探针 `summary.json` 的 300/118/2 完全一致，说明 1 m 判定函数与原迹读取一致。

## 4. 复用三组的 1 m / 0.5 m 双口径（`reuse_outcomes.json`）

0.5 m 一列是原迹里环境的 `done_reason`（训练到达半径 0.5 m），1 m 一列是用论文线 `outcome_1m`
从同一批原迹重判（`f7_analysis.outcome_1m`，raw reached 或连续执行路径最近点 < 1 m）。

| 组 | 条件 | 1 m：到/撞/超 | 0.5 m：到/撞/超 | 原迹 |
|---|---|---|---|---|
| S0 P6 #E2 dev420 | 420 | **281 / 138 / 1** | 203 / 143 / 74 | `routea_p6_u256_e2_dev420_run01_restored/full/P6`（仓库 main：`jindu/routea_p6_u256_e2_dev420_run01/full/P6`） |
| S0 P6 #E2 val20 | 80 | **61 / 14 / 5** | 42 / 14 / 24 | `routea_p6_u256_e2_val20_run01/full/P6`（仓库 main：`jindu/routea_f5_run01/results/eval_val/full/P6`） |
| S2 P6 G0 dev420 | 420 | **300 / 118 / 2** | 216 / 122 / 82 | `routea_probe_e3g_run01/runs/G0_dev` |
| S2 P6 G0 val20 | 80 | **59 / 14 / 7** | 41 / 14 / 25 | `routea_probe_e3g_run01/runs/G0_val` |
| S3 BC #E2 dev420 | 420 | **156 / 141 / 123** | 110 / 150 / 160 | `jindu/routea_f5_run01/results/eval_dev/full/bc` |
| S3 BC #E2 val20 | 80 | **39 / 16 / 25** | 29 / 16 / 35 | `jindu/routea_f5_run01/results/eval_val/full/bc` |

S0 两行与 Claude 的独立复算（dev 281/138/1、val 61/14/5）逐项一致。

## 5. 偏离本指引之处

1. **T2 的条件不在“前 10 个”之外**：条件 3（`ac9ae3eec09720b3d7d029d6`）就在前 10 个内，且它在前 10 个里唯一触发；
   但触发发生在第 22–23 个决策，所以 `f10_chain_probe.py` 跑满 60 个决策而不是 10 个。
2. **T6 的“有无引导各 100 个决策”用臂级比较**（none 358 个决策 vs g1 359 个决策），因为前 10 个条件里
   g1 只有 2 个决策真正触发；同时保留 `guided/unguided` 分组统计（见 `t6_timing.json`）。
3. **T4/T5 的 `trace_sha256` 不相等**：新入口在每条 transition 里多了 `probe_metrics`（计时、delta_exec、
   点数），而 F5 原迹没有；两条原迹的**决策输出**仍逐位相同。比较脚本因此以逐决策输出为准，
   并把整条 JSONL SHA 一并记下备查。
4. **T1/T4/T5 逐位比较的字段集合**：`full_action_16x11`、`policy_action_8x11`、`executed_action_4x11`、
   `candidate_h4_path`、`executed_path`、`planned_policy_path`、`post_state.position`、`action_seed`、`decision`。
5. **186 银行未跑**（按 2026-11-25 决定留到最后一次揭示），本目录只含 dev420 前 10 个条件的探针。

## 6. 等放行

第 3 步按新口径只剩 dev420（S1、S4）+ val20（S1、S4），预计约 4.2 h（dev420 两臂约 3.5 h、val20 两臂约 0.7 h）。
收到“放行”后开始，逐臂落 `runs/<label>_<split>/` 并用同一 `collect()` 生成 `conditions.csv`/`summary.json`。

## 7. 附：论文 177/186 任务口径的运动学结果来自哪里（顺带查，只报位置）

第 0 步说“P6/BC 在 186 银行上没有回合”，指的是**论文规划器 186 任务银行**（`paper_186_v1_0_run01/input/bank.json`，
`d34c10d6…`）。论文正文的“186 任务”运动学表是另一个池子，由三份银行拼成，**P6/BC 在其中是有回合的**：

| 组件 | 任务数 × 重复 = 回合 | 运行（P6/BC 原迹） | 银行 content_hash |
|---|---|---|---|
| test12 | 48 × 5 = 240 | `program/results/routea_paper_g3_e2_run01/eval_test/{bc,P2,P4,P6}`（另有 `routea_test12_independent_run02`） | `b3a872bb…` |
| retained_run02 | 60 × 5 = 300 | `program/results/routea_paper_c5_run02/full/{bc,P2,P4,P6}` | `0b64e3ed…` |
| added（ext_v2） | 91 × 5 = 455 | `program/results/routea_final_existing_e2_run01/full/added/{bc,P2,P3,P4,P6}`（`…run04` 是 bc 的重跑，`…run05` 是 bc 的 33 任务部分批） | `d65382bf…`（extension_v2 `b45d6cf7…`） |

- 三者合计 **199 任务 / 995 回合**（Claude 在 `routea_final_Oprime_review/routea_final_Oprime_Claude_review.json` 复核过
  `component_counts = {test12:240, retained_run02:300, added:455}`、分类 59/70/70、隔离违规 0）。
- 论文侧聚合脚本：`program/paper_186_geometry_20261110_v1_0/paper_snapshot/研究报告/analysis_20261108/kin_task_level.py`
  （输入 `codex_details/tasks_199.csv` + 一份 `rows.json`），输出 `oprime_task/main.csv`（**186 任务**：56/64/57→57/67/62）与
  `main_05m.csv`。
- 199 → 186 的 13 个去除任务写在同目录 `oprime/exclusions.json`（9 个 cul_de_sac 双失败 + 4 个 yaw 场景几何与训练重合）。
  **177** 是更早一版口径（F3 配图卡：56/64/57），对应 199 去掉 22 个任务；本地没有单独的 177 运行，
  它是同一批回合的不同去除规则聚合出来的，不是另一次评价。

因此“论文 177 任务上的被挡结果（BC 35、P6 55）”的底层原迹就是上表三处，聚合中间产物在
`program/paper_186_geometry_20261110_v1_0/paper_snapshot/研究报告/analysis_20261108/oprime_task/`。

## 8. 第 3 步正式评价（dev420 的 S1、S4；val20 的 S1、S4）

驱动脚本 `execution/formal_driver.sh`，单进程顺序跑；每条命令的完整 argv、日志、退出码、起止时间在
`execution/S{1,4}_*.{command.txt,log,rc,start,end}`。

| 臂 | 划分 | 条件 | 退出码 | 耗时 |
|---|---|---|---|---|
| S1 P6 g1 | dev420 | 420 | 0 | 61 min |
| S4 BC g1 | dev420 | 420 | 0 | 66 min |
| S1 P6 g1 | val20 | 80 | 0 | 13 min |
| S4 BC g1 | val20 | 80 | 0 | 12 min |

合计 **152 min**。四个臂合计 1000 个回合、35,365 个决策。

完整性（`report.json` → `integrity`）：四个臂条件数 420/420/80/80、唯一条件数相同、重复 0、缺失 0、rc 全 0。
每臂另核对：`policy`、`use_base_policy`、银行 content_hash、`reset_seed`、`initial_hash`、`task`、`geometry`、
`obstacle_sha256` 与银行逐条一致（`collect()` 内断言）；`guidance.wrapper_sha256` 四臂均为 `8fa3b263…`。

## 9. 结果

### 9.1 1 m 主口径（论文线 `outcome_1m`，从原迹重判）

| 臂 | 划分 | 到达 | 碰撞 | 超时 |
|---|---|---|---|---|
| S0 P6 #E2（复用） | dev420 | 281 | 138 | 1 |
| **S1 P6 g1** | dev420 | **304** | **114** | **2** |
| S3 BC #E2（复用） | dev420 | 156 | 141 | 123 |
| **S4 BC g1** | dev420 | **170** | **108** | **142** |
| S0 P6 #E2（复用） | val20 | 61 | 14 | 5 |
| **S1 P6 g1** | val20 | **62** | **11** | **7** |
| S3 BC #E2（复用） | val20 | 39 | 16 | 25 |
| **S4 BC g1** | val20 | **42** | **12** | **26** |

### 9.2 0.5 m 口径（原迹里的环境 `done_reason`，训练到达半径）

| 臂 | 划分 | 到达 | 碰撞 | 超时 |
|---|---|---|---|---|
| S0 P6 #E2（复用） | dev420 | 203 | 143 | 74 |
| **S1 P6 g1** | dev420 | **221** | **114** | **85** |
| S3 BC #E2（复用） | dev420 | 110 | 150 | 160 |
| **S4 BC g1** | dev420 | **122** | **111** | **187** |
| S0 P6 #E2（复用） | val20 | 42 | 14 | 24 |
| **S1 P6 g1** | val20 | **41** | **12** | **27** |
| S3 BC #E2（复用） | val20 | 29 | 16 | 35 |
| **S4 BC g1** | val20 | **31** | **12** | **37** |

### 9.3 逐臂 delta（引导 − 同策略 #E2，仅算术差，不判定）

| 臂 | 1 m Δ到/撞/超 | 0.5 m Δ到/撞/超 |
|---|---|---|
| S1 P6 dev420 | +23 / −24 / +1 | +18 / −29 / +11 |
| S4 BC dev420 | +14 / −33 / +19 | +12 / −39 / +27 |
| S1 P6 val20 | +1 / −3 / +2 | −1 / −2 / +3 |
| S4 BC val20 | +3 / −4 / +1 | +2 / −4 / +2 |

## 10. 引导统计与计时

| 臂 | 决策数 | 触发决策 | 触发比例 | 有引导的条件 | `delta_exec` 最大 | `delta_exec` 总和 | 去噪中位/ p95 |
|---|---|---|---|---|---|---|---|
| S1 P6 dev420 | 12,280 | 1,698 | 13.8% | 204 / 420 | 1.204 m | 742.26 m | 223.8 / 257.0 ms |
| S4 BC dev420 | 17,119 | 1,807 | 10.6% | 212 / 420 | 1.411 m | 777.38 m | 168.3 / 203.1 ms |
| S1 P6 val20 | 2,684 | 163 | 6.1% | 27 / 80 | 1.012 m | 66.20 m | 227.2 / 273.5 ms |
| S4 BC val20 | 3,282 | 212 | 6.5% | 31 / 80 | 1.212 m | 96.39 m | 161.3 / 192.3 ms |

`delta_exec` 是逐决策的“基线链 vs 引导链前 4 个执行航点最大世界系偏差”（F9 定义）；未触发时为 0。
计时是 `forward` 前后 `torch.cuda.synchronize()` 的 DDIM 环耗时，按条件取中位数后再统计。

## 11. 回传内容与校验

| 路径 | 内容 |
|---|---|
| `request.json` | 16 个受控文件 SHA256、两银行 SHA256/content_hash、P6 回执、g1/G0 设置 |
| `probes/` | T1/T4/T5/T6 的 10 条件探针（明文原迹，5 臂） |
| `comparisons/T1.json`、`T4.json`、`T5.json` | 逐决策输出逐位比较结果 |
| `t2_chain.json` | T2 的两条 DDIM 链逐步比较 |
| `t3_cloud.json`、`t6_timing.json`、`t7_rejudge.json` | 代价取点、计时、1 m 重判 |
| `reuse_outcomes.json` | 复用三组的 1 m / 0.5 m 双口径 |
| `runs/<arm>/{identity.json,summary.json,conditions.csv}` | 每臂身份、结局汇总、逐条件表（明文） |
| `packs/<arm>_traces.tar.zst` | 每臂 `commits/` + `traces/` 全量原迹（2000 个文件，压后 92 MB） |
| `traces_manifest.sha256` | 上述 2000 个原迹文件的逐文件 SHA256（与 tar 内一致） |
| `report.json` | 每臂完整性、结局、引导与计时统计 |
| `execution/` | 所有命令 argv、日志、rc、起止时间 |
| `SHA256SUMS.txt` | 本目录（不含 `runs/*/traces`、`runs/*/commits`，它们在 tar 里）逐文件 SHA256 |

校验方式：`sha256sum -c SHA256SUMS.txt`；解包 `packs/<arm>_traces.tar.zst` 后 `sha256sum -c traces_manifest.sha256`
（manifest 里的路径是 `runs/<arm>/…`，需在 `runs/` 目录下执行或先 `cd runs`）。

## 12. 未做

- 没有跑 186 银行与 test12（按 2026-11-25 决定留到最后一次揭示）。
- 没有改任何权重、银行、判定函数；本轮没有新增策略编号。
