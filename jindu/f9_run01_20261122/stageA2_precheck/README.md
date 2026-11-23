# F9 阶段 A2 预检包（教师被挡任务采样器）

日期：2026-11-23。按《F9 阶段 A2 方案与 WSL 执行指引 v1_0》§三/§四执行，**未训练**，等待 Claude 放行 §五。

## 内容

| 路径 | 内容 |
|---|---|
| `code/a2_sampler.patch` | 4 个文件的统一差异（新增 sampler + f9_support + run_routea_f9 + 测试） |
| `code/a2_sampler_change.md` | 逐文件、逐函数的改动说明与前后 SHA256 |
| `code/before_manifest.json` | 改动前原件清单（原件在 `program/code_backups/20261123_f9_a2_before/`） |
| `probes/a2_probe.json` | §四 探针 2/3/4/5 的原始输出 |
| `probes/a2_probe.log` | 探针完整终端日志，rc=0 |
| `probes/a2_unit_tests.log` | `test_f9_online.py` 15 项（13 原有 + 2 新增）全通过 |

## 探针结果摘要

- 单元测试：**15/15 OK**（新增 2 项：default 路径保持原采样器；blocked 的筛选/库排除/常量）
- 筛选正确性（探针 2）：200 个被挡任务 **全部 < 0.3 m**，净空 min/median/max = −1.0854 / 0.0000 / 0.2992；
  接受率 **0.2367**（845 个候选）；单任务最多 21 次被拒（上限 500）；逐个用 `collision_geom`
  重算与记录值完全一致
- 反例（探针 2b）：原生成器 200 次抽样中 **115 次**判“不挡”（≥0.3 m），例：1.41 / 11.18 / 1.55 m
- 与库任务不重叠（探针 3）：200 个中重叠 **0**；把库任务本身送进判定 → **被拒绝**
- 随机数隔离（探针 4）：采样前后全局 torch/numpy RNG 逐位不变，PPO 采样器状态逐位不变
- 默认路径不变（探针 5）：`default` 模式重放教师任务序列与 P28 记录的前两回合 **完全相同**

## 需要 Claude 注意的一处口径

§四.4 字面要求“同一 actor 与采集器状态下 default/blocked 各跑一次 PPO 采集、前 1024 条转移逐位相同”。
预检做的是等价的**机制级**验证：教师采样只用自有 RNG，不触碰全局 RNG 与 PPO 采样器状态。
端到端的 1024 条转移逐位比较由 pipeline 在 P35 第 257 轮自动执行（不同即阻断训练）。
若要求在预检阶段就做端到端版，需额外跑两次 PPO 采集（各约 6 分钟），可另行安排。

## 身份

P35 已在 `RUNS` 登记（stage A、seed 202611081），`configure --policy P35` 会写入
`f9.teacher_task_sampler: blocked`。本包**不含**任何 P35 配置或权重。
`plan.json` 的 `source_files` 已同步（216 份，新增 1、改 2），`sources()==plan`。
