# F8 采样探针 run02 — 完整执行报告（v1.2）

- 日期：2026-11-16 / 17
- 包：`F8_probe_code_v1_2.zip`，sha256 `966ff404…0f7164`（与指导一致）
- 驱动目录：`/home/cjj/AUV/program/f8_probe_code_20261116_v12`
- 结果目录：`/home/cjj/AUV/program/results/f8_sampling_probe_run02`
- 冻结：60 回合 / 211 状态 / 28 任务 / 5 不适用；`frozen_hash a6c2291d…d2285`，文件 sha256 `c46ff2d8…bb05de`
- 结论：**门禁通过 → 20 次计时 → P6/P0 full → summarize 全部完成，退出码均为 0。** 未改阈值/判据/采样集/权重/奖励，未覆盖任何原始结果。

## 1. 总览

| 阶段 | 结果 | 耗时 | 退出码 |
|------|------|------|--------|
| P6 gates（3 状态） | 通过，逐位匹配 | ~30 s | 0 |
| P0 gates（3 状态） | 通过，逐位匹配 | ~20 s | 0 |
| P6 timing20 | 20 次续飞 | 67.8 s | 0 |
| P6 full | 211 状态 / 1688 续飞 / 13715 样本 | 6157 s（1 h 43 m） | 0 |
| P0 full | 211 状态 / 480 续飞 / 13715 样本 | 3856 s（1 h 04 m） | 0 |
| summarize | 18 个分组 | < 1 s | 0 |

链：22:23:13 开始 → 00:06:17 P6 完成 → 01:11:04 P0 完成 → 01:11:05 汇总完成（总 2 h 48 m）。
`full` 内部重新执行了 3 状态门禁，P6、P0 均 `passed=true`，输入 hash 全匹配。

## 2. 计时（timing20，P6）

- 20 次续飞，`elapsed_s=67.77`，`seconds_per_flight_including_open_loop_and_replay=3.3885`
- `projected_2400_s=8132.4`（按名义 2400 次）
- 计时样本均为 3–6 决策即碰撞的短飞行，故该投影偏乐观；full 实测 P6 为 3.6477 s/次（均值 6.91 决策），P0 为 8.0331 s/次（均值 13.38 决策）。

## 3. 数据完整性审计（逐条流式核对）

| 检查 | P6 | P0 |
|------|----|----|
| 样本状态数 | 211 | 211 |
| samples 记录 | 13715（64 随机 + 1 确定性） | 13715 |
| 每状态 draw 0..64 齐全 | ✅ | ✅ |
| 每状态 `policy_input_hash` 唯一且 == 冻结 provenance | ✅ | ✅ |
| 每状态障碍 hash == 冻结 header | ✅ | ✅ |
| `genuine_backs` 与冻结一致 | ✅ | ✅ |
| 确定性样本 `pre_state` == 冻结 | ✅ | ✅ |
| **P6 与 P0 同状态输入 hash 完全相同** | 211/211 | 211/211 |
| flights | 1688（211×8，draw 0..7） | 480（60×8） |

P6 原始结局：`collision 1658 / reached 9 / time_limit 21`；1 m 判据后 `collision 1645 / reached 43`。
P0 原始结局：`collision 427 / time_limit 41 / reached 12`；1 m 判据后 `collision 427 / time_limit 39 / reached 14`。
（原始 9 个 reached 变 43 个 1m reached：部分碰撞前已进入 1 m 球，按既有 `outcome_1m` 判据计为 reached，符合规定。）

## 4. 汇总（`paired_probe_summary.json`；1 m 主判据，碰撞体 0.1 m）

p_plan = 每状态 64 个随机样本中完整 8 步规划无碰撞的比例，再对状态取中位数；
p_reach / p_nocol = 每状态 8 次续飞的 1 m 结局比例，再对状态取均值。续飞只统计 `genuine_backs` 含该偏移的状态。

| 分组 | 状态 | 续飞 | p_plan | p_reach | p_nocol | 1 m 结局 |
|------|------|------|--------|---------|---------|----------|
| P6/2/all | 60 | 60 | 1.000 | 0.035 | 0.035 | collision 463, reached 17 |
| P6/2/overflight | 31 | 31 | 0.984 | 0.008 | 0.008 | collision 246, reached 2 |
| P6/2/lateral | 29 | 29 | 1.000 | 0.065 | 0.065 | collision 217, reached 15 |
| P6/4/all | 57 | 57 | 1.000 | 0.024 | 0.024 | collision 445, reached 11 |
| P6/4/overflight | 28 | 28 | 1.000 | 0.004 | 0.004 | collision 223, reached 1 |
| P6/4/lateral | 29 | 29 | 1.000 | 0.043 | 0.043 | collision 222, reached 10 |
| P6/6/all | 50 | 50 | 1.000 | 0.018 | 0.018 | collision 393, reached 7 |
| P6/6/overflight | 27 | 27 | 1.000 | 0.000 | 0.000 | collision 216 |
| P6/6/lateral | 23 | 23 | 1.000 | 0.038 | 0.038 | collision 177, reached 7 |
| P6/8/all | 37 | 37 | 1.000 | 0.027 | 0.027 | collision 288, reached 8 |
| P6/8/overflight | 17 | 17 | 1.000 | 0.007 | 0.007 | collision 135, reached 1 |
| P6/8/lateral | 20 | 20 | 1.000 | 0.044 | 0.044 | collision 153, reached 7 |
| P0/2/all | 60 | 0 | 1.000 | — | — | 无续飞（P0 只在第 4 偏移续飞） |
| P0/4/all | 57 | 57 | 1.000 | 0.029 | 0.114 | collision 404, reached 13, time_limit 39 |
| P0/4/overflight | 28 | 28 | 1.000 | 0.045 | 0.143 | collision 192, reached 10, time_limit 22 |
| P0/4/lateral | 29 | 29 | 1.000 | 0.013 | 0.086 | collision 212, reached 3, time_limit 17 |
| P0/6/all | 50 | 0 | 1.000 | — | — | 无续飞 |
| P0/8/all | 37 | 0 | 1.000 | — | — | 无续飞 |

要点（只陈述数字，不选路线）：

- P6 在所有偏移上 `p_nocol` 与 `p_reach` 相等，因为 1 m 判据下 P6 没有 time_limit 结局（全为 collision/reached）。
- P0 第 4 偏移有 39 次 time_limit，`p_nocol`（0.114）明显高于 `p_reach`（0.029）——不与 1 m 球互斥的“未碰撞”包含了悬停到时限。
- P6 越顶子集在所有偏移上 `p_reach` 都低于侧绕子集（2/4/6/8：0.008 vs 0.065、0.004 vs 0.043、0.000 vs 0.038、0.007 vs 0.044）。
- P0 只有第 4 偏移有续飞：57 个有效状态 + 3 个短回合（back=4 映射到 decision 0）共 60 状态 480 次，符合“短回合不伪装成真正第 4 决策”。
- 60 回合中 7 个状态 `genuine_backs` 为空（极短回合），其 56 次 P6 续飞不进任何偏移分组，但仍在 `flights.jsonl` 中保留。
- 视场代理（`in_fov_approx`）仅按测地前方 3 m 水平/竖直角度判定，不含遮挡；P0/4 有 26/57 状态在代理视场内。

## 5. 兼容修复（唯一一处非包内文件）

v1.2 仍缺 E2 override 的同级 vendor 文件，gates 第一次运行即失败于此（`e5_runtime.py:86`）。已从 BASE 复制
`third_party/dppo_official_cc7234a/model/diffusion/diffusion_vpg.py`（sha256 `8f495b48…aac4`）到 DRIVER 同路径，
**未改包内任何文件**。中性依据：包内 `e5_runtime.py` 与 BASE 逐字节相同；BASE vs NATIVE vendor 的 103 个 `.py`
仅 2 个不同（`diffusion_vpg.py`、`e5_runtime.py`），其余 101 个相同；该文件仅 import 两个两侧相同的模块。
建议 v1.3 包补上此文件。详见 `logs/compat_fix_note.md`。

## 6. 工件

```
P6_gates/                     v1.1 式首次失败（缺 vendor 文件），仅 identity.json，原样保留
P6_gates_fix01/               gates.json + identity.json（通过）
P0_gates_fix01/               gates.json + identity.json（通过）
P6_timing20/                  summary.json, flights.jsonl(20), samples.jsonl(195), gates.json, identity.json
P6_full/                      summary.json, flights.jsonl(1688), samples.jsonl(13715), gates.json, identity.json
P0_full/                      summary.json, flights.jsonl(480), samples.jsonl(13715), gates.json, identity.json
paired_probe_summary.json     18 个分组汇总（权威）
paired_probe_summary.csv      同内容 CSV 便于查看
logs/                         全部命令、stdout/stderr、退出码、兼容修复说明、校验和
```

复现命令见 `logs/P6_full.log` / `P0_full.log` / `P6_timing20.log` 首行（完整路径参数），链式执行见 `logs/full_chain.log`。
