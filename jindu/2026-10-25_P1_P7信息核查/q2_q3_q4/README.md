# Q2/Q3/Q4 数据交付说明（2026-10-25）

本目录是对 `2026-10-25_P1_P7信息核查` 的补充：Q2/Q3/Q4 三项的客观原始数据。
只提供数据与文件定义，不含统计、判断或结论。

## Q3：E5 checkpoint normalizer 全 11 维读出

- 文件：`q3_e5_normalizer_result.json`
- 来源 checkpoint：`data/outputs/2026.08.16/08.53.26_e5_lambda0p5/checkpoints/latest.ckpt`
  （SHA256 = `c1fca4ac2735928b18fa7a9c069f90ad181eb84092acee379b073dc6a61bfc9a`，与 P2 绑定值一致）
- 读取方式：只读 `torch.load(map_location='cpu')`，未修改任何权重/文件；键前缀 `normalizer.params_dict.action.*`
- 与 P2 已有 4 个值的交叉验证：scale[0]/scale[1]/scale[2]/offset[9] 全部一致（见 JSON 内 `p2_cross_check`）

action scale（11 维）：

| 通道 | scale | offset | input_std |
|---|---:|---:|---:|
| 0 | 4.2736897468566895 | −0.1184377 | 0.2339898646 |
| 1 | 4.796350955963135 | −0.0121713253 | 0.208491832 |
| 2 | 27.20646095275879 | −0.0147005673 | 0.0367559642 |
| 3 | 9.731785774230957 | 0.0050986158 | 0.1027560532 |
| 4 | 10.268287658691406 | 0.0005289478 | 0.0973872244 |
| 5 | 23.666271209716797 | 0.0009131068 | 0.0422542207 |
| 6 | 3.234375238418579 | 0.0046621761 | 0.30917871 |
| 7 | 3.37571120262146 | 0.0006898291 | 0.296233803 |
| 8 | 8.363972663879395 | −0.0014079089 | 0.1195603982 |
| 9 | 5.873570919036865 | 0.0002764771 | 0.1702541858 |
| 10 | 1.0812376737594604 | 0.0021327771 | 0.9248660207 |

（P2 将通道 9 标为 Δyaw，依据为 offset 匹配；通道 10 的 input_std=0.925 rad 与
通道 9 的 0.170 rad 同时给出，通道语义归属由接收方自行判定。通道 0-2 为 world Δx/Δy/Δz。）

## Q4：非密集、起点直连通畅组 300 条逐决策数据

- 文件：`q4_per_decision_300.csv`（11807 行）、`q4_group_classification.json`
- 来源：`routea_common_a1_v1_run01/full/{bc,rl_391a6656a0f7_u128,rl_391a6656a0f7_u256}/traces/`
  各 420 条 = 84 任务 × 5 repeat，共 1260 条已验收原迹
- 分组判据（与 P4 同源）：`role == 'dense_recovery'` 定义密集；
  `Scene.clear(start_pos, goal_pos)`（0.10m 水平碰撞代理、垂直分离豁免、球解析点线距离）
  定义初始直连
- 分类验证（对 P4 表逐项核对，全部一致）：
  - 分组条件数：密集受阻 125 / 密集通畅 175 / 其余受阻 20 / 其余通畅 100
  - 各分组碰撞计数 BC/u128/u256：{90,31,10,0} / {109,23,10,8} / {105,27,10,17}
- CSV 字段：`policy, task_id, repeat, decision, pos_x, pos_y, pos_z, yaw,
  goal_x, goal_y, goal_z, done_reason`
  - 位置/偏航取自每条 transition 的 `pre_state`（决策前状态，world 系）
  - 目标取自 episode_start 的 `task.goal_pos`
  - 目标组条件数 = 100 × 3 策略 = 300 条轨迹

## Q2：同状态行为分散度原始采样（18 状态 × 20 次 × 3 策略 × 2 采样模式）

- 目录：`q2/`；扁平导出：`q2/q2_samples_flat.csv`（2160 行）
- 协议：复用 `routea_a1_mechanism_val20_run03` 的 same_state 机制
  - bank：`scripts/routea_paired_support/a1_mechanism_bank_v1.json`（18 冻结状态，draws=20）
  - 策略：BC = E5 latest.ckpt；u128/u256 = `routea_common_b2560_seed202609203_v1`
    `checkpoint_update_000128` / `checkpoint_update_000256`
  - 模式：deterministic（eta=0）/ random（eta=1），由 `_E5Policy.predict` 的
    `deterministic` 参数翻转实现（源码字节未改动，身份闭包不受影响）
- 初噪配对：`paired_seed(state_id, draw)` 在三策略与两模式间完全一致
  （同 seed → 同初始 x_T，由构造保证；跨模式核验 360/360 键一致，断言通过）
- 完整性：`q2/run_complete_001.json` status=complete；108 个状态文件
  （2 模式 × 3 策略 × 18 状态）；每次采样含 `draw, seed, initial_noise_hash,
  h4_pos(xyz), h4_yaw, h8_pos(xyz), h8_yaw, full_action_16x11, policy_input_hash`
  - H4/H8 = 冻结状态 pre_state 上累加 full 轨迹前 4/8 步（yaw 已 wrap 到 (−π, π]）
- 只含原始采样值，不含离散度/方向偏差等任何统计量

## 复现脚本

- `scripts/q2_run_routea_same_state_dispersion.py`：Q2 采样驱动（新增文件，
  未修改任何已跟踪源码）
- `scripts/q3_read_e5_normalizer.py`：Q3 只读提取
- `scripts/q4_extract_300.py`：Q4 分类与提取（含 P4 计数断言）
- `scripts/q2_export_flat_csv.py`：Q2 扁平 CSV 导出
