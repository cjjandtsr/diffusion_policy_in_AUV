# F8 采样探针 run02 — Windows 侧副本

- 本目录：`C:\Users\cjj\Desktop\IDP3\program\f8_probe_run02_20261117`
- 来源（WSL）：`/home/cjj/AUV/program/results/f8_sampling_probe_run02`
- 复制方式：`cp -a` 保字节；复制后对 `logs/checksums.txt` 全部 35 项执行 `sha256sum -c`，**全部一致**。
- 日期：2026-11-16 / 17

## 内容

| 路径 | 说明 | 大小 |
|------|------|------|
| `F8_probe_run02_report.md` | 完整执行报告（门禁、计时、审计、18 组汇总、边界声明） | 7 KB |
| `paired_probe_summary.json` | **权威汇总**：P6/P0 × 越顶/侧绕 × 偏移 2/4/6/8 | 6.5 KB |
| `paired_probe_summary.csv` | 同内容 CSV | 2.3 KB |
| `P6_gates_fix01/`、`P0_gates_fix01/` | 通过的 3 状态门禁（gates.json + identity.json） | — |
| `P6_gates/` | 首次失败尝试（包缺 vendor 文件），仅 identity.json，原样保留 | — |
| `P6_timing20/` | 20 次计时：summary / flights / samples / 门禁 | 1.4 MB |
| `P6_full/` | 211 状态、1688 续飞、13715 样本 | 113 MB |
| `P0_full/` | 211 状态、480 续飞、13715 样本 | 98 MB |
| `logs/` | 每条命令首行、stdout/stderr、退出码、链日志、兼容修复说明、校验和 | 340 KB |
| `_additive_for_v1_3/` | **不属于结果集**，给 v1.3 补包用（见下） | 20 KB |

## 原始数据（权威）

- `P6_full/samples.jsonl`：13715 行 = 211 状态 × (64 随机 + 1 确定性)
- `P6_full/flights.jsonl`：1688 行 = 211 状态 × 8 draw
- `P0_full/samples.jsonl`：13715 行（同上，P0 策略）
- `P0_full/flights.jsonl`：480 行 = 60 状态 × 8 draw
- 每行都带 `state_id`、`obstacle_sha256`、`policy_input_hash`，可回溯到 `inputs/frozen_geometry.json`（哈希 `a6c2291d…d2285`）

校验：在本目录执行 `certutil -hashfile P6_full\samples.jsonl SHA256`，或逐项对照
`logs/checksums.txt`（35 条，含 samples / flights）。

## 给 v1.3 补包用（`_additive_for_v1_3/`）

`diffusion_vpg.py`，sha256 `8f495b48d98180f1932161f188c014c40f3b408975c512794bb1d5975575aac4`，
应放在包内：

```
third_party/dppo_official_cc7234a/model/diffusion/diffusion_vpg.py
```

原因与中性证明见 `logs/compat_fix_note.md`：v1.2 包只带 `uav_idp3/`，而 `e5_runtime.py` 在传入
`min_std_vector` 时会按 `Path(__file__).parents[1]/model/diffusion/diffusion_vpg.py` 以文件方式加载它。
它是 BASE（F7 当前）版本，不是 NATIVE 历史版本；BASE 与 NATIVE 的 vendor 树 103 个 `.py` 只有这一个与
`e5_runtime.py` 不同。

## 注意

- 所有日志内部记录的是 WSL 绝对路径（`/home/cjj/AUV/...`），这是执行现场原始记录，未改写。
- 目录内文件未裁剪、未筛选；失败原件与成功结果并存。
