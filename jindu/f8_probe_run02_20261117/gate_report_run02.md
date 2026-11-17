# F8 采样探针 run02 — 3 状态门禁结果（v1.2）

- 日期：2026-11-16
- 交付包：`/mnt/c/Users/cjj/Desktop/IDP3/program/f8_replay_scope_fix_20261116/F8_probe_code_v1_2.zip`
  - zip sha256 = `966ff4049f665b1cc089d9a6184dc7ecff36d1199844c63d52c6d1d5ab0f7164`（与指导一致）
  - 解包目录 `/home/cjj/AUV/program/f8_probe_code_20261116_v12`；清单内 122 个声明 hash（2 改动文件 + 60 commit + 60 trace）逐项校验 **0 不符**；frozen 文件 sha256 未变。
- 结果目录：`/home/cjj/AUV/program/results/f8_sampling_probe_run02`
- 结论：**P6、P0 3 状态门禁均通过，逐位匹配。** timing20 / full 未执行（按指导等用户安排 GPU 时间）。

## 1. 实际命令与退出码

| # | 命令 | 退出码 | 输出目录 |
|---|------|--------|----------|
| 1 | `run … --policy P6 --phase gates`（未加 compat 前置） | 1 | `P6_gates/`（仅 identity.json） |
| 2 | `run … --policy P6 --phase gates`（compat 修复后） | **0** | `P6_gates_fix01/` |
| 3 | `run … --policy P0 --phase gates`（compat 修复后） | **0** | `P0_gates_fix01/` |
| 4 | `--phase timing` / `--phase full` / `summarize` | 未执行 | — |

第 1 次失败点与 v1.1 的 `raw_history_hash` **不同**，是新的、独立的依赖缺失（见 §3）；原始失败目录保留未动。

## 2. 门禁结果

`P6_gates_fix01/gates.json` / `P0_gates_fix01/gates.json`：

| 检查 | P6 | P0 |
|------|----|----|
| `passed` | true | true |
| 独立原回合数 | 3 | 3 |
| state_id | `01ef44a774d3d22c24da` / `3e0c6caaa4a7cada2458` / `47eb395ead644494fe70` | 同左 |
| 位置/速度/偏航 期望==实际 | true（逐位） | true（逐位） |
| 策略输入 hash 期望==实际 | true | true |
| 确定性重复相同 | true | true |
| 少重放一步被拒绝 | true | true |
| P6 `planned_policy_path == expected_path`（脚本内断言） | 通过 | — |
| `actual_use_base_policy` | false | true |

关键 hash（第一个 anchor）：P6 与 P0 的 `actual_input_hash` 均为
`34d000283c5b2dc210eed43f9d5ee75b963a0f06ca081cf4872f38e5673db6fb`，
与 v1.1 冻结期望的 `policy_input_hash` 完全一致；**P0 与 P6 同状态输入 hash 相同**，符合“P0输入hash与同状态P6一致”。

v1.1 的失败字段（`raw_history_hash`）在 v1.2 中已不再出现：恢复整个 run 的原 #E2 `torch.jit.optimized_execution(False)` 范围后，reset/前缀重放的合成点云逐位复现。

## 3. 本次发现的第二个问题与兼容修复（已记录，非静默替换）

第一次 v1.2 gate 失败于：

```
FileNotFoundError: .../f8_probe_code_20261116_v12/third_party/dppo_official_cc7234a/model/diffusion/diffusion_vpg.py
```

原因：`load_override(current_root)` 执行 `uav_idp3/e5_runtime.py`；当传入 `min_std_vector` 时，该模块按
`Path(__file__).resolve().parents[1]/"model/diffusion/diffusion_vpg.py"` 以文件方式加载“当前”VPGDiffusion 实现
（`e5_runtime.py` 第 85–89 行）。v1.2 包只带 `third_party/dppo_official_cc7234a/uav_idp3/`，未带同级 `model/` 树。

修复（未改动包内任何文件，未改 zip）：

- 从 BASE 复制 `third_party/dppo_official_cc7234a/model/diffusion/diffusion_vpg.py` 到 DRIVER 同路径；
  sha256 = `8f495b48d98180f1932161f188c014c40f3b408975c512794bb1d5975575aac4`。
- 计算中性证明：
  - 包内 `uav_idp3/e5_runtime.py` 与 BASE 的 `e5_runtime.py` 逐字节相同（`bade7d2a…`），即包意图就是“当前=F7 v1.1”；
  - BASE 与 NATIVE 的 vendor 树共 103 个 `.py`，**仅 2 个不同**（`diffusion_vpg.py`、`e5_runtime.py`），其余 101 个逐字节相同；
  - 被复制的 `diffusion_vpg.py` 只 import `model.diffusion.diffusion` 与 `model.diffusion.sampling`，这两个模块在两侧完全相同。
- 记录：`logs/compat_fix_note.md`。建议 v1.3 包直接补上该 vendor 文件（或改为从 `--repo` 解析同级 vendor）。

## 4. 边界声明

- 未修改任何阈值、判据、采样集、seed、权重、奖励或原始结果；未训练；未覆盖 v1.1 或本次首次失败的目录。
- 未运行 `test12`；原生 P6 源码仅经 `native_system` 只读加载（runtime identity `d987235a…`、state sha `a91c9f42…` 均校验通过）。
- `evidence/source_P6` 与 frozen 一致：抽查 anchor `01ef44a774d3d22c24da` 对应
  `evidence/source_P6/traces/1b48b6c5970d51f635fc58fb.attempt001.jsonl` 第 4 行（1-based），provenance 与冻结状态**逐字段相同**；
  `source_anchor_check.json` 声明 60 回合 / 211 状态全部 `provenance_matches=true`。
- 观察：冻结来源确为 `wsl_snapshot/routea_f5_run01/results/eval_dev/full/P6`，与 WSL 当前 `jindu/routea_a2_…` 同名轨迹 hash 不同——v1.1 报告中“来源轨迹不在盘上”的判断由此得到解释，v1.1 未取错行。

## 5. 下一步（等用户安排）

timing20 与 full 均属 GPU 评价，尚未执行。若按指导推进，命令为：

```bash
"$PY" "$DRIVER/scripts/run_f8_sampling_probe.py" run "${COMMON[@]}" --policy P6 --phase timing --output "$OUT/P6_timing20"
```

timing 会输出前 20 次续飞各次含重放耗时与 `projected_2400_s`；目前尚无实测耗时，不做估算。
