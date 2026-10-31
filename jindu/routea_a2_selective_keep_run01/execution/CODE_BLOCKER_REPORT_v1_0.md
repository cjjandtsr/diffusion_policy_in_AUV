# A2 执行阻塞报告 CODE_BLOCKER_REPORT v1.0（2026-10-24）

**结论**：07_T_u129 在 λ 标定阶段因 A2 交付代码访问不存在的 runtime 属性而崩溃（rc=1）。属交付源码缺陷（受保护源码），按执行指导 §6 停止并回传 Codex，未做任何源码修改。

## 1. 失败签名

```
AttributeError: 'E5PPODiffusion' object has no attribute 'alphas_cumprod'
  File "diffusion_policy_3d/rl/dppo_uav_rebuild/engine/selective_keep.py", line 117, in loss
    alpha = runtime.alphas_cumprod.to(device)[t].to(x0.dtype).reshape(-1,1,1)
  调用链: pipeline.py:646 calibrate_keep → selective_keep.py:167 keep.loss → :117
```

rc=1；命令/退出码/日志已保留：`$OUT/execution/07_T_u129.{command,rc,log,pid}`。

## 2. 根因 [FACT 代码]

- `E5PPODiffusion` 定义于 `third_party/dppo_official_cc7234a/uav_idp3/e5_runtime.py:137`（继承官方 PPODiffusion）。其注册的 buffer 为：`betas`(:195)、`ddim_t`(:196)、`ddim_alphas`(:197, = `alphas_cumprod[ddim_t]`)、`ddim_alphas_prev`(:203-207)，另有 plain attr `native_ddim_alphas_cpu`(:220)。**不存在 `alphas_cumprod` 属性。**
- `selective_keep.py:117` 按 `alphas_cumprod[t]`（时间步值索引）取 alpha；runtime 实际提供的是 `ddim_alphas`（**与 ddim_t 位置对齐**，不是时间步值索引）。因此修复不是简单改名，需把 `t ∈ tail(ddim_t)` 的采样改为对 tail 的**位置索引**（`ddim_alphas[-ft_denoising_steps:]` 同序对应）。
- 该缺陷同样会在正式训练中触发：`trainer.py:297` 每次逻辑更新的 keep loss 走同一 `loss()` 路径。即使跳过标定，训练也会崩溃 → 判定为阻塞性缺陷。
- `loss()` 中其余属性已验证存在：`runtime.ddim_t`、`runtime.ft_denoising_steps`（:114 先于 :117 执行成功）、`predict_epsilon`（e5_runtime.py 中存在）。

## 3. 为什么不在 WSL 自修

1. 执行指导 §6：受保护源码身份冲突 → 停止回传；修改交付文件会改变子身份闭包 hash，且与 Windows 交付清单及 Codex 侧源码失同步。
2. 身份链事实 [FACT]：C 臂 u129 checkpoint 已在当前（未修改）A2 源码下提交，child_runtime_identity_hash=9bf0de30…。任何 selective_keep.py 改动 → 新 identity → C u129 无法 resume → **C u129 必须在修复后重跑**（实测 ~7 min，无实质成本）。
3. Codex 是交付源码的唯一权威（Windows 侧需同步修改并重发交付清单）。

## 4. 当前产物状态（全部证据保留，未清理）

| 产物 | 状态 |
|---|---|
| 01_parent receipt（PARENT_VERIFIED, sha256=a9cd8e3f） | ✅ 完好 |
| 04_reference（126 任务/252 回合/549 前缀，reference.pt sha256=8759fc40） | ✅ 完好，修复无需重采 |
| 05_configure（C.yaml/T.yaml/plan_receipt.json） | ✅ 完好，修复无需重配 |
| 06_old u129（A1 代码重放，identity 391a6656） | ✅ 完好，修复无需重跑 |
| 07_C_u129（child identity 9bf0de30，fork_lineage 完整） | ✅ 完好但**修复后需重跑** |
| 07_T_u129 | ❌ 标定前失败，无 checkpoint/metrics 提交；`$OUT/arms/T/` 残留 `fork_lineage.json`+`r1_plan_probe.json`，`$WEIGHTS/T/` 空。**重跑前需清空这两个残留文件**（`_write_json_once`/run.lock 会拒绝非空目录）；失败证据（log/rc）保留不动 |

## 5. 修复后重入路径（建议）

1. Codex 修复 selective_keep.py:117 的 alpha 取用（ddim_alphas 位置对齐语义）+ Windows 源同步 + 更新交付清单 sha256。
2. WSL：按新清单重跑 02_delivery → 03_tests → 移除 T 残留（fork_lineage.json/r1_plan_probe.json）→ 重跑 07_C_u129 + 07_T_u129 → 08_gate。
3. 06_old/04_reference/05_configure 不受影响，无需重跑。

## 6. 交付定位

- 本报告：`program/results/routea_a2_selective_keep_run01/execution/CODE_BLOCKER_REPORT_v1_0.md`
- 交接：`program/jiaojie/A2_首次门禁_标定崩溃_交接_20261024.md`
- 全部执行证据在 `$OUT/execution/`（00~07 各 run 的 command/log/rc、pre_existing_a2 快照、delivery_sha256.txt、task_plan 在 `routea_a2_selective_keep_run01_plan/`）
