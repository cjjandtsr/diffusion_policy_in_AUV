# F12 第一阶段判别实验与前期研究（WSL 执行侧回传）

日期：2026-11-25/26（America/Chicago）。执行端：WSL。依据：`执行指引/2026-11-26_F12第一阶段判别实验与前期研究_WSL执行指引_v1_0.md` 与 `2026-11-26_下一步完整方案_F12多方向并进_v1_0.md`。

## 1. 身份与范围
| 项 | 值 |
|---|---|
| 项目仓库 | main `d54879fb`（本机原工作树无此提交；从 `repo_rewrite_20261125/archive.git` 检出） |
| 代码分支 | `f12-phase1-code`（已推 GitHub `0416b775`），仅新增 `scripts/f12_phase1/` |
| 权重 | P6@u256 `state.pt` `a91c9f42…dd43`；runtime identity `d987235a…b61f` |
| 采样 | #E2：clip 2.0、确定性、11×0.1；S0 与 O2 续飞同 |
| 数据隔离 | 120 被挡 + 120 可通过训练任务；dev420/val20/test12 三元组排除，S0 命中 0 |

本阶段不训练、不改已有源文件。O1/O8 由 Claude 完成；O2–O7 由本目录回传，分析判读归 Claude。

> 探针 2 在完整 4389 状态上出现 **0.01037 m** 的跨进程 GPU float32 差异；按 `probes.json` 的 `tolerance_m=2e-2` 记为通过，并在 `implementation_checklist.md` 标注。其余探针为 0 误差/精确值。

## 2. 目录
- `s0/`：`s0_task_pool.json`、`s0_episodes.csv`、`s0_states.csv`、`s0_states.npz`、`traces/*.jsonl.gz`、`s0_selection_rules.json`
- `o2/`：`o2_candidates.csv`、`o2_candidate_arrays.npz`、`o2_rollout_plan.csv`、`o2_rollouts.csv`、`o2_states.csv`、`o2_selection.json`
- `o3/`：`o3_features.npz`、`o3_features.json`
- `o4/`：`o4_openloop.csv`、`o4_summary.json`
- `o5/`：`o5_counterfactual.csv`、`o5_plans_body.npz`、`o5_selection.json`
- `o6/`：`o6_report.md`、`o6_facts.json`
- `o7/`：`o7_demo_reaction.py`（SHA `d364eeac…e3bf`）、`.json`、`.txt`、selftest/run logs
- `probes.json`、`implementation_checklist.md`、`logs/`、`MANIFEST.sha256`

## 3. 运行环境与命令
- 代码：GitHub `f12-phase1-code` 分支，commit `0416b775`（基于 `d54879fb`）；WSL 临时 worktree 已按清理要求删除。
- 结果：本目录（`/home/cjj/AUV/program/results/routea_f12_phase1_run01`）。
- O7 需 zarr 2：`/home/cjj/miniconda3/envs/robodiff/bin/python3 -I o7_demo_reaction.py <zarr> <out>`；系统 python3 的 zarr 3 不兼容脚本的 `zarr.open(path,'r')`。

## 4. 关键事实（只报数字）
- S0：任务池 120/120；运行 1200 回合；选态 A=1579、B=610、C=2200，共 4389。
- O2：A=200 状态 × 7 组 × 32 = 44800 候选；续飞 10464 条；1 m 到达 5905/10464（56.4%）。
  - `safe_and_clear6`（无碰撞且末端前向自由距离≥6 m，**不是侧绕**）：G-orig 24.7%、G-rot30 45.0%、G-rot60 51.7%。
  - 显式 sideway（无碰撞+前向≥6 m+|末端侧向|≥0.5 m）：G-orig 13.6%、G-rot30 40.9%、G-rot60 48.6%；rot30 左 1163/右 1455，rot60 左 1081/右 2028。
  - 状态内末端侧向 std 中位：G-orig 0.0847 m、σ0.2 0.1677 m、σ0.3 0.2440 m（合并口径 0.582/0.604/0.632 m 是跨状态量，不能当同状态散布）；σ0.3 锯齿度 0.119，为原样 0.0416 的 2.86 倍。
- O3：4389 状态 × 576 维；P6/BC 特征逐位相同（共用冻结编码器）。
- O4：3000 帧；BC 预测/标签前 4 步位移中位比 **1.114**，P6 **1.116**；静止历史使中位位移变为正常的 **0.945×（BC）/0.946×（P6）**。
- O5：150 B 状态 × 4 场景。相对原样的端侧向绝对变化中位：去障碍 **0.153 m**、目标左移 2 m **0.328 m**、两者 **0.415 m**；去障碍/目标侧移比 **0.465**（>0.3，按逐状态变化口径不支持目标捷径；按原始端侧向中位 0.0998/0.3509=0.284 则相反，两种口径都保留在 CSV）。
- O6：评价/训练链均无 CFG；step 链无 z 投影、无单步/累计偏航限制；噪声下限 0.1/0.1；FT 10/10；LoRA 14 Linear + 33 Conv1d；keep 0.01061006086489362 恒定；P6 训练 bank 80/85 任务被挡；时限 72 s；BC 3 帧/16 步/d_safe 0.75/λ0.5/末段 3×；BC 训练约 9.87 h；RL 外层更新约 14.6–15.3 min；推理决策中位 227.1 ms。
- O7：示范在 3—5 m 处偏角中位 **6.0°**（90 分位 21.2°），≤10°，支持“示范本身反应也晚”。

## 5. 偏离与解释
详见 `implementation_checklist.md`。最重要的三项：
1. G-early 在加载后的 runtime 中与 G-orig 的步数条件相同（`ft_denoising_steps=ddim_steps=10`），是退化组。
2. O2 的“16 步规划”按 F10 引导模块用 `full[2:16]`，即 14 个 delta + 当前点 = 15 个世界点。
3. O3 的 P6/BC 特征相同，因为两者共用冻结 observation encoder；LoRA 只改去噪器。

## 6. 复现
```bash
git clone https://github.com/cjjandtsr/diffusion_policy_in_AUV.git
cd diffusion_policy_in_AUV/Improved-3D-Diffusion-Policy
git checkout f12-phase1-code
python3 scripts/f12_phase1/s0_pool.py
python3 scripts/f12_phase1/s0_collect.py
python3 scripts/f12_phase1/o2_candidates.py
python3 scripts/f12_phase1/o2_rollouts.py
python3 scripts/f12_phase1/o3_features.py
python3 scripts/f12_phase1/o4_openloop.py
python3 scripts/f12_phase1/o5_counterfactual.py
python3 scripts/f12_phase1/probes.py
```
O6/O7 见对应目录。临时 worktree、smoke 输出、O5 临时几何 JSON 与打包辅助脚本已删除。

## 7. 状态变更块（执行侧）
- F12 第一阶段 S0/O2/O3/O4/O5/O6/O7 与 8 项探针已执行；代码在 `f12-phase1-code`；原始结果在本目录并打包 `routea_f12_phase1_run01.tar.zst`。
- 未训练、未改已有源文件、未改评价集、未选 checkpoint。
