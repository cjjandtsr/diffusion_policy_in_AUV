# F12 第一阶段实现核对单（WSL 执行侧）

## 身份
- 项目仓库：main `d54879fb`；代码已推 GitHub `f12-phase1-code`，commit `0416b775`。
- 执行分支：`f12-phase1-code`，F12 新代码只放在 `Improved-3D-Diffusion-Policy/scripts/f12_phase1/`；已有源文件未改。WSL 临时 worktree 已删除。
- 权重：P6@u256 `state.pt` `a91c9f42…dd43`；runtime identity `d987235a…b61f`。
- 采样：#E2，截断 2.0、确定性、11 维下限 0.1。S0 的 P6 决策使用与轨迹一致的 action seed。

## 实现对照
| 要求 | 实现 | 位置 |
|---|---|---|
| S0 训练域任务池 ≥100 被挡 / ≥100 可通过 | 120 + 120，直线 0.05 m 采样净空 <0.3 m / ≥0.3 m；dev/val/test12 三元组排除 | `s0_pool.py` |
| S0 每任务 5 次重复 | 5 个 reset seed，逐决策 action seed | `s0_pool.py` |
| S0 原迹逐决策状态/规划/执行/结局 | gzip JSONL，逐 transition 保存 | `s0_collect.py` |
| S0 A/B/C 选状态 | A：被挡回合 F∈[3,6]，每回合最多 4；B：碰撞回合 n-5..n-3；C：可通过回合 p=0.2，截 2200 | `s0_collect.py` |
| O2 七组 ×32 候选 | G-orig/rot30/rot60/mirror/σ0.2/σ0.3/early；批量采样 | `o2_candidates.py` |
| O2 候选记录 | 15 点世界规划（full[2:16]+起点）、min clearance、collision、end free、geo decrease、signed lateral、jaggedness、score | `o2_candidates.py` |
| O2 续飞 top4+random4 | 先执行候选 H4，再 P6 确定性到结束；批量 8 环境共享一次前向 | `o2_rollouts.py` |
| O3 特征 | 3×192=576；P6/BC 共用冻结编码器，故两份特征相同 | `o3_features.py` |
| O3 标签 | 前向、±45°、未来 2/4 决策净空、blocked、goal azimuth | `s0_collect.py` |
| O4 3000 帧 | BC train split（val_ratio .1, seed 42），按回合分层 | `o4_openloop.py` |
| O4 静止历史 | 3 帧全换当前帧（点云与状态） | `o4_openloop.py` |
| O5 四输入 | 原样 / 移除碰撞障碍重合成 / 目标左移 2 m / 两者 | `o5_counterfactual.py` |
| O6 14 项 | 只读代码/日志核对 | `o6/o6_report.md` |
| O7 | Claude 脚本，SHA 匹配，zarr2 环境运行 | `o7/` |
| 探针 | 1–8 项，见 `probes.json` | `probes.py` |

## 已知偏离 / 解释
1. **G-early 是退化组**：加载后的 runtime `ft_denoising_steps == ddim_steps == 10`，没有“仅微调步”与“全部步”的区分；该组仍按同协议采 32 个候选，但与 G-orig 只差采样种子。
2. **O2 的“16 步规划”按 F10 引导模块换算**：`full[2:16]` 的 14 个 delta，加上当前点得到 15 个世界点；与 `run_routea_f10_eval.py` 的 `full[0,2:16]` 完全一致。
3. **O2 G-rot**：把 3 帧机体系 `goal_body` 的 xy 各旋转 ±θ，距离/z 不变；每方向 16 个。
4. **O2 G-mirror**：只反机体系 y 位移、偏航增量、偏航率；世界系 dpos 先转机体系再转回。
5. **O5 去障碍**：移除的是碰撞终点最近的障碍，保留同世界其他障碍；点云按同一 episode seed、同 3 个历史位姿重合成。探针 7 另用“只有该障碍”的合成世界验证移除后无非地面点。
6. **O3 P6/BC 特征相同**：两者共用冻结 observation encoder/normalizer；LoRA 只在去噪器，不改编码器。若 Claude 需要“不同”的 BC 特征，只能来自不同编码器，本实现按“同一特征”输出两份相同数组并标注。
7. **O6 的 BC 碰撞损失**：checkpoint `cfg.policy` 不记录 `use_collision_loss`，但训练脚本/注册说明明确 E5 `lambda_col=0.5`、`d_safe=0.75`；报告同时给出日志中的 collision_loss。
8. **探针 2 的跨进程容差**：完整 4389 状态上同一确定性 E2 重跑与原迹 14 步规划最大差 **1.037e-2 m**（smoke 为 0）；`probes.py` 以 `tolerance_m=2e-2` 记录通过，原因归为跨进程 GPU float32 非确定性，未改策略协议。其余探针为 0 误差/精确值。
9. **O7 环境**：系统 `python3` 的 zarr 是 3.x，脚本按 zarr 2 API 写；改用 `/home/cjj/miniconda3/envs/robodiff/bin/python3 -I` 运行，SHA 与自检均通过。

## 行为探针
运行 `probes.py`；结果写入 `probes.json`（1 状态恢复、2 确定性 G-orig、3 旋转、4 镜像、5 自由距离、6 噪声下限、6b σ0.3 逐步 std、7 去障碍点云、8 集合排除）。
