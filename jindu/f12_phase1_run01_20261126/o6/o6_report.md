# F12 O6 事实核对（只读）

依据代码：`/home/cjj/AUV/temp/f12_code_wt/Improved-3D-Diffusion-Policy`，main `d54879fb`，分支 `f12-phase1-code`。
运行证据：`program/results/routea_f9_run01`、`routea_f10_run01`，权重 `data/outputs/...`。

## 1. dev420 评价链与训练采集链是否使用 CFG
**结论：都不使用 CFG。**
- 评价入口 `scripts/evaluate_dppo_uav_formal_endpoint.py:384-390` 只传 `cond=condition`，没有无条件分支；`third_party/dppo_official_cc7234a/uav_idp3/e5_runtime.py:185` 明确拒绝 `global_cond_uncond`。
- 训练重建链 `diffusion_policy_3d/rl/dppo_uav_rebuild/engine/f9_online.py` / `paired_sample` 同样只构造条件 `global_cond`。
- 仓库里确有旧的 CFG 代码 `diffusion_policy_3d/rl/ppo_trainer.py:187-249`，但它是旧 PPO 训练器，不被 F9/F10 rebuilt 入口导入；不能据此说评价链用了 CFG。
```python
# e5_runtime.py:185
if "global_cond_uncond" in cond:
    raise E5RuntimeError("E5 checkpoint has no CFG/unconditional branch")
# evaluate_dppo_uav_formal_endpoint.py:384
samples = self.system.official_runtime(
    cond=condition, deterministic=True, return_chain=True,
    use_base_policy=self.use_base_policy)
```

## 2. 轨迹投影（z 范围、单步偏航、累计偏航）
**结论：训练/评价的 step 链没有轨迹投影。**
- `kinematic_uav_env.py:735` 的 `np.clip(pos[2], z_min, z_max)` 只出现在 `_perturb_to_mid_state()`（reset 的中途状态扰动），不是 `step()`。
- `step()` 在 `kinematic_uav_env.py:775-789` 直接执行 `world_wp[3]`，只对 yaw 做 `normalize_angle`（±π 环绕），没有单步或累计偏航限制。
- 高度只在终止判定里用 `z_max + 1.0`：`kinematic_uav_env.py:1080,1166`；没有把动作投影回 z∈[1,5]。
- 评价侧的 `task["z_range"]` 只设置环境 `z_min/z_max`（`evaluate_dppo_uav_formal_endpoint.py:111-117`），不修改动作。
```python
# kinematic_uav_env.py:775
new_pos = world_wp[last_idx, POS].copy()
new_yaw = float(world_wp[last_idx, YAW])
...
# :789
self._drone_yaw = normalize_angle(new_yaw)
```

## 3. 探索噪声下限与似然噪声下限
**结论：两者都是 0.1（标量；评价 #E2 用 11 维 0.1 向量）。**
- `entrypoints/build_system.py:84-85` 从 `config.dppo.min_sampling_std` / `min_logprob_std` 读取，P6 训练配置均为 0.1。
- F10 #E2 覆盖在 `scripts/run_routea_f10_eval.py:335-337`：`min_std_vector=[.1]*11`。
- 运行探针读取 `runtime.min_sampling_denoising_std == 0.1`、`runtime.min_logprob_denoising_std == 0.1`。
```python
sampling_floor = float(config.dppo.min_sampling_std)
logprob_floor = float(config.dppo.min_logprob_std)
```

## 4. 微调的去噪步
**结论：10 步全部微调；`ft_denoising_steps == ddim_steps == 10`。**
- `build_system.py:98` 构造 `ft_denoising_steps=10`，`num_inference_steps=10`。
- 加载后的 runtime 探针：`ddim_steps=10, ft_denoising_steps=10`。
- 因此“只在微调步注入探索噪声”与“全部 10 步注入”在当前配置下没有可区分的步；O2 的 G-early 组是退化组，回执会单列该事实。

## 5. LoRA 作用层
**结论：只作用于去噪器，14 个 Linear + 33 个 Conv1d；观测编码器的 2 个 Linear + 10 个 Conv1d 冻结。**
- `actor_ddpm/lora.py:20-24` 固定计数；`install_lora()` 只替换这 47 个模块。
- rank=8、alpha=8（P6 配置 `lora.rank/alpha`）。
```python
E5_DENOISER_LINEAR_TARGETS = 14
E5_DENOISER_CONV1D_TARGETS = 33
E5_FROZEN_ENCODER_LINEAR_MODULES = 2
E5_FROZEN_ENCODER_CONV1D_MODULES = 10
```

## 6. 保持项系数与调度
**结论：P6 的 keep 系数为 0.01061006086489362，固定不衰减；更新时按梯度关系叠加。**
- `scripts/run_routea_f9.py:264` 写死 `keep_coefficient=.01061006086489362`；`run_routea_f5.py:123` 拒绝重标定。
- `engine/trainer.py:302-332` 把 keep 梯度按系数加入 PPO 梯度；没有时间调度（不是 warmup/anneal）。
- P6 的 `a2.keep_seed=2026102301` 只用于参考样本，不是系数调度。

## 7. P6 训练任务中被挡任务比例
**结论：按 P6 训练 bank（`routea_f8_diag_run03/bank.json`），85 个任务中 80 个被挡（94.1%）；425 个条件中 400 个被挡（94.1%）。**
- 定义：起点—目标直线按 0.05 m 采样，最小几何净空 < 0.3 m。
- 分角色：dense_recovery 57、yaw_lateral_recovery 8、anticipatory 8、support_expansion 7。
- 注意：这是条件/任务比例，不是转移比例；O1 报告按转移统计被挡约 20%，两者口径不同。

## 8. 评论家输入
**结论：评论家读 576 维全局条件 + episode 标量上下文，不读点云原始点。**
- `engine/critic_state.py:with_history_scalar()` 增加 `beta = best_distance / initial_distance`（1 维）。
- `with_episode_context()` 再增加 6 维：`beta`、`initial_distance/10`、起点相对当前位置的机体系 xy、z、剩余时间比例；共 7 维上下文。
- 值网络实现见 `engine/value.py`；actor 条件 `global_cond` 保持 576 维不变。

## 9. 回合时限
**结论：60 个决策 × 1.2 s = 72 s；E2 环境 `max_episode_steps=60`。**
- P6 配置 `environment.max_episode_steps=60`；`KinematicUAVEnv.step` 每决策执行 4 个 0.3 s waypoint。
- 评价记录器 `routea_paired_support/episodes.py:51` 固定 `deadline=60`。
- O8 的 72 s 即由此而来。

## 10. BC 配置
**结论：历史 3 帧、预测 16 步、执行 8 步；碰撞辅助损失 d_safe=0.75、λ=0.5；末段加权 3×（dist<2 m）；点云增广；动作 gaussian 归一化。**
- 历史/视距：checkpoint `cfg.n_obs_steps=3, horizon=16, n_action_steps=8`；`dataset/drone_dataset.py:60-68` 对应 `horizon=16`。
- 碰撞损失：`policy/phase1a_policy.py:67-68` 默认 `collision_lambda=0.5, collision_d_safe=0.75`；`model/common/collision_loss_v9.py:26` 注明训练侧 0.75。E5 注册说明 `policy_loader.py:48-52` 明确 `lambda_col=0.5`。
- 末段加权：`drone_dataset.py:60` `end_weight=3.0, end_dist_threshold=2.0`。
- 增广：`drone_dataset.py:184-220`，dropout 0.7–1.0、jitter 0.02、depth noise 0.001/0.03、旋转 ±30°。
- 归一化：`drone_dataset.py:172-181` 用 `LinearNormalizer.fit(..., mode='gaussian')`；点云/agent_pos 为 identity。
- dpos 三维统计（整个 `merged_body_12.zarr`，60151 帧）：mean `[0.027713, 0.002538, 0.000540]`，std `[0.233988, 0.208490, 0.036756]`，min `[-0.628862, -0.634994, -1.0]`，max `[0.626215, 0.684423, 1.0]`。
```python
# drone_dataset.py:172
def get_normalizer(self, mode='gaussian', **kwargs):
    data = {'action': self.replay_buffer['action']}
    normalizer = LinearNormalizer()
    normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
```

## 11. 示范点云坐标系
**结论：只按偏航旋转的机体系（+x 前、+y 左、+z 上），原点在无人机机体中心，不是相机光心。**
- `synth_pointcloud.py:351-379` 明确 body→world 仅绕 z 轴 yaw；相机安装偏移 `(0.12, 0, 0.015)` 在机体系内，合成后再把点变换回 body。
- `kinematic_uav_env.py:800-811` 用 `synth_pc.synthesize(wp_pos, wp_yaw)` 生成 body-frame 点云；`_build_state` 的 `goal_body` 也是同一 yaw 旋转。
```python
# synth_pointcloud.py:354
body  frame: +x=forward, +y=left, +z=up
body→world: R(yaw) rotation around z-axis + pos_world
```

## 12. BC 一次完整训练耗时
**结论：约 9 小时 52 分（从输出目录创建 2026-08-15 19:55 到 logs 最后写入 2026-08-16 05:47）。**
- 目录：`data/outputs/2026.08.16/08.53.26_e5_lambda0p5/`；日志 `logs.json.txt` 到 epoch 299。
- 这是文件时间戳估计，不是训练器内部计时；训练日志没有 `wall_seconds` 字段。

## 13. 强化学习一轮外层更新耗时
**结论：F9 P35/P36 实测约 14.5–15.3 分钟/外层更新（8 次更新约 1 h 56 m–2 h 02 m）。**
- P35 `metrics_update_000257.json` 21:16:20 → `000264.json` 23:18:54（2 h 02 m 34 s / 8 = 15.3 min）。
- P36 09:02:03 → 10:58:44（1 h 56 m 41 s / 8 = 14.6 min）。
- 每轮 1024 转移、20 次 optimizer commit（P6 配置 `rollout_transitions=1024`、`expected_optimizer_commits_per_outer_update=20`）。

## 14. 推理一次决策耗时
**结论：P6 #E2 去噪中位 212.1 ms、p95 271.6 ms；整次决策中位 227.1 ms（F10 T6，358 个决策）。**
- 来源：`program/results/routea_f10_run01/t6_timing.json`，README §2 T6。
- 计时在 `torch.cuda.synchronize()` 包围下，仅去噪循环 vs encode+denoise+unnormalize 全程。
- 带 g1 引导时中位 228.2 ms / 241.3 ms，量级不变。

## 附：O6 的边界
- 以上“结论”均为代码/文件证据；未把历史文档中的旧配置当作现行。
- 若后续改动 `build_system.py`、`kinematic_uav_env.py` 或 P6 配置，本核对需重跑。
