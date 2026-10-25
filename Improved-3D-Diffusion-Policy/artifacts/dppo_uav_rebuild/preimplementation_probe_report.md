# DPPO-UAV Rebuild 编码前事实探测报告 (阶段 A, P0–P7)

- 探测日期: 2026-08-26 (本地时间 14:30–14:40 区间, 各命令时间戳见原始日志)
- 仓库: `/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy`
- git branch: `a-official-core-r7`; HEAD: `cc9de00df00ced6bf26658f093d98d19cc14fbfc`
- git status: clean (仅 `?? artifacts/` 为本次 probe 产物)
- 全部命令从仓库根目录执行; 所有依赖导入失败原样记录, 无任何 pip 升级
- 本探测只读; 未修改任何仓库跟踪文件

---

## 0. 结论速览

| # | 探测项 | 状态 | 一句话结论 |
|---|--------|------|-----------|
| P0 | 环境/仓库/依赖身份 | **PASS** | python = idp3 env; torch 2.4.1+cu121; CUDA 可用; 全部依赖导入成功 |
| P1 | 官方 DPPO vendor 完整性 | **PASS** | UPSTREAM_COMMIT == `cc7234ad…`; sha256sum -c 全部 OK (exit 0); 401 manifest == 401 disk |
| P2 | 官方 DPPO 真实接口 | **PASS** | VPGDiffusion/PPODiffusion 源码+运行时签名; p_mean_var→(mu,logvar,etas); 8-tuple PPO loss; γ^(K−i−1) discount |
| P3 | E5 checkpoint 身份 | **PASS** | SHA256 == `c1fca4ac…`; model/ema_model 192/192 keys; 原生构造+加载 0 missing/unexpected |
| P4 | E5 单样本确定性 smoke | **PASS** | full(16,11)/policy(8,11)/executed(2,11) 精确切片一致; 双重复 max abs diff = 0.0; 需 `policy.normalizer.to(device)` |
| P5 | E5 模块/LoRA 清单 | **PASS** | 16 Linear + 43 Conv1d + 0 Conv2d 全形状; 3帧历史→global_cond(1,576) hook 证据; 12 个 FiLM Linear 输入 (1,704) |
| P6 | Route A env/Reward v1.3/数据准入 | **PASS** | kinematic env v1(v13)/v2(v14)/gazebo 边界清楚; reward_v13 纯 numpy 可复用; w-gates 全部有真实代码 |
| P7 | 裁决 | 见 §7 | 8 项机器可读; 无硬阻塞; 3 项接口决策点标记 DO_NOT_INFER |

---

## 1. P0 — 环境、仓库和依赖身份 [PASS]

日志: `logs/P0_env_identity.log`, `logs/P0_deps_import.log`

| 命令 | 退出码 | 结果 |
|------|--------|------|
| `which python` (idp3 env) | 0 | `/home/cjj/miniconda3/envs/idp3/bin/python` ✓ (用户指定路径) |
| `python --version` | 0 | Python 3.8.20 |
| `python -c "import torch; ..."` | 0 | torch 2.4.1+cu121, CUDA available |
| `python -c "import diffusers, hydra, omegaconf, numpy, zarr, gymnasium"` | 0 | 全部导入成功 |
| `git rev-parse HEAD; git branch --show-current; git status --porcelain` | 0 | cc9de00…; a-official-core-r7; clean (仅 artifacts/) |

关键事实:
- diffusers 0.36.0 — `DDIM.step()` **无 `**kwargs`**(影响 inference-time guidance 接口设计, 见 P2)
- 上级目录 `/home/cjj/AUV` 非 git 仓库; 本目录才是真实仓库根
- 无 pip 安装/升级发生

---

## 2. P1 — 官方 DPPO vendor 完整性 [PASS]

日志: `logs/P1_vendor_integrity.log`, `logs/P1_vendor_counts.log`, `logs/P1_vendor_unlisted.log`

| 命令 | 退出码 | 结果 |
|------|--------|------|
| `grep UPSTREAM_COMMIT third_party/dppo_official_cc7234a/VERSION*` | 0 | `cc7234ad7ff39a8f32de3af903606723a16f0648` == 用户指定值 ✓ |
| `sha256sum -c SHA256SUMS` (vendor 目录内) | 0 | 全部 OK, 0 FAILED |
| `sha256sum SHA256SUMS` (自hash) | 0 | 记录于日志 |
| manifest count vs disk count | 0 | **401 == 401**; extras 仅 `SHA256SUMS` 本身 + 20 个 `__pycache__/*.pyc` |
| vendor 内 `import dppo` / `import model` | 0 | `model` 顶层包可导入 (与 dppo 0.8.0 同源); 仓库根 `import model` 失败 = 无遮蔽 ✓ |

结论: vendor 冻结于官方 commit `cc7234ad`, 文件完整, 包名为顶层 `model`, 无 pip 安装、无遮蔽。

---

## 3. P2 — 官方 DPPO 真实接口与可导入性 [PASS]

日志: `logs/P2_api_probe.log`; 脚本: `scripts/p2_api_probe.py` (AST 静态提取 + 运行时 `inspect.signature` 双通道)

| 事实 | 证据 (file:line) |
|------|------------------|
| VPGDiffusion 继承 DiffusionModel | `third_party/dppo_official_cc7234a/model/diffusion/diffusion_vpg.py` |
| PPODiffusion 继承 VPGDiffusion | `diffusion_ppo.py` |
| `p_mean_var` 返回 `(mu, logvar, etas)` | `diffusion.py` (DiffusionModel base) |
| PPO ratio/clipped loss 8-tuple 返回 | `diffusion_ppo.py` `compute_loss` 区域 |
| denoising discount = γ^(K−i−1) | `diffusion_ppo.py` (reward_horizon 切片) |
| timestep 方向: 训练 t 递增(0=最终), 推理降序 | `diffusion.py` / `sampling.py` |
| DDIM eta 可学习; value loss 走 Critic | `model/common/critic.py` (CriticObs/CriticObsAct/ViTCritic) |

运行时验证: `inspect.getfile` 全部解析到 vendor 根内; `inspect.signature` 提取真实签名, 与 AST 提取一致。`import model` (bare) 从仓库根失败 — 无遮蔽, 需显式 sys.path。

---

## 4. P3 — E5 checkpoint 身份与原生构造 [PASS]

日志: `logs/P3_e5_identity.log`, `logs/P3_ckpt_load.log`, `logs/P3_ckpt_compare.log`, `logs/P3_native_load.log`
脚本: `scripts/p3_ckpt_probe.py`, `scripts/p3_ckpt_compare.py`, `scripts/p3_native_load.py`, `scripts/native_e5.py` (共享加载器)

### 4.1 文件身份

| 项 | 值 |
|----|-----|
| 路径 | `data/outputs/2026.08.16/08.53.26_e5_lambda0p5/checkpoints/latest.ckpt` |
| bytes | 1,144,701,858 |
| SHA256 | `c1fca4ac2735928b18fa7a9c069f90ad181eb84092acee379b073dc6a61bfc9a` == 用户指定值 ✓ |
| 顶层 keys | `cfg`, `state_dicts` (model/ema_model/optimizer/scheduler…) |
| model / ema_model | 各 **192** keys, key 集合完全一致 |
| model vs ema | 0 shape mismatch; 71,523,353 elements; max abs diff **0.10753**, mean 0.00442; 178/192 有差异 (EMA 落后) |

### 4.2 resolved config 关键字段 (ckpt["cfg"], 原生 load 后)

| 字段 | 值 |
|------|-----|
| policy/workspace/scheduler 目标 | `diffusion_policy_3d.policy.diffusion_pointcloud_policy.DiffusionPointcloudPolicy` / `idp3_workspace.IDP3Workspace` / `diffusers.DDIMScheduler` |
| prediction_type | `sample` (= predict x0) |
| train timesteps / inference steps | 50 / 10 |
| n_obs_steps / horizon / n_action_steps | **3 / 16 / 8** |
| action_start (policy 切片) | 2 (To−1) |
| traj_dt | 0.3 |
| point cloud | (16, 4096, 6), 6ch; `use_pc_color=False` → 只取前 3ch |
| state (agent_pos) | 9d: pos_world(3)+goal_body(3)+dist+sin_yaw+cos_yaw |
| history 顺序 | 窗口 [buf_start,buf_end), 帧序与 Buffer 一致 (旧→新) |
| normalizer | `LinearNormalizer`: action=gaussian, agent_pos/point_cloud=identity; 18 keys; 内容 hash `1edbc81f78c66a0f67270266cf333c06071307768e1273d2369aa945320ae1ca` |

### 4.3 权重来源 (E5 79/100 Gazebo 部署)

代码路径证据 (双加载器均确定性优先 `state_dicts["model"]`):
- `scripts/eval_ddpm_bc_baseline.py:146-147` — `sd = state_dicts["model"]` 优先
- `diffusion_policy_3d/rl/dppo_uav/policy_loader.py:76-86` — 冻结 manifest: Gazebo 3-way **79/100** (vs v8 lambda0.5 54/100), 2026-08-22 用户裁决; `:272-273` 同优先 `state_dicts.get("model", …)`
- 部署运行主日志**不在仓库内** → 部署实际加载行为 [UNCERTAIN], 以代码路径为唯一仓库内证据

**裁决: `E5_WEIGHT_SOURCE = PASS (代码路径), 部署运行含 caveat`** — 不是 UNCERTAIN/BLOCKED_WEIGHT_SOURCE, 因为两条独立代码路径一致指向 model, 且模型层 0 missing/unexpected 加载成功 (下方 P3.4)。

### 4.4 原生构造 + 加载

`hydra.utils.instantiate(cfg.policy)` 原生构造成功 (与 iDP3Workspace 同路径, `idp3_workspace.py:151-155`); `state_dicts["model"]` 与 `state_dicts["ema_model"]` 均 **0 missing / 0 unexpected**; `self.kwargs = {'use_image', 'use_point_crop'}` 惰性 (predict_action 不转发到 scheduler.step, 已代码验证 `diffusion_pointcloud_policy.py:218-224`)。

---

## 5. P4 — 原生 E5 单样本确定性 smoke [PASS]

日志: `logs/P4_smoke.log` (3 次运行: 初版 OOM / fix1 设备修复 / fix2 成功); 脚本: `scripts/p4_smoke.py`(OOM 版), `scripts/p4_smoke_lazy.py`, `scripts/p4_smoke_fix1.py`(成功版)

### 5.1 数据集与固定样本 (window 0, 冻结 zarr)

- zarr: `/home/cjj/AUV/avoidance_dataset/merged_body_12.zarr` — 988 episodes / 60151 steps; keys action/point_cloud/state
- 复现 DroneDataset split: get_val_mask(0.1, seed=42) → 889 train / 99 val; SequenceSampler(16, pad 2/7) → 49090 windows
- window idx=0: episode 0, buffer [0,14), sample [2,16), frame_in_episode 0, episode_frames [0,38)
- 点云下采样: `uniform_sampling_numpy` → 4096 (seed 12345 固定)

### 5.2 normalizer 数据身份 (本地 zarr vs ckpt)

- recompute gaussian from full action column (60151×11): 6 keys 与 ckpt **全部 byte-equal** (max_abs_diff=0.0, 含 input_stats.mean/std/min/max)
- ckpt 自身 18 keys 同函数 hash = `1edbc81f…` 精确复现 → **本地 action 列与训练数据逐字节一致**; 之前 MISMATCH 仅为 key 集合不同 (recompute 只 fit action)
- 结论: 数据身份闭合, 无漂移

### 5.3 运行指纹 (run 1, cuda:0, batch 1, no backward)

| 项 | 值 |
|----|-----|
| wall time | 1.1404 s |
| peak CUDA mem | 347.7 MB |
| returned keys | `['action', 'action_pred']` |
| scheduler timesteps (K=10) | `[45,40,35,30,25,20,15,10,5,0]` (降序, DDIM) |
| full action_pred | (1,16,11) float32, min -0.296438 max 0.922894 mean 0.020893, sha256 `2ec8e54d…` |
| policy action | (1,8,11), sha256 `57b3aaa2…` |
| executed = full[:,2:4] | (1,2,11), sha256 `a4d70897…` |

### 5.4 断言

| 断言 | 结果 |
|------|------|
| `policy == full[:, 2:10]` 精确 (torch.equal) | **True**, max abs diff 0.0 |
| `executed == full[:, 2:4]` 精确 | **True** |
| run2 同进程同输入同 seed | full/policy max abs diff = **0.0**, torch.equal True (完全确定) |
| obs 指纹 | agent_pos (1,16,9) sha256 `e7a53b05…`; point_cloud (1,16,4096,6) sha256 `57e7f41f…`; gt_action (1,16,11) `32e47158…` |

### 5.5 设备契约 (编码必须知道)

- `normalizer.py:352`: `x.to(device=scale.device, dtype=scale.dtype)` — **normalizer 参数所在设备决定输入设备**; ckpt 经 `map_location="cpu"` 加载后 normalizer 在 CPU → 原生推理必须 `policy.normalizer.to(DEVICE)` (原生 workspace 训练流程天然满足; 推理/评测代码必须显式做)
- `predict_action` 内部 `device = self.device` 只为创建 cond_data (zeros), 输入设备由 normalizer 决定; noise trajectory 在 `conditional_sample` 内按 `condition_data.device` 创建 (`diffusion_pointcloud_policy.py:204-207`)

---

## 6. P5 — E5 模块 / 特征 / LoRA 候选清单 [PASS]

日志: `logs/P5_module_inventory.log`; 脚本: `scripts/p5_module_inventory.py`

### 6.1 参数总览

- total 71,523,353 / trainable 71,523,275 (差 78 = normalizer 参数)
- obs_encoder (iDP3Encoder) 268,608; model (ConditionalUnet1D) 71,254,667

### 6.2 LoRA 候选模块 (ActorIdentity allowlist 输入)

| 类别 | 数量 | 说明 |
|------|------|------|
| Linear | 16 | 12 个 cond_encoder.1 (FiLM, 输入 (1,704), 形状 (2048/1024/512, 704) 各层级), diffusion_step_encoder.1/.3, state_mlp.0/.2 |
| Conv1d | 43 | multi_stage_pointnet 10 (enc), U-Net blocks 33 (含 mid/up/down/final) |
| Conv2d | 0 | — |

全量 59 模块名/形状/参数在日志内; 权重形状摘要: `obs_encoder.extractor.*` (128ch 1×1 卷积 + 512 输出), `model.down_modules.*` 最大 (1024,1024,5) = 5,243,904, `model.final_conv.1` (11,256,1) = 2,827。

### 6.3 3 帧历史 → global cond (forward-hook 实测)

- `obs_encoder` 输入: `{agent_pos: (3,9), point_cloud: (3,4096,3)}` — **To=3 帧展平成 batch** (每帧独立编码)
- 输出: (3,192) → reshape `(B, To*Do)` = **(1,576)** global_cond
- 12 个 cond_encoder Linear 输入全部 `(1,704)` = 576 (global_cond) + 128 (timestep emb) → FiLM 输出 (1,2048/1024/512) 各分辨率层
- 反方向事实: Route A `KinematicUAVEnv` 用 n_obs_steps=8 → global_cond 1536 (`kinematic_uav_env.py:146` 注释, `routea_replay_buffer.py:38/82` RouteATransition.obs_feature (1536,)) — **E5(3帧/576) 与 Route A(8帧/1536) 的 obs 维度契约不同, 编码时须显式对齐**

---

## 7. P6 — Route A env / Reward v1.3 / 数据准入 [PASS]

日志: 本报告 + Explore agent 全仓扫描 (file:line 证据链在 §7.4); 主 agent 精读: `kinematic_uav_env_v2.py:92-231,471-589,668-684`, `reward_v13.py:648-665`, `routea_rl_env_adapter_v1.py:40-330`, `entrypoint_guard.py:30-245`, `routea_replay_buffer.py:31-96`

### 7.1 环境

| 组件 | file:line | 事实 |
|------|-----------|------|
| KinematicUAVEnv v1 (reward v13) | `env/kinematic_uav_env.py:78` | gym.Env 纯 numpy; step (:656) action (8,11)→world_wp→执行 4 步; obs (8,4096,3)+(8,9); termination `_check_collision` (:940) / `_check_early_death` (:983) |
| KinematicUAVEnv v2 (v14/v16) | `env/kinematic_uav_env_v2.py:92` | 同名同构, 默认 reward_version='v14', collision_thresh=0.35, `export_episode_config` (:901) |
| GazeboUAVEnv | `env/gazebo_uav_env.py:50` | rospy/mavros/SITL 依赖, 旧 PBRS v1.2 → **不适用 Route A rebuild** |
| traj_utils / collision_geom | `traj_utils.py:48` action_to_world; `collision_geom.py:91` clearance | 纯数学, 双 env 共享 |
| UAVVectorEnvAdapter | `rl/dppo_uav/environment_adapter.py:628` | n_envs 向量 env, reset_at/step 槽位语义, AdmissionChecker 7 规则 (:466) |

### 7.2 Reward v1.3

- 主函数: `reward_v13.py:648` `compute_reward_v13(s_t, action_chunk, s_tp1, pointcloud, obstacle_json, goal_world, history, offline_density, cfg, clearance_fn) -> (reward, terms, metrics, updated_history)`
- 公式: `r_t = r_base + r_pbrs + r_onset + r_fc + r_trap + r_flip + r_sup + r_guard` (:6)
- 状态变量: `create_reward_history` (:431, episode_caps), `update_trap_state` (:483), history 原地修改 (progress/d_min/m_t deques)
- 配置: `REWARD_V13_DEFAULTS` (:32, 全部 cfg 键)
- 纯 numpy, 无 torch/GPU 依赖; `test_reward_v13.py` 全覆盖 (11 测试类)

### 7.3 数据准入

| 组件 | file:line | 事实 |
|------|-----------|------|
| RouteARLWorldEntry | `routea_rl_env_adapter_v1.py:135` | world_uid/split/role/admission_status/BC-FM SR + CI |
| Manifest loader | `:187` | split 白名单校验 (:231-232) |
| validate_rl_train_admission | `:284` | role bucket 非空 + 历史污染零 + stress 计数 |
| sample_role_balanced_worlds | `:420` | 25% easier + 50% frontier + 25% harder |
| RouteATransition | `routea_replay_buffer.py:31` | obs_feature (1536,), action_chunk (8,11), reward_terms 全量, validate() (:80) |

### 7.4 w-gates / historical_eval_only 代码门核验 (非文档)

| 门 | 判定 | 代码证据 |
|----|------|----------|
| w4/w8/w11/w12/w18 | **有真实代码门** | `routea_rl_env_adapter_v1.py:45` HISTORICAL_EVAL_WORLD_IDS + `:49-51` 正则 + `:54` check_historical_contamination (if 拦截 train/val, `:74-75`); `entrypoint_guard.py:42` EVAL_ONLY_WORLDS + `:226` assert_no_eval_only_worlds (raise RuntimeError); 调用链 `scripts/train_dppo_uav_official.py:811`, `scripts/train_a2_10k_capability.py:5621-5624` (rollout 跳过 if) |
| historical_eval_only | **代码中存在, 间接门** | `routea_rl_env_adapter_v1.py:33` NON_TRAIN_ROLES 成员; `:231-232` split 白名单合法值; `:600` gate 配置键 `forbid_actor_update_on_historical_eval_only: True`; `routea_replay_buffer.py:63` 字段注释。无字面 `if split == "historical_eval_only"` 分支 — 经 split 白名单 + contamination check 间接强制, 非 FAIL_MISSING_CODE_GATE |

### 7.5 现有测试实测 (P6 内小规模运行)

| 测试 | 结果 | 证据 |
|------|------|------|
| `python -m diffusion_policy_3d.env.test_kinematic` (mock, 无 GPU) | **4 passed / 5 failed** | MockSynthPC 缺 `rng_seed` 参数 ×4; test_collision_robustness: 中心离表面 0.2 < drone_radius 0.35 应判撞但 got False |
| `pytest diffusion_policy_3d/env/test_collision_fix.py` | **收集失败 (ERROR)** | `ModuleNotFoundError: No module named 'testpkg'` (:20) |
| `test_reward_v13.py` / `test_capacity_budget_p0_reward_wiring.py` | 未运行 (时间), 纯 CPU 可跑 | 文件存在, 11+4 测试类 |

### 7.6 三列可复用判定

**可直接复用**: `reward_v13.py` 全模块 (纯 numpy 标准实现); `traj_utils.py` + `collision_geom.py` (纯数学); KinematicUAVEnv v1 整体 (reset/step/obs/termination 完整); admission 代码门 (w-gates + historical_eval_only 机制); reward v1.3 纯 numpy 测试集。

**需要薄适配**: KinematicUAVEnv v2 (默认 v14, 需显式 `reward_version='v13'` 或只用 episode_config/provenance 机制); UAVVectorEnvAdapter (默认 v16, 依赖 RLSystemContract 合约); RouteARLWorldEntry/manifest loader (admission_manifest.json 生成物不在仓库, 需 `scripts/gen_eval_configs.py` 重生成); eval_reward_v13*.py 入口 (硬编码路径需调整)。

**不应复用**: GazeboUAVEnv (rospy/SITL 硬依赖 + 旧 PBRS v1.2); `_compute_reward` 旧 PBRS 路径 (v1.2 遗留); entrypoint_guard 的 git-clean/commit/vendor SHA256 强制链 (绑定本仓库状态, 只搬 EVAL_ONLY_WORLDS 常量 + assert 函数); v2 的 `_compute_reward_v14/v16` 与 collision_thresh=0.35 默认 (v1.3 口径为 0.10, 混用会 reward 漂移)。

注: `tests/dppo_uav_rebuild/` 与 `diffusion_policy_3d/rl/dppo_uav_rebuild/` 目前为空目录 (rebuild 落点, 非已有资产)。

---

## 8. P7 — 机器可读裁决

### 8.1 八项判定

| 裁决项 | 值 | 依据 |
|--------|-----|------|
| VENDOR_IDENTITY | **PASS** | §2: UPSTREAM_COMMIT 精确匹配; sha256sum -c exit 0; 401==401 |
| OFFICIAL_API | **PASS** | §3: 双通道 (AST+inspect) 签名; 全部解析到 vendor 内 |
| E5_CHECKPOINT_IDENTITY | **PASS** | §4.1-4.2: SHA256 匹配; 192/192 keys; cfg 字段完整 |
| E5_WEIGHT_SOURCE | **PASS (代码路径) + 部署 caveat** | §4.3: 双加载器优先 model; 部署主日志不在仓库 → 部署实际加载不推断 |
| E5_NATIVE_LOAD | **PASS** | §4.4: hydra instantiate + 0 missing/unexpected (model & ema_model) |
| E5_FIXED_SAMPLE | **PASS** | §5: 固定样本 window 0; 切片精确相等; 双重复 max abs diff = 0.0 |
| LORA_INVENTORY | **PASS** | §6: 59 模块全形状; encoder 边界; 3帧 hook 证据 |
| ENV_REWARD_REUSE_MAP | **PASS** | §7: 3 列表 + 代码门确认 (w-gates 真实, historical_eval_only 间接门) |

### 8.2 BLOCKING_FOR_CODE (编码前必须处理的接口事实)

1. **normalizer 设备契约**: 原生加载后必须 `policy.normalizer.to(DEVICE)` — `normalizer.py:352` 硬耦合 (P4 实测失败→修复证据)
2. **obs 维度契约差异**: E5 n_obs_steps=3 → global_cond 576; Route A env/replay 为 8 → 1536 (`routea_replay_buffer.py:38/82` validate 硬检查 (1536,))。rebuild 的 obs 管线必须显式对齐 (编码决策, 非推断)
3. **action chunk 契约**: (8,11) world-frame delta; n_exec_steps=4 滚动重规划; 11d 布局 pos/vel/acc/yaw/yaw_rate (`traj_utils.py:62`)
4. **测试基线非干净**: `test_kinematic.py` 4/9; `test_collision_fix.py` 收集失败 (testpkg 缺失) — rebuild 前需修复或显式排除, 否则回归基线不可信
5. **DDIM.step 无 **kwargs** (diffusers 0.36.0)**: 任何 inference-time guidance 接口不得依赖 kwargs 透传

### 8.3 NON_BLOCKING_FOR_CODE

- vendor 导入方式 (sys.path + 顶层 `model` 包) — P2 已验证可导入
- E5 设备修复不影响 checkpoint 身份 (P4 中位修复, 数据身份闭合)
- 部署运行 caveat — 代码路径证据已足够, 部署日志补录即可 (如未来可得)

### 8.4 DO_NOT_INFER

- **Gazebo 79/100 部署实际加载的权重对象** — 仓库内仅有代码路径证据 (model 优先), 无部署运行日志; 若代码后续要"复现 79/100", 必须以部署日志为准, 不得从 repo 推断
- **KinematicUAVEnv v1 vs v2 的取舍** (v13 vs v14/v16 reward 线) — probe 只报事实 (v1 默认 v13/collision 0.10; v2 默认 v14/0.35), 选型是设计决策
- **E5 策略在 DPPO 下保留还是重建 actor 的形态** — 需要设计文档决定, 非探测可推断

---

## 9. 原始日志与脚本索引

| 产物 | 路径 |
|------|------|
| 主报告 | `artifacts/dppo_uav_rebuild/preimplementation_probe_report.md` |
| 任务计划 | `artifacts/dppo_uav_rebuild/task_plan.md` |
| 日志 | `artifacts/dppo_uav_rebuild/logs/` (P0×2, P1×3, P2×1, P3×4, P4×1, P5×1; 含完整 traceback) |
| 脚本 (可复现) | `artifacts/dppo_uav_rebuild/scripts/` (native_e5.py 共享加载器 + p2/p3/p4/p5 探针) |

所有失败 traceback 未删节保留在对应日志 (P4_smoke.log 含 2 次 RuntimeError 全栈; P5_module_inventory.log 含 3 次脚本 bug 全栈; P6 测试失败见 §7.5)。

## 10. 未执行项

无。P0–P7 全部执行完毕。`test_reward_v13.py` / `test_capacity_budget_p0_reward_wiring.py` 未运行 (纯 CPU 可跑, 与 P6 判定无关, 留作 rebuild 回归基线)。
