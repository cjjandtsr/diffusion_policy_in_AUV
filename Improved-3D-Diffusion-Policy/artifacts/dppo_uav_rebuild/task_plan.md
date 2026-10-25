# Task Plan: DPPO-UAV 重建编码前探测 (阶段 A, P0-P7)

## Goal
在编码前冻结事实：官方 DPPO vendor 身份与接口、E5 checkpoint 身份与原生行为指纹、
Route A env/Reward v1.3/数据准入可复用性。交付主报告 + 原始日志。

## Repo facts (P0 recon)
- 工作根目录（所有 shell 命令从此执行）: `/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy`
- git branch: `a-official-core-r7`; HEAD: `cc9de00df00ced6bf26658f093d98d19cc14fbfc`; status: clean (recon 时)
- 上级 `/home/cjj/AUV` 非 git 仓库（环境提示 "Is a git repository: false" 源于此）
- vendor: `third_party/dppo_official_cc7234a` (存在于工作根目录下)
- E5 ckpt: `data/outputs/2026.08.16/08.53.26_e5_lambda0p5/checkpoints/latest.ckpt` (1,144,701,858 bytes)
- 报告: `artifacts/dppo_uav_rebuild/preimplementation_probe_report.md`
- 原始日志: `artifacts/dppo_uav_rebuild/logs/`

## Phases
- [x] Phase 0: Recon + plan files
- [x] Phase 1: P0 环境/仓库/依赖身份 (PASS)
- [x] Phase 2: P1 vendor 完整性 (PASS)
- [x] Phase 3: P2 官方 DPPO 接口 + 可导入性 (PASS)
- [x] Phase 4: P3 E5 checkpoint 身份 (PASS)
- [x] Phase 5: P4 E5 单样本确定性 smoke (PASS — 需 policy.normalizer.to(device))
- [x] Phase 6: P5 模块/LoRA 清单 (PASS — 59 候选模块, 3帧 history hook 证据)
- [x] Phase 7: P6 Route A env/reward/admission 盘点 (PASS — 3 列表, w-gates 真实代码门)
- [x] Phase 8: P7 裁决 + 主报告 (8 项全 PASS/代码路径; 5 BLOCKING_FOR_CODE 接口事实 + 3 DO_NOT_INFER)

## Key Questions
1. vendor UPSTREAM_COMMIT 是否 == cc7234ad7ff39a8f32de3af903606723a16f0648，SHA256SUMS 全 OK？
2. VPGDiffusion/PPODiffusion 真实签名、返回值字段、timestep 方向？
3. E5 ckpt SHA256 == c1fca4ac…？model vs ema_model 谁被 E5 79/100 部署实际加载？
4. E5 原生加载与 predict 是否确定性可复现？
5. w4/w8/w11/w12/w18 与 historical_eval_only 是否有代码门禁？
6. 八项机器可读状态，哪些 BLOCKING_FOR_CODE？

## Decisions Made
- 报告落位 artifacts/dppo_uav_rebuild/（用户指定路径，非 /tmp）
- 日志逐项落 logs/PXX_*.log；Bash 输出同时 tee 保存
- 执行纪律: 失败项不现场修复，继续后续独立项；不删改仓库文件
- P3 禁止使用 RL policy_loader.py 证明身份

## Errors Encountered
- P1 对比脚本路径前缀 ./ 不统一 → 全部显示 unlisted；sed 归一化后 401==401
- P4 eager 版 OOM (exit 137, copy_from_path 物化 5.9GB point_cloud) → 改 lazy create_from_path
- P4 设备错配 RuntimeError ×2: (1) 输入 CPU 因 policy.device=None 且 normalize 不迁移；(2) 根因 normalizer.py:352 `x.to(device=scale.device)` — ckpt normalizer 参数在 CPU → 需 policy.normalizer.to(device)
- P4 normalizer hash 不一致 → 闭合: recompute 只 fit action (6 keys), ckpt 18 keys 含 agent_pos/point_cloud identity; 重叠 6 key 全部 byte-equal; ckpt 自身 hash (同函数) 复现 1edbc81f…
- P5 hook 签名: torch 2.4 Module forward hook 无 kwargs; obs_encoder 输入是 dict
- P6 测试状态: test_collision_fix.py 收集失败 (testpkg 模块不存在); test_kinematic.py 4/9 通过 (mock 缺 rng_seed 参数 + collision_robustness 判撞失败)

## Status
**COMPLETE** — 主报告 `preimplementation_probe_report.md` 已交付 (2026-08-26)。P0-P7 全部 PASS。下一阶段: 编码前设计 (obs 维度契约 576 vs 1536 对齐、reward 线 v13 vs v14 选型、权重来源部署日志补录)。
