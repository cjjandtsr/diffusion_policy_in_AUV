# RL Rebuild Traceability Matrix

A-Official-Core design v1.0 §6.2. **Canonical rows may not be deleted or
merged; any OPEN/FAIL/BLOCKED row blocks task completion claims.**

| ID | Requirement | Source | Code target | Test | Evidence | Status |
|----|-------------|--------|-------------|------|----------|--------|
| OBS-RECENCY | 取最新3帧 | §2.2 | observation_adapter.py + all callers | sentinel test | pytest log | **PASS (2026-08-11 R3 closure; R4/G5 复验 2026-08-13)** — 关闭位置 R4/G5 复验；**R4.2 G5 v2 复验 PASS**（`G5_GATE_REPORT_v2.json`：history_indices==[5,6,7] test-asserted，1 passed in 483.44s，completed=32）；R3.6 fresh：全仓 CPU suite 462 passed（含 test_observation_recency.py 7 tests：meta.history_indices==[5,6,7]、frames [5,6,7] never [0,1,2]、shapes (B,3,4096,3)/(B,3,9)、obs_steps=1→[7]、拒绝 0/超限）+ G5 真实环境 obs=latest 3 帧断言（task_plan_t5_rework G5 run 9 GREEN） |
| PC-VALID | invalid PC不得入buffer | §4.7 | pointcloud_gate.py/env | 10k calibration | pytest log | **PASS (2026-08-09 R2 rework P1-2; rework #2 2026-08-09; rework #3 2026-08-09; rework #4 2026-08-09; R4/G5 复验 2026-08-13)** — **R4.2 G5 v2：SAFE_END=0（Guide R4.2 语义——退化带经 R2.5 far-dome + v1.1 §4.7 analytic 合同成为 VALID 帧，P10 决策点①实测关闭）+ invalid stored transitions=0（buffer invariant 6 + gate 断言）**；production-path 10k reset validity（`G3_GATE_REPORT_v6.json`：invalid_pc=0、unvalidated_fallback=0，11/11 gates，resolved_world_count=21==manifest，tree ba1f1aac）＋ 120-episode production rollout validity（`ROLE_METRICS_EVIDENCE_v4.json`：gate_invalid_count=0、invalid_frames_zero=true、transitions==env_steps=3457 → invalid buffer transition=0；rework #3 精确 refill 预算后 attempted=120==completed）＋ 10k rollout stress（`NO_RAY_STRESS_EVIDENCE_v4.json`：invalid=0）＋ rework #4 lifecycle 负向恢复路径 15 项总通过（其中 12 项属于 A/B/C 映射，另 3 项为 p7 eval 持久化回归；`LIFECYCLE_EVIDENCE_v1.json`：A reset-invalid 关闭→下一 task 成功、B AdmissionExhausted→切换、C loop truncation→slot 关闭；tree ba1f1aac）（b967036；rework #2 f83c19d/6d6a805/0fa888e；rework #3 6b80587；rework #4 5419305） |
| ACT-OFFSET | full index2开始动作区 | §2.3 | phase1a_actor_adapter.py | index sentinel | pytest log | **PASS (2026-08-11 R1 regression + R3 closure; R4/G5 复验 2026-08-13)** — 关闭位置 R4/G5/G6 复验；**R4.2 G5 v2 复验 PASS**（executed action == deployment slice test-asserted，逐 transition 0 偏差）；**R4.3 G6 v2 复验 PASS**（one-update 后 unexecuted PPO grad ≤1e-8 test-asserted，`G6_GATE_REPORT_v2.json`）；R1 regression 已发生（权威 275/0，R1_CORRECTNESS_REPORT_v2.json）；R3.6 fresh：test_action_region.py 8 tests（index sentinel：policy==full[:,2:10]、executed==policy[:,:4]==full[:,2:6]、slice_executed_logprobs [...,:4,:]、denoising_indices [4,3,2,1,0]）+ test_reward_horizon_mask.py（unexecuted 区域精确不变性 + unexecuted grad=0）+ G5 部署切片逐 transition 断言 |
| ROLE-PROP | role是一等episode字段 | §4.8 | task/env adapter | 10k resets | pytest log | **PASS (2026-08-09 R2 rework P1-2; rework #2 2026-08-09; rework #3 2026-08-09; rework #4 2026-08-09)** — production sampler 10k reset role 比例（`G3_GATE_REPORT_v6.json`：3055/3003/2940/1002，deltas max 0.6pp ≤ 2pp）＋ 120-episode attempted/completed/voided/transition 比例（`ROLE_METRICS_EVIDENCE_v4.json`：completed=120==attempted=120、voided=0、in_flight_at_collection_stop=0、transition 0.3277/0.2745/0.3063/0.0914；v2 的 attempted=123/in_flight=3 为 refill 超额物证，留史）（b967036；rework #2 f83c19d/6d6a805/0fa888e；rework #3 6b80587；rework #4 5419305，tree ba1f1aac） |
| ADMISSION | reachability 100% | §4.9 | env admission | audit | pytest log | **PASS (2026-08-09 R2 rework P1-2; rework #2 2026-08-09; rework #3 2026-08-09; rework #4 2026-08-09)** — production 10k admission audit（`G3_GATE_REPORT_v6.json`：admission_blocked=0、clearance_fail=0、reachability_fail=0、dense_path_stretch_fail=0、unvalidated_fallback=0、eval_leakage=0，11/11 gates；world_inputs_hash 只 hash resolved pool 21，非 glob 180）＋ rework #4 AdmissionExhausted→sampler 切换（`LIFECYCLE_EVIDENCE_v1.json` path B：attempt 消耗 + admission_blocked 计数 + 零部分 mutation）（b967036；rework #2 f83c19d/6d6a805/0fa888e；rework #3 6b80587；rework #4 5419305，tree ba1f1aac） |
| DPPO-PARITY | loss与官方一致 | §5.4 | official_loss_adapter.py | parity | JSON | **PASS (T3/T4 tests; R1 已闭合 2026-08-08)** — frozen official synthetic parity（T3，oracle 未修改 vendor）+ actual-batch identity（LOGP-BATCH-ID 行，G5_ACTUAL_BATCH_IDENTITY_v2.json）+ 唯一 loss closure（official_loss_adapter 单一路径）；G2 复验 |
| GAE-MASK | truncation bootstrap | §3.5 | gae.py | analytic test | pytest log | **PASS (2026-08-08 R1.3/R1.4; R4.3 G6 v2 复验 2026-08-13)** — 逐 transition next_value + TD bootstrap 仅由 terminated 屏蔽；truncated 使用自身 next_values bootstrap；递推在 episode boundary 停止（a94976a/b02f343，R1_CORRECTNESS_REPORT.json）；**R4.3 G6 真实 fixture 覆盖三种 transition（terminated=15/truncated=17/plain batch-cut=1196）+ resume advantages/returns hash 逐位一致（`G6_GATE_REPORT_v2.json`）** |
| EQUIV-REG | 不破坏既有等价性 | §5.6 | adapters | frozen fixture | `ADAPTER_REGRESSION_FIXTURE.json` + `GAZEBO_KINEMATIC_REGRESSION_REPORT.json`（g4_adapter_regression/） | **PASS (R4.1, 2026-08-13；2026-08-15 FINAL-TREE-EQUIV-REG 在返工树重跑恢复：新 fixture sha 0826fce1… gate 3/3，见 FINAL-TREE-EQUIV-REG 行)** — replacement adapter non-regression fixture（replacement_adapter_non_regression_fixture；原 fixture 永久未找到 ASSUMPTION-4；不重新证明 Gazebo 等价性 [FACT] 等价性保持）：3 确定性 case（straight_no_obstacle/yaw_lateral/z_composite，seed 7018/7001/7002，3 个 admitted rl_train worlds 各 5×(4,11) 动作步）；oracle = 冻结直接 KinematicUAVEnv v2（synth_pc_pool 默认 cuda:0，镜像 E040 build_envs 715/736-739），candidate = UAVVectorEnvAdapter（device cuda:0）；每步比较 position/yaw/vel/agent_pos/reward/termination 于设计 §5.6 阈值 allclose(rtol=1e-5, atol=1e-6) + action command 逐位 + admission 决定 cfg 精确复现；**point cloud 比较 = 设计阈值 allclose（非逐位）：冻结 CUDA synth（depth noise + dropout）实测 run-to-run 非位级稳定（cuda 60/60 复跑 ULP 偏差，max 1.6e-5 abs；cpu 0/60）——设计 §5.6 自身验收即 np.allclose，gate 记录逐帧 max_abs_diff**；no-ray 帧按 R2.5 设计差异跳过 PC 比较（position/state 仍比较）；expected 帧冻结为 .npy sidecar（内容 sha256 绑定）+ fixture/report 双 sha256 互绑；schema 测试 tests/dppo_parity/test_gazebo_kinematic_regression.py 4 passed（live 源文件 hash 新鲜度 + world/obstacles hash + sidecar 内容 hash）；证据：`/home/cjj/AUV/program/reports/routeA_rl_env_v1/g4_adapter_regression/`（fixture sha256 49f50f6f…；report PASS）|
| CKPT-AUDIT | 完整可恢复 | §4.16 | checkpoint.py | resume test | `R3_AUDIT_REPORT_v3.json`（v1/v2 历史） | **PASS (T4 tests; R3.2 3ea3bba + R3.6 fresh 2026-08-11; REWORK 2026-08-11 external-review REWORK REQUIRED → reworked)**；**external-review #2 (2026-08-11) 再裁决 FAIL → REWORK #2，rework #2 closed（a1d8e96 + 36e30c1 + 67fa369，fresh 审查 review_R3_rework2.md PASS 0 P0 / 0 P1 / 4 P2 全处置）** — v1 PASS 存疑项：LoRA 配置身份、环境 RNG、内部一致性验证缺失、config_hash 缺 8 轴、bank v2 无绑定 → schema v3（30 fields + lora_identity + environment_rng_state + config_layer；v2 显式 audit-only）+ training_identity 8 轴 config layer（base_seed/n_envs/lora_rank/lora_alpha/lr_actor/lr_value/actor_optimizer/value_optimizer）+ structured-state 一致性验证（lr_scheduler/sampling_std/reward_norm/numerical_mode）+ environment_rng_state 三键 + restore 顺序闭合（identity→lora→sampler→structured→rng→loads，sampler 校验在任何 state load 前，spy 0 次）；**rework #2 增补**：LoRA checksum 重算绑定实际张量（`lora_checksum_of(lora_state)` vs declared，P0-1a）+ wired-identity-empty-state hard-fail（P0-1b）+ key 集/参数数/shape 校验（live 无 lora_params 亦 hard-fail，review P2-1）+ env RNG 值级验证（seed_source 枚举 / restorable 真 bool / python_hash_seed 逐字符匹配，P0-2）+ reward_norm 完整 canonical dict（P1-1）+ episode outcome 类型/finite/范围（P1-2）+ _fake_checkpoint 迁移 schema v3（P1-3）+ bank canonical-content-hash 口径实证；负向证据 test_r3_rework_negative.py 39 tests + test_r3_rework2_negative.py 17 tests（"字段齐全但内容错误" negative matrix）+ 探针 5/5 REJECTED + clean-tree 全量 518 passed / 0 skipped / 7 deselected in 178.61s（`cpu_regression_r3_rework2_final_20260811.log` sha256 7b10c06f…）；**rework #3 closed（2026-08-13，RULING-7，`6f2028d`+`e2b4cd4`+`5a44d4b`+`3a367d7`+`3054eee` 等 7 commits，fresh 独立审查 `review_R3_rework3.md` PASS 0 P0 / 0 P1 / 5 P2）**——schema v4（`n_lora_param_tensors` 参数张量轴，v3 audit-only）+ 真实链正路径 + env RNG blake2b/PYTHONHASHSEED preflight/restorable 语义 + GPU 门禁 run1 RED（`gpu_gate_rework3_20260813.log` sha256 6638a77c…）→ run2 **2 passed in 4405.12s**（`gpu_gate_rework3_20260813_run2.log` sha256 fdff7d9e…）+ 定向 14/14 + clean-tree 全量 **532 passed / 9 deselected in 175.96s PYTEST_EXIT=0**（`cpu_regression_r3_rework3_20260813_final.log` sha256 515329a2…）；**external-review #4 (2026-08-13) 再裁决 FAIL → REWORK #4 进行中（P1：`validate_checkpoint_lora_usable` 未比较 `len(live lora_params)` vs declared `n_lora_param_tensors`——saved=4/live=2 被接受、policy 先加载后失败、部分 mutation，违反 validation-first；已本地对抗探针复现）**；**rework #4 closed（2026-08-13：`4b947a0` RED 4 failed in 3.45s（失败模式与裁决逐条对应）→ `1f0e5bb` live 参数张量数校验前置（不硬编码 2n）→ 定向 98 passed / 11.08s（`targeted_r3_rework4_20260813.log` sha256 `d64e6c45…`）→ clean-tree 全量 536 passed / 9 deselected / 171.04s PYTEST_EXIT=0（`cpu_regression_r3_rework4_20260813.log` sha256 `6a57a1fe…`）→ GPU 双 gate 2 passed in 4380.17s（`gpu_gate_rework4_20260813.log` sha256 `515f7bbe…`）→ fresh 独立审查 `review_R3_rework4.md` PASS-WITH-CONDITIONS（0 P0 / 1 P1 文档级 / 3 P2，条件全处置 `6bd3e91`：freeze-lift 记账 + 过时声明修正 + 计划矛盾解析））**；**R4.3 G6 v2 复验 PASS（2026-08-13，`G6_GATE_REPORT_v2.json`）**——真实 one-update 全链路（1 passed in 2203.22s：g8 optimizer state 保存/加载、g9 same-seed resume 25 metric groups + GAE hashes 一致）+ fresh-process probe（counters 1→2→3、env-steps [541,1078,1780] 严格累积、sampler attempt [16,32,48] 跨进程延续、全 finite/梯度>0）；R4/G6 复验 |
| EVAL-AUDIT | episode级paired证据 | §4.17 | evaluator | schema test | `R3_AUDIT_REPORT_v3.json`（v1/v2 历史） | **PASS (T4 tests; R3.3 c05d131+5e7c442 / R3.4 e30d835 / R3.5 77e2fa2+7c9a9af / R3.6 fresh 2026-08-11; REWORK 2026-08-11 external-review REWORK REQUIRED → reworked)**；**external-review #2 (2026-08-11) 再裁决 FAIL → REWORK #2，rework #2 closed（a1d8e96 + 36e30c1 + 67fa369，fresh 审查 review_R3_rework2.md PASS 0 P0 / 0 P1 / 4 P2 全处置）** — 冻结 paired bank v3（bank_id 绑定完整内容：`_bank_content` 顶层只排除 bank_id、bank_version 在内容内；canonical_sha256 sort_keys=True；paired_bank_from_dict 校验顺序修复 = schema gates FIRST、bank_id binding LAST 作 catch-all）+ empty-LoRA hard-fail（P0-1：eval 空 lora_state 曾跳过恢复 → validator 拒绝 "empty LoRA parameter set"）+ config_hash 自比恒等修复（P1-2：eval 从 erc.config_layer 重算 hash 比对，不再与自身比较）+ eval pool = rl_train ∪ rl_eval（P1-1：eval-only worlds w4/w8/w11/w12/w18 可达，CLI 消息 "not in the admission-manifest eval pool (rl_train ∪ rl_eval)"）+ artifact bank/density provenance（P1-3：bank_id + offline_density_sha256 入 artifact）+ **rework #2 增补**：episode outcome 类型/finite/范围验证（success/collision/timeout 严格 bool、return 有限数值、steps 非负 int；return=NaN 拒绝，review P2-3 超大 int 恒有限不转换 float）+ bank canonical-content-hash 口径实证（`test_bank_id_survives_formatting_changes`：语义/内容变化 → 绑定变化，格式变化（缩进/键序）不破坏）；canonical eval 入口 scripts/eval_dppo_uav_official.py；**rework #3 closed（2026-08-13，RULING-7，fresh `review_R3_rework3.md` PASS 0 P0 / 0 P1 / 5 P2）**——GPU 门禁 gate 2 canonical paired eval（真实 identity facts + 冻结 bank + 官方 eval 入口 subprocess + artifact 全绑定：checkpoint hash / LoRA checksum / bank_id / density sha256 / 逐 record frozen hash / TRACE_STEP_REQUIRED_KEYS / projection False，6 paired episodes，bank 11ff61c8e8f7）+ `_eval_episode` unnormalize CUDA→host 修复（`3054eee`，run1 RED TypeError cuda:0→numpy）；R4/G7–G10 复验 |
| LOGP-BATCH-ID | 实际 PPO minibatch 无更新时 old/new logprob 恒等 | §5.7/R1 | train_loop.py + numerics.py + phase1a_actor_adapter.py | `test_actual_batch_logprob_identity.py` + `test_g5_real_no_update_gpu.py` | `G5_ACTUAL_BATCH_IDENTITY_v2.json` + `G5_GATE_REPORT_v2.json` | **PASS (2026-08-08 R1.2 rework + rework #2; R4.2 G5 v2 复验 2026-08-13 SUPERSEDED；**R4 返工重跑 G5 v3 2026-08-15 恢复 PASS（新 kernel 下 identity max_abs_diff=0.0、ratio=1.0、SAFE_END=0，`G5_GATE_REPORT_v3.json`，pytest exit 0）**：真实 32 eps no-update 全路径 identity max_abs_diff=0.0、ratio_mean=1.0、zero-adv pg_loss≤1e-7、completed=32/voided=0/SAFE_END=0、REJECT/EXCLUDE/HARD=0，1 passed in 483.44s)** — 官方 element-level randperm + microbatch=1 likelihood recompute + 行为策略 CFG λ；GREEN：B=1 max_abs_diff=0.0、1_randperm=0.0、B=128=4.57763671875e-4 informational（v1 运行值；rework #2 运行值 3.662109375e-4）、ratio mean=1.0、actor update norm=0（85d304e→61d2b77→6680cd7；阈值未放宽；rework #2 closure：权威运行 275/0 EXIT=0 + 证据 v2，见 Evidence ledger） |
| GAE-EP-BOUNDARY | terminated/truncated/SAFE_END 不跨 episode 传播 | §3.5/R1 | gae.py | analytic pytest + fixture JSON | `R1_CORRECTNESS_REPORT.json` | **PASS (2026-08-08 R1.3)** — continuation mask = same-episode ∧ ¬terminated ∧ ¬truncated；episode_id 变化即使 mask 误置 False 仍阻断递推；buffer 末条 sampling cut continuation=0（a94976a + 4 解析测试） |
| NEXT-VALUE-PER-T | 每条 transition 使用自身 next_obs bootstrap | §3.5/R1 | trainer.py | analytic pytest | `R1_CORRECTNESS_REPORT.json` | **PASS (2026-08-08 R1.4)** — `_stack_obs(buffer.next_obs)` 全量 critic forward，next_values shape=(T,) 逐 transition；spy critic 手算 GAE expected 独立验证两 episode 泄漏探针（b02f343 + test_episode_boundary_stops_advantage_leak） |
| SAMPLER-LIVE | production train loop 实际调用 role sampler | §4.8/R2 | train_loop.py + environment_adapter.py | caller test + trace | `G3_GATE_REPORT_v6.json` | **PASS (2026-08-09 R2.1/R2.2, rework P1-1; rework #3 2026-08-09)** — train_loop `run_e2e_batch` 经注入 `task_sampler.next_attempt()` 采样（train_loop.py:809），不自建 sampler；gate audit 用同一 factory（gate_audit.py:124）；`test_r2_6_contracts.py` 5/5 静态断言同路 + 10k G3 11/11；rework #4：per-slot admission + AdmissionExhausted 捕获后仍走同 sampler `next_attempt()`（`LIFECYCLE_EVIDENCE_v1.json` path B）（5458504→4eb696c；rework P1-1 四 gate 化 + gates_ok 全查 + 负向测试 af70e2d；证据 v6，tree ba1f1aac） |
| SEED-RESUME | global episode/attempt/seed/cursor 可恢复且不重复 | §4.8/R2 | environment_adapter.py | fresh-process JSON | `G3_GATE_REPORT_v6.json` | **PASS (2026-08-09 R2.1/R2.3, rework P1-3; rework #3 2026-08-09)** — EpisodeTaskSampler 全局 attempt 计数器 + `(base_seed, global_attempt_index)` 确定性派生 episode_seed（environment_adapter.py:318-422）+ snapshot/restore（`_attempt` 恢复）；fresh-process 恢复测试 test_episode_task_sampler.py；10k 唯一 seed 序列（G3 11/11）；rework P1-3：sampler_state 持久化入 checkpoint + 强制恢复 + 真两进程 continuation JSON（RED 实测 run2 seed 复用 [1000,1001] → GREEN，6e5dd93）；rework #4：AdmissionExhausted 不消费 global seed 进度外状态、slot 保持可 reset（`LIFECYCLE_EVIDENCE_v1.json` path B）（5458504→dece2d9→6e5dd93；证据 v6，tree ba1f1aac） |
| ROLE-TRANSITION | transition 与 completed/voided episode 比例分别记录 | §4.8/R2 | metrics.py | metrics JSON | `ROLE_METRICS_EVIDENCE_v4.json` | **PASS (2026-08-09 R2.2, rework P1-2; rework #2 2026-08-09; rework #3 2026-08-09)** — `role_stats_report`（metrics.py）按 role 记录 attempted/completed/voided/transition 比例（train_loop.py:964-970,1076）；role 一等 episode 字段经 `next_attempt().task_spec.role`（4eb696c）；production-path 120-episode JSON 实测（真实 4 槽，requested=resolved_n_envs=4）：completed=120==attempted=120、voided=0、in_flight_at_collection_stop=0、transitions=3457==env_steps、transition proportions 0.3277/0.2745/0.3063/0.0914、7 invariants 全 True（b967036；rework #2 f83c19d/6d6a805/0fa888e；rework #3 6b80587 精确 refill 预算；rework #4 5419305 —— `ROLE_METRICS_EVIDENCE_v4.json`：completed=120==attempted=120、voided=0、in_flight=0、transitions==env_steps=3457、7 invariants 全 True、role 38/34/37/11，tree ba1f1aac；v2 attempted=123/in_flight=3 为超额物证留史；旧引用 G3_GATE_REPORT_v2 仅有 reset 计数、无 role metrics，audit P2-10 指出并替换；v1 证据因 n_envs 接线错误作废，见 Evidence ledger） |
| META-SOURCE-HASH | manifest source entry hash 与实际内容一致 | §4.8/R2 | environment_adapter.py + integration.py | unit/audit JSON | `G3_GATE_REPORT_v6.json` | **PASS (2026-08-09 R2.3, rework P1-6; rework #3 2026-08-09)** — `canonical_sha256` 统一 manifest source hash；`manifest_entry_hash` 为 source-entry hash 与 `episode_config_hash`（运行时）拆分绑定；rework P1-6：canonical payload 覆盖全部 spec-input 字段 + 逐字段 sensitivity 测试（533e497）；rework #3 终版（6b80587）：**全字段无白名单**（raw manifest 全部固定字段，仅排除 manifest_entry_hash 自身）+ resolved_role/resolved_admission_id/resolved_generator_version/resolved_obstacle_path=`<uid>.world.obstacles.json`（与 TaskSpecRegistry 默认及实际加载路径一致）+ 无 default=str（非 JSON 字段硬失败）；G3 10k hash_missing=0、manifest_hash_mismatch=0（dece2d9→533e497→6b80587；rework #4 5419305 —— G3 v6 hash_missing=0、manifest_hash_mismatch=0，tree ba1f1aac） |
| META-EPISODE-HASH | 实际 start/goal/yaw/seed/admission hash 可复算 | §4.8/R2 | environment_adapter.py | unit/eval JSON | `G3_GATE_REPORT_v6.json` | **PASS (2026-08-09 R2.3; rework #3 2026-08-09)** — EpisodeConfigRecord + `episode_config_hash = canonical_sha256({identity, episode_seed, start, goal, start_yaw})`，train_loop 逐 episode 写入；可复算测试 test_role_sampling.py；rework P2-9：reset info episode_id 统一为 EpisodeConfigRecord 唯一 ID（7d99f08）；rework #4：两阶段 reset_at 的 record 构建在 admission 阶段只读完成（`LIFECYCLE_EVIDENCE_v1.json` path B 零部分 mutation）（dece2d9→7d99f08；证据 v6，tree ba1f1aac） |
| VECTOR-ORDER | n_envs>1 不丢环境且 env_id 顺序稳定 | §4.10/R2 | environment_adapter.py | vector test | `G3_GATE_REPORT_v6.json` | **PASS (2026-08-09 R2.4, rework P2-7/P2-9; rework #3 2026-08-09)** — 真实同步 vector：`reset_at(env_ids)` 只重置指定 slot、`step` 全 slot 同步、done slot 必须 `reset_at` 才可步进；结果按 env_id 0..n_envs-1 顺序返回；rework P2-7：step() 统一 preflight 原子化（不再逐 slot 半推进）（7d99f08）；rework #3（6b80587）：slot lifecycle 四态 INITIAL/ACTIVE/DONE/VOIDED —— reset_at 只接受 INITIAL/DONE/VOIDED、ACTIVE 在任何 mutation 前原子 RuntimeError、mark_voided(env_ids)/step_at(env_ids, actions)/abort_episode(env_ids)；rework #4（5419305）：train loop + adapter 生命周期闭合（裁决重开项）—— per-slot 两阶段原子 reset_at（admission 全 slot 只读 → 全部通过才 mutation，AdmissionExhausted 零部分 reset）、reject_episode/mark_truncated 显式 ACTIVE→DONE、is_active() 守卫防 double-close、refill budget 精确递减（per-slot refill list，修复 n_envs>n_episodes 超额启动）；10 lifecycle tests + 2 p7 eval 恢复路径 pins，`LIFECYCLE_EVIDENCE_v1.json` 15 项总通过（其中 12 项属于 A/B/C 映射，另 3 项为 p7 eval 持久化回归；tree ba1f1aac）；test_vector_env_ordering.py + test_real_adapter.py + test_lifecycle_rework4.py + test_p7_eval_persistence.py（b8e7227→7d99f08→6b80587→5419305） |
| NO-RAY-SEMANTICS | 退化 fallback 不入 buffer 且 pilot invalid=0 | §4.7/R2 | pointcloud_gate.py + validated_synth_adapter.py | fixture + calibration | `NO_RAY_ROOT_CAUSE_REPORT.json` + `NO_RAY_STRESS_EVIDENCE_v4.json` | **PASS (2026-08-09 R2.5, rework P1-4/P1-5; rework #3 2026-08-09)** — 冻结 fallback 的 all-far 单点折叠（synth_pointcloud.py:489-500）被 ValidatedSynthAdapter 物理 far-endpoint 编码取代（DEGENERATE_ENCODING，解析边界 z>10·sin31.41°≈5.211m）；gate 契约：hit_count=0 仅 analytic RayGeometryReport 证据下 valid（设计 v1.1 §4.7 合同裁决，hit_count>0 OR analytic 通过；far-dome 合法化），单点 all-far 永远 invalid（unique floor + allclose-far 双保险）；stress 10,000 帧全在 adapter/rollout 路径 invalid=0 / far-dome=1,970 / report_failures=0 / resets=466（NO_RAY_STRESS_EVIDENCE_v4.json，版本化 harness scripts/no_ray_stress.py，tree ba1f1aac；rework #3：cadence reset 命中 ACTIVE 前 abort_episode 显式丢弃，6b80587；rework #4 5419305：`NO_RAY_STRESS_EVIDENCE_v4.json` invalid=0 / far-dome=1,970 / report_failures=0）（3db5778→c4545b6→6b80587→5419305） |
| CRITIC-STATE-ONLY | value checkpoint 不重复嵌入 policy | §4.13/R3 | critic.py | key-set test | `R3_AUDIT_REPORT_v3.json`（v1/v2 历史） | **PASS (2026-08-11 R3.1 5a5748d)** — PolicyObsEncoder 以非注册强引用持有 policy（object.__setattr__；weakref GC 过期已修）；ValueCritic 新增 value_state_dict()/load_value_state_dict()（仅 mlp.*，strict=True + 非 mlp key 拒绝）；_build_checkpoint_data 与 restore_training_state 切换 value-only API；value checkpoint 不再内嵌 policy keys |
| RESUME-HASH-GATE | contract/manifest/BC/code/config mismatch hard-fail | §4.16/R3 | checkpoint.py | negative tests | `R3_AUDIT_REPORT_v3.json`（v1/v2 历史） | **PASS (2026-08-11 R3.2 3ea3bba + R3.6 fresh; REWORK 2026-08-11 external-review REWORK REQUIRED → reworked)**；**external-review #2 (2026-08-11) 再裁决 FAIL → REWORK #2，rework #2 closed（a1d8e96 + 36e30c1 + 67fa369，fresh 审查 review_R3_rework2.md PASS 0 P0 / 0 P1 / 4 P2 全处置）** — schema v2 → **schema v3**（30 fields + lora_identity + environment_rng_state + config_layer；v2 显式 audit-only）+ pre-load identity validation：8 组 hash 字段 mismatch 均在 policy load_state_dict 前 hard-fail（spy proof 0 次 load）；**config_hash 覆盖 training_identity 8 axes**（base_seed/n_envs/lora_rank/lora_alpha/lr_actor/lr_value/actor_optimizer/value_optimizer，任一变化 → hash 改变；base_seed 对无该属性 sampler 记录 None）；**sampler 身份校验提前到任何 state load 之前**（restore 顺序 identity→lora→policy-non-empty→lora-usable→sampler→structured→rng→loads，sampler state-load spy 0 次）；schema v1 拒绝；空 policy_state 拒绝；**rework #2 增补**：LoRA checksum 重算绑定 + wired-empty-state + key/参数数/shape 全部在任何 state load 前（spy 0 次）+ env RNG 值级验证（含 python_hash_seed 逐字符匹配 live 进程）+ reward_norm 完整 canonical dict — 负向 25 functions / 39 collected tests（rework #1）+ 17 tests（rework #2 negative matrix）+ 探针 5/5 REJECTED；R3.6 fresh-process 24 passed + rework 负向 25 functions / 39 collected tests（fresh 合计 63）；**rework #3 closed（2026-08-13，RULING-7，fresh `review_R3_rework3.md` PASS 0 P0 / 0 P1 / 5 P2）**——schema v4 校验链（key 集命名契约 + `len(lora_state) == n_lora_param_tensors` + 逐张量 shape vs live lora_params，全在任何 state load 前）+ env RNG 值级验证升级（restorable 真 bool + True-only + python_hash_seed 逐字符）+ `restorable=True` 仅固定非空 PYTHONHASHSEED 时记录 + 入口 preflight hard-fail；GPU 门禁 gate 1 双 resume（outer_iteration [1,2,3]，计数器/optimizer/sampler cursor 与连续世界逐项一致，25 metric groups）；**external-review #4 (2026-08-13) 再裁决 FAIL → REWORK #4 进行中（同上 P1：live LoRA 参数张量数未三方校验）**；**rework #4 closed（2026-08-13：live 计数校验在任何 shape 遍历/state load 前（`1f0e5bb`）+ 4 负向测试 spy 零 load（`4b947a0`）+ fresh 独立审查 `review_R3_rework4.md` PASS-WITH-CONDITIONS 条件全处置，证据见 Evidence ledger (R3 rework #4 additions)）**；**R4.3 G6 v2 复验 PASS（2026-08-13）**——G6 fresh-process probe 在正式 resume 链上实测 counters/seed/cursor 恢复（`G6_GATE_REPORT_v2.json` probe counter chain [1,2,3] + sampler attempt [16,32,48]） |
| LEGACY-FORMAL-GUARD | 所有旧入口 subprocess formal 非零退出 | §8.2/R3 | entrypoint_guard.py | closure report | `R3_AUDIT_REPORT_v3.json`（v1/v2 历史） | **PASS (2026-08-11 R3.3 c05d131+5e7c442)** — 全部 7 个 legacy 脚本 subprocess `--mode formal --dry_run` 非零退出（test_canonical_entrypoints.py 参数化：RED 先证 E040 exit 0 → GREEN 8/8）；E040 最小 guard 含 hash 前后入账（pre `7868d1be…` → guarded `c548efc2…`，ENTRYPOINT_CLOSURE_REPORT.md §1）；test_all_legacy_scripts_wire_the_guard |
| EVAL-STD | eval 使用 resolved sampling/logprob std | §3.4/R3 | train_loop.py + evaluator.py | spy test | `R3_AUDIT_REPORT_v3.json`（v1/v2 历史） | **PASS (2026-08-11 R3.4 e30d835)** — _eval_episode 双 floor 从 LoopConfig 透传（min_sampling/min_logprob）；SpyPolicy RED 证实 eval 静默走 policy default 0.01 → GREEN（test_eval_std_passthrough 2 tests）；std 由 checkpoint 自身 exact_resolved_config 驱动 |
| CAPABILITY-AUDIT | 三类能力指标来自 episode traces，非 placeholder | §4.17/R3 | evaluator.py | artifact test | `R3_AUDIT_REPORT_v3.json`（v1/v2 历史） | **PASS (2026-08-11 R3.5 77e2fa2+7c9a9af; REWORK 2026-08-11 external-review REWORK REQUIRED → reworked)**；external-review #2 (2026-08-11) PASS 保留，episode artifact 完整性已补（rework #2 P1-2 闭合）— capability_metrics.py（d_soft^H/onset/trap/side_flip/support cell 全参数 + OfflineDensity frozen dataclass）+ build_runtime_eval_artifact 以 compute_capability_metrics 替换 placeholder（删 _CAPABILITY_PLACEHOLDER）+ offline density 缺失 hard-fail（eval 前置 RuntimeError / CLI exit 2）+ validator 升级（每组 required keys/finite 非 bool/n_episodes role 对齐/sr·cr∈[0,1]）；**NaN/Inf 不再被掩码成有效指标（P1-4）：capability_metrics + evaluator validator 对任一必需 trace 字段 NaN/Inf 拒绝 artifact**（移除 np.isfinite 掩码生成有效指标的路径）；**rework #2（P1-2）episode outcome 聚合层同标准闭合**：return 有限数值（int 恒有限，review P2-3）、steps 非负 int、success/collision/timeout 严格 bool，负向测试 + 探针 NAN_EPISODE_RETURN 拒绝；test_capability_metrics 27 passed（R3.6 fresh 实测）；**rework #3 closed（2026-08-13，RULING-7，fresh `review_R3_rework3.md` PASS 0 P0 / 0 P1 / 5 P2）**——GPU 门禁 gate 2 真实 artifact 能力组非空断言（anticipatory_avoidance / fullspace_support_expansion / dense_obstacle_ceiling 三组）+ episode outcome 聚合层标准在 GPU 路径实测闭合 |
| GATE-ORDER | G3→G10 无跳跃 | §5/R4 | gate scripts | gate ledger | `G10_GATE_REPORT.json` | OPEN — R4 关闭（每 gate 固定 commit/tree/hash）；**2026-08-14 外部审查裁决 `NO-GO for G8/G9`（review `RouteA_R0R4_完整性官方差异与后续准入审查_v1_0_20260814.md`）：R4/G5/G6/G7 证据 SUPERSEDED_BY_R5_TRANSITION_KERNEL_FIX，R5.0–R5.7 修复后按 R5.7 顺序重跑 G4→G7** |

| TRANSITION-KERNEL-OFFICIAL-PARITY | rollout/update 与冻结官方 DPPO 同 transition distribution（每 DDIM index mean/std/logprob/sample parity） | 审查 §4.1/§8 K1-K4 | `dppo_transition_kernel.py` + `test_transition_kernel_vendor_parity.py` | vendor tensor oracle | pytest log | **PASS (R5.1, 2026-08-14)** — 新建 `diffusion_policy_3d/policy/dppo_transition_kernel.py`（vendor p_mean_var DDIM 语义 + 双独立 std floor + index 表，无 t_prev terminal 分支）；`dp_sampler.py`/`chain_logprobs.py` 删本地公式（K6 source gate）；`phase1a_policy.py` sampling 循环接入（freeze-lift 记账：85ac9f0 `3871b8de…` → `04c72cde…`）；vendor parity **22/22（K1-K4 全 10 index rtol=1e-5/atol=1e-6，含最后一步 floored std）**；K5 all-k grad coverage **1 passed（3 fixtures × 5 index 全 finite 非零）**；全量 CPU **570 passed / 2 skipped / 9 deselected** |
| ALL-FT-DENOISE-GRAD-COVERAGE | 每个拟训练 denoise index（k=0..4）对 LoRA 有 finite 非零梯度 | 审查 §4.6/§8 K5 | `test_all_denoise_grad_coverage.py` | controlled fixtures | pytest log | **PASS (R5.1, 2026-08-14)** — 3 fixtures × 5 FT index 全 finite 非零 LoRA grad（1 passed in 3.05s）；trainer 的 grad-less-cleanest 旁路已删除（strict autograd.grad）；`test_cleanest_step_minibatch_has_real_gradient` 断言 cleanest-only minibatch executed grad>0 |
| BASE-UNET-GRAD-ABSENCE | base UNet requires_grad=False；backward 后 .grad is None；optimizer param set 精确为 LoRA | 审查 §4.2/§8 O1-O4 | `lora_adapters.py` + `test_base_unet_grad_absence.py` | grad absence test | pytest log | **PASS (R5.2, 2026-08-14)** — insert_lora 冻结全部非 LoRA 参数（autograd 级，官方 diffusion_vpg.py:82-85 语义）+ identity report（base_params_frozen/trainable_params）；test_base_unet_grad_absence 4/4（O1 base .grad is None、O2 LoRA finite 非零、O3 optimizer param IDs 精确相等、O4 一步后 base 逐位不变 LoRA 变化）；全量 CPU 574 passed / 2 skipped / 9 deselected |
| PAIRED-POLICY-RNG | paired eval 两侧使用同一 policy-noise 序列；artifact 记录 RNG contract | 审查 §4.5/§8 E1-E3 | `paired_rng.py` + `test_paired_policy_rng.py` | identical-policy sentinel | pytest log | **PASS (R5.4, 2026-08-15)** — `policy_noise_seed = uint64(SHA256(bank_id||entry_id||replica)[0:8])`（审查 §7 R5.4 公式）；`paired_rng_context` 每侧隔离（全局 RNG 保存→32 位投影播种→恢复；env 自身 RNG 不受扰）；EpisodePairedRecord + artifact 增 `policy_noise_seed/replica_id/rng_contract_version`。测试 3/3：E1 identical-policy sentinel（同权重同噪声 → outcome+逐 trace 字段位级一致，elapsed 墙钟除外）、E2 派生公式、E3 artifact 合同。全量 CPU 580 passed / 2 skipped / 9 deselected（唯一 FAIL = G4 fixture 新鲜度，R5.7 重建）。**REOPEN 处置（2026-08-17, `92aa0f4`）**：canonical eval 传入真 bank_id（此前空串派生，artifact 不可重推导）；`validate_eval_artifact` 从 (bank_id, episode_config_hash, replica) 重算每条 seed 并逐条比对；真 GPU identical-policy sentinel `test_r58e_paired_sentinel_gpu.py`（真 Phase1a+真 bank，两侧同 policy → per-episode delta 全 0） |
| EXACT-TRANSITION-BUDGET | 每 seed 精确相同 environment-transition budget；rollout_cut 区分 truncated；active slot 跨 update 延续 | 审查 §4.3/§8 B1-B5 | `train_loop.py` + `environment_adapter.py` + `checkpoint.py` + `test_exact_transition_budget.py` | exact-budget test | pytest log | **PASS (R5.3 机制, 2026-08-14)** — `LoopConfig.target_transitions`（while 条件内联预算检查，杜绝陈旧 flag 过冲）；预算边界 = rollout_cut（cut transition truncated=True 供 GAE bootstrap，env slot 保持 ACTIVE）；per-slot `state_dict/load_state_dict`（env 运动学/history/RNG + adapter bookkeeping，真实 adapter 与 mock 双实现）；in-process 采用（adoption）+ checkpoint schema v5 `active_slot_states` 跨更新精确恢复；entry `--target_transitions` 累计停止。测试 4/4（B1 精确 25=25、B2 cut 槽保持 ACTIVE、B4 restored vs uninterrupted 逐 transition 位级一致、B5 状态往返）；全量 CPU 577 passed / 2 skipped / 9 deselected（唯一 FAIL = G4 fixture 新鲜度——R5.7 FINAL-TREE-EQUIV-REG 重建后转绿）。**训练侧预算数字待 G7 重预注册（R5.7）冻结**。**REOPEN 处置（2026-08-17, `6054bca`）**：P0-1 修复（`--target_transitions` 曾启动即 UnboundLocalError，功能整体死亡——resume 计数先于 LoopConfig 加载；逐 batch `compute_batch_target` 重建；`--rollout_transitions_per_update`；超调用 hard-fail）；P0-2 修复（进程内 live ACTIVE slot 直采不调 load_state_dict）；[8,8,8,1] 验收 + 同 env resume 回归测试 |
| CHECKPOINT-RETENTION-AND-DISK-PREFLIGHT | 轻量 resume ckpt（不含重复 base）+ release ckpt + 原子写 + retention + 启动磁盘 preflight | 审查 §4.4/§8 S1-S5 | `checkpoint.py` + `test_checkpoint_retention.py` | stress JSON | pytest log | **PASS (R5.5 机制, 2026-08-15)** — resume ckpt 剥离 base model.* keys（冻结 base 由 BC ckpt 重新物化，bc_checkpoint_hash 身份绑定；restore strict=False + missing 必须 ⊆ model.* 硬门禁）；release ckpt 在 final batch 物化完整 policy（entry `checkpoint_kind=release`）；`CheckpointBuilder.save` 原子写（tmp+fsync+replace）；entry `_retain_checkpoints`（latest 2 轮转）+ `_disk_preflight`（projected + 20GiB reserve > free 即启动前 hard-fail）。测试 4/4（S1 轻量子集+release 完整、S2 原子、S3 retention 6→2、S4 preflight 双向）。全量 CPU 584 passed / 2 skipped / 9 deselected（唯一 FAIL = G4 fixture 新鲜度，R5.7）。真实大小节约由 R5.7 G7 重跑实测记录。**REOPEN 处置（2026-08-17, `7eaa547`）**：final release 改 completion 基准（预算提前达成也有可评估 release，不再依赖 batch_idx==max_updates）；metrics.jsonl sidecar 跨进程保护 release 不被轮转；jsonl fsync + 撕裂行容错；canonical eval 对 resume-kind 明确报错 |
| FINAL-TREE-EQUIV-REG | R4 返工完成后用最终 tree/hash 重跑 G4 adapter non-regression | 审查 §2/§7 | G4 gate scripts | final-tree fixture replay | `GAZEBO_KINEMATIC_REGRESSION_REPORT.json` | **PASS (2026-08-15)** — 返工树（tree cde87a65）上重建 fixture（sha 0826fce1…）+ gate 重放 **3/3 PASS**；schema/新鲜度测试 4/4 转绿（返工改动的源文件已在新 fixture 重新固定）；R4.1 旧证据的 SUPERSEDED 标记随之解除 |

## Evidence ledger

- 2026-08-06 T1: `pytest tests/dppo_parity/ -v` → 22 passed (test_contract 8,
  test_observation_recency 7, test_action_region 7). Persistent evidence:
  `program/shiyan/dppo_parity_t1_20260806.log` (committed).
- 2026-08-06 T1 review gate: PASS-WITH-CONDITIONS (3 conditions + 2 warnings
  fixed; reviewer adf7f23978c7e599a). No blocking items.
- 2026-08-06 T2: `pytest tests/dppo_parity/ -v` → 58 passed (T1 22 +
  pointcloud gate 15 + role/admission 18 + review-fix 3). Persistent
  evidence: `program/shiyan/dppo_parity_t2_20260806.log` (committed).
- 2026-08-06 T2 review gate: PASS-WITH-CONDITIONS (reviewer
  a34b1f7027b42fc31). Conditions fixed: manifest counts 15/18 + t1 log in
  changed_files; formal-phase rate gate removed (design §4.7 table has none);
  known_gaps adjudicated by user (A* source + rule-6 hash-proxy); warnings
  fixed (decide() phase validation, reset() type hints, deque). 58 passed.
- 2026-08-06 T3: `pytest tests/dppo_parity/ -q` → 89 passed (prior 58 still
  green + buffer 10 + GAE 6 + official parity 11 + F10 mask 4 = 31 new).
  Persistent evidence: `program/shiyan/dppo_parity_t3_20260806.log`.
  Test-body fixes at GREEN (3 sites, arithmetic/assertion defects — see log).
  Oracle = unmodified `third_party/dppo_official_cc7234a` PPODiffusion.loss().
- 2026-08-06 T3 review gate: PASS-WITH-CONDITIONS (reviewer
  a544f9b52fa5b4b80). C1 fixed: known_gaps + unexpected_changes adjudicated
  by user (all approved) → ASSUMPTION-7. W1/W2 fixed: rollback now reverts
  both commits (55925bd 4b71a2a); result_project_commit = 55925bd. 89 passed.
- 2026-08-06 T4: `pytest tests/dppo_parity/ -q` → 124 passed (prior 89 still
  green + value loss 5 + trainer 6 + checkpoint 8 + metrics 7 + eval artifact
  9 = 35 new). Persistent evidence:
  `program/shiyan/dppo_parity_t4_20260806.log` (committed).
  Implementation fixes at GREEN: checkpoint base64 (torch 2.4.1
  weights_only=True rejects raw bytes), metrics std correction=0 (declared
  matches-numpy contract). Test-body fixes at GREEN (4 sites — see log):
  FakeActor interface, oracle oldlogprobs (B, Ta, Da) selection, clip-branch
  semantics (official is elementwise max), frozen-dataclass mutation →
  dataclasses.replace (6 tests).
- 2026-08-06 T4 review gate: PASS-WITH-CONDITIONS (independent reviewer,
  no blocking items). Conditions + warnings fixed (see evidence log §7b):
  C1 manifest rollback → `git revert a72c8bd 8270c6d` (was orphaned
  pre-amend hash); W1 randperm re-permuted per epoch (official :311);
  W2 field-count labels 22→25; W3 +2 eval hard-fail tests (all 8 §4.17
  rules covered); W4 critic-warmup/eta omission recorded in known_gaps;
  N1 roundtrip asserts contract_serialized bytes; N2 DPPO-PARITY cites
  T4 value-loss tests. All 6 adjudication items user-approved →
  ASSUMPTION-8. 126 passed (cf9eb7b).
- 2026-08-06 T5: closure audit + e2e integration → all 18 dppo_uav
  entry points have callers in the loop (ENTRYPOINT_CLOSURE_REPORT.md).
  `pytest tests/dppo_parity/ -q` → 160 passed (CPU 158 + GPU 2; GPU =
  real BC ckpt `data/outputs/2026.05.17/13.36.20_v8_phase2_sweep_lambda0.5/
  checkpoints/latest.ckpt` + LoRA r=8 + mock env, no-update + one-update
  single-batch forward+backward, peak < 8 GiB). Evidence:
  `program/shiyan/dppo_parity_t5_20260806.log` (committed).
  DPPO-PARITY: per-dim per-step chain-window logp parity-anchored to
  compute_per_step_logp_from_chain (2 parity tests). CKPT-AUDIT /
  EVAL-AUDIT: runtime contract hash at the §4.17 eval call site
  (ASSUMPTION-8 item 6), validated in the GPU run.
  GPU-caught defect: dp_sampler `_model_output_to_epsilon`
  prediction_type='sample' batched-t broadcast (crash at B=20; B=1
  passed by accident) — fixed with RED test, scalar-t path unchanged
  (recorded as unexpected_change, §6.4).
- 2026-08-06 T5 review gate (G1): **PASS** — independent code-reviewer,
  no conditions, no blockers. All 7 checks re-run by execution:
  frozen files untouched; oracle byte-identical (diff -rq zero differs,
  SHA256SUMS 401/401); py3.8 AST clean on 17 files; 160 passed (GPU
  pair 2 passed in 19.27s); TDD narrative coherent + dp_sampler RED
  spot-checked (pre-fix reproduces the identical B=20 RuntimeError);
  6 known-gap items located; code quality (py3.8 typing, frozen
  dataclasses, no bare except, no print, 11d preserved).
  T5 known_gaps + unexpected_changes → §6.4 user/Zread adjudication
  (pending, mirroring the ASSUMPTION-8 flow).
- 2026-08-06 **T5 completion REVOKED by adjudication (ASSUMPTION-9)**:
  self-audit `program/review/review_T5_design_conformance_20260806.md`
  found A1–A6 deviations + B1–B3 design contradictions; user adjudication
  adds two CRITICAL defects (N1 shared-obs-encoder updated by critic
  optimizer; N2 checkpoint saves the pre-update LoRA snapshot). The
  "design text not on disk" premise is false — design v1.0 exists on disk
  (1622 lines, SHA256 4ede81feecbde9a1580110b431e36ff8bcd01d7f2303320a5e2284bff4049407).
  Status statement: T1–T4 components + unit tests retained; T5 integration
  and completion claim fail; system = "component-level scaffold + mock
  single-batch integration". Rework order: N1/N2 → B1 mean-pool critic →
  A2–A6 contracts → T2-ext real env wiring + A* → repository-wide closure →
  G3 → G4 → G5 → G6 → T6/G7 micro-train → T7/G8 multi-seed. No micro-train
  or formal training before G3–G6 structured evidence.

## Decisions

- **R5-RULING-1** (2026-08-14, external review v1.0 `RouteA_R0R4_完整性官方差异与后续准入审查_v1_0_20260814.md`, locally verified finding-by-finding and ACCEPTED): **`NO-GO for G8/G9`** — F1 P0 (transition kernel 把有效 timestep 0 当 terminal，2/5 denoise 步零 logprob/零梯度，两个 seed 全部 batch denoise 3/4 KL=0.0 实证), F2 P1 (base UNet 无 requires_grad=False), F3 P1 (无固定 transition budget), F4 P0-OPS (292.6MB/ckpt、磁盘剩 19GB、1.46TB 不可持续), F5 P1 (paired eval 未配对 policy RNG), F6 P1 (G7 gate 检不出死步), F7 P2 (denoise index 命名歧义), F8 (objective 风险)。G5/G6/G7 旧 closure = SUPERSEDED_BY_R5_TRANSITION_KERNEL_FIX；G7 旧运行 = DIAGNOSTIC_INVALID_FOR_G7_PASS。修复按 R5.0–R5.7（§7）+ §8 gate 矩阵 + §11 ENGINE-CORRECT DoD，分支 `a-official-core-r5`。

- **ASSUMPTION-4** (2026-08-06, user-accepted): `exec_steps=4` written into
  `RLSystemContract` as the de-facto training value (kinematic env
  `n_exec_steps=4`, E040 executed `[0:4)`). Not equivalence-certified —
  EQUIV-REG stays OPEN until the T9 Gazebo re-freeze CI produces a fixture.
- **ASSUMPTION-5** (2026-08-06, user-adjudicated): A* planner source = port
  of pre-v2 `kinematic_uav_env.py:558 _astar_3d` into `dppo_uav/planner.py`
  at T7, with backup + docs (review gate F1: v2 env has no A* symbols).
- **ASSUMPTION-6** (2026-08-06, user-adjudicated): Admission rule-6 =
  hash-proxy semantics — metadata_hash covers sampled start/goal/role;
  TaskSpecRegistry requires non-empty hash; equality against the manifest
  record is the consistency check.
- **ASSUMPTION-7** (2026-08-06, user-adjudicated, review gate C1): T3
  known_gaps + unexpected_changes recorded as-is. Gaps: (1) compute_gae
  accepts `truncated` but never masks bootstrap — by design ([PROVEN] §3.5),
  truncated drives env-reset bookkeeping in T4; (2) buffer consumption path
  (batching for the update loop) is T4; (3) trainer integration of
  compute_gae + dppo_policy_loss is T4/T5. unexpected_changes: 3 test-body
  defects fixed at GREEN (GAE mid-buffer hand values A_1=-2.0 / ret[1]=0.0;
  identity assertions compared whole dict to array; invariant1/2 fixtures now
  thread obs=t0.next_obs) — reviewer re-derived and confirmed all three.
- **ASSUMPTION-8** (2026-08-06, user-adjudicated, review gate C1): all 6
  T4 known_gaps/audit findings accepted as-is: (1) entropy term = 0
  placeholder — phase1a actor contract exposes logprobs only (official
  entropy head not ported), ent_coef defaults 0 so inert; (2) bc_loss off —
  official use_bc_loss=False default; (3) executed_vs_unexecuted_grad_norm
  is 0 by construction under the F10 mask — recorded as a mask-regression
  detector, not an independent learning signal; (4) §4.16 field count
  reconciled 21 categories → 25 fields (schema_version + 24; several
  categories expand to two fields; design text not on disk); (5) official
  critic-warmup gate + learn_eta machinery not ported — phase1a has no
  warmup gate and no eta head; (6) validate_eval_artifact keeps the
  expected_contract_hash default (fail-closed: forgetting the runtime hash
  hard-fails, never silently passes); T5 call site must pass the runtime
  RLSystemContract hash.
- **ASSUMPTION-9** (2026-08-06, user/Zread adjudication, supersedes T5
  completion): **T5 COMPLETE/PASS revoked**. The self-audit found the
  "design text not on disk" premise false (design on disk, 1622 lines,
  SHA256 `4ede81feecbde9a1580110b431e36ff8bcd01d7f2303320a5e2284bff4049407`).
  Verdicts, all binding:
  - **A1 CRITICAL**: critic 576-concat deviates from E040 mean-pool
    (3×192 → mean pool → 192→128→64→1); hidden 128→192 changed without
    record. **B1**: first version must return to the E040 mean-pool
    baseline; permutation test replaced by latest-window sentinel
    ([0..7]→[5,6,7]) + observation content sensitivity + recorded
    permutation-invariance of mean pool; temporal critic is a separate
    ablation only after G6/G7. Critic obs encoder frozen/detached; critic
    optimizer contains value head params only.
  - **A2 HIGH**: §7.4 obstacle projection must be in the checkpoint/eval
    contract (`use_obstacle_projection == False`), hard-fail when True.
  - **A3 CRITICAL**: closure report lines 7/8/9 mislabel
    TaskSpecRegistry/RoleSampler/AdmissionChecker as "closed T5" —
    train_loop.py has 0 references; whole report failed the design's
    repository-wide entry audit (must be rewritten repository-wide).
  - **A4 HIGH**: sampling vs logprob std must be two explicit parameters
    (official: `min_sampling_denoising_std` / `min_logprob_denoising_std`).
  - **A5 MEDIUM**: history_indices (observation-selection metadata) must
    enter trace/eval artifact.
  - **A6 HIGH**: per-batch/epoch role proportion statistics must be
    recorded (RoleSampler.proportion_report must have a caller).
  - **N1 CRITICAL**: critic loss updates the shared BC obs encoder —
    `critic.parameters()` includes policy submodule params, GPU test uses
    `AdamW(critic.parameters())`, test_critic.py:91-102 asserts the bug.
    Fix: freeze the policy obs encoder; critic encoder forward under
    no_grad/detach; critic optimizer = value head params only; verify all
    base params update norm=0 before/after one-update.
  - **N2 CRITICAL**: checkpoint saves the pre-update LoRA snapshot —
    `extract_lora_state` clones before the update and LoRA adaptors are
    forward-hook closures invisible to `policy.state_dict()`. Fix: save the
    live post-update LoRA state; verify update norm>0, checkpoint LoRA ==
    live LoRA, fresh-process load reproduces action/logprob, resume
    reproduces the next update numerically.
  - **B2**: design §4.5 field definitions revised to five distinct names
    (`chains_full` (B,K_f+1,H_f,11) / `logprobs_full` (B,K_f,H_f,11) /
    `chains_policy` (B,K_f+1,H_a,11) / `old_logprobs` (B,K_f,H_a,11) /
    `executed_action` (B,A_e,11)); implementation's H_a choice confirmed.
  - **B3**: §8.2 task table is authoritative (T5 closure+e2e / T6 micro-
    train / T7 multi-seed / T8 formal / T9 Gazebo); real env wiring +
    TaskSpecRegistry + RoleSampler + AdmissionChecker + A* move into a
    **T2 extension work package**, completed and accepted before real
    G5/G6; "integrated in T7" comments in environment_adapter.py are
    implementer-written, not design text.
  - **G4**: Gazebo—kinematics equivalence stays [FACT]; BLOCKED means only
    "equivalence holds historically but the frozen fixture is not found" —
    never "not yet proven equivalent". Find the original fixture; if
    permanently lost, a new hashed non-regression fixture requires a
    separate user decision.
  - RTM: every "PASS (…pending T7)" row is OPEN/BLOCKED (this table).
  - Missing symbol `HISTORICAL_INVALID_FOR_FORMAL_TRAINING` must be added
    as formal-mode rejection protection for legacy E-series entry points.
  - Eval: `run_e2e_batch` must persist the paired-eval JSON; capability
    metrics are placeholders; eval rollout needs its own point-cloud
    validity gate; history_indices / obstacle projection contract / role
    counts missing.
  - Gating order: G3 (real 10k calibration) → G4 (equivalence
    non-regression) → G5 (real-env no-update) → G6 (real one-update +
    base freeze + live LoRA checkpoint + optimizer resume) → T6/G7
    micro-train → T7/G8 multi-seed. No micro-train or formal training
    before G3–G6 structured evidence.

## Evidence ledger (R0 additions)

- 2026-08-07 **MasterPlan v1.0 adjudication (R0 reopen)**:
  `program/zhidao/Guide_A_OfficialCore_Completion_MasterPlan_v1_0.md`
  declares **REWORK REQUIRED; G5 FAIL; G6 REOPEN; T6/G7 未解锁**,
  superseding the previous G5/G6 "GREEN" closeouts (`7d4a387`,
  `77568e7`, `76166b8`) and the P10 handover
  `program/jiaojie/A-OfficialCore_G6门禁_交接_20260807.md`: those were
  accepted under semantics that do not satisfy the design — G5 identity
  verified only at batch=1 (B=128 measured `max_abs_diff≈0.0831`, a
  batch-size kernel bias), GAE uses a scalar final `next_value` and does
  not stop recursion at episode boundaries, the train loop derives
  seeds from attempt index, and the vector adapter runs only slot
  `[0]`. All gates must be re-run from the real production path with
  corrected semantics per `program/zhidao/Guide_A_OfficialCore_R{1..4}*.md`
  (R1 PPO/GAE → R2 sampler/env → R3 checkpoint/entrypoint/eval → R4
  re-gates G4/G5/G6 + G7–G10).
- 2026-08-07 **R0 baseline recorded**: implementation branch
  `a-official-core-rework` from HEAD `0f92f35` (tree
  `90b2b9c4abd8cc9683e290a960758ce55e90b0d0`); vendor
  `third_party/dppo_official_cc7234a` SHA256SUMS 401/401 OK; E040
  frozen 5 files sha256 verified identical to the closure-report
  baseline (`d1168142…/3673e1b2…/c5a164bb…/3871b8de…/7868d1be…`) with
  zero diff vs freeze commit `85ac9f0`; design v1.0 on disk is now
  SHA256 `6d540098fa40f461668aa68a3444d489c41a48dacf1569df0fe8a1d076b3ce6c`
  (1641 lines — the B2-amended version; ASSUMPTION-9 quoted
  `4ede81fe…` for the pre-P6 text). Repo root has no AGENTS.md/CLAUDE.md.

## Evidence ledger (R1 additions)

- 2026-08-08 **R1.1 RED**: `pytest -m gpu -vv tests/dppo_parity/test_actual_batch_logprob_identity.py` → FAIL at baseline: B=128 max_abs_diff=0.128997802734375 (batch-size kernel bias, RED log `temp/r1_actual_batch_identity_red.log`，2026-08-10 temp 清理时删除，关键数字保留于本行); baseline JSON `temp/r1_actual_batch_identity_before.json` fixed fields (TF32-on, deterministic off). Committed e896e20.
- 2026-08-08 **R1.2 rework** (user ruling, see ASSUMPTION-11): official element-level randperm restored; deferred B=128 old-logprob pass deleted; stored old logprobs computed at rollout time B=1 with the behavior policy's CFG λ; ppo_update recomputes the likelihood at microbatch=1; optimizer batch stays 128 with full B=128 advantage normalization + single backward (gradient accumulation). CPU suite `pytest tests/dppo_parity/ -q -m "not gpu"` → **237 passed, 7 deselected** (`program/reports/routeA_rl_env_v1/official_core_r1/cpu_regression_r1_20260808.log`，原 `temp/r1_cpu_suite.log`，2026-08-10 迁入); oracle `test_official_loss_parity + test_reward_horizon_mask + test_chain_logprobs` → **24 passed**; vendor `third_party/dppo_official_cc7234a` zero-diff (`git diff --exit-code HEAD~4`).
- 2026-08-08 **R1.3** (a94976a): `compute_gae` new API (per-transition next_values + episode_ids), 4 analytic tests with hand-computed constants, scalar final-next-value branch deleted. `pytest -vv tests/dppo_parity/test_terminated_truncated_gae.py` GREEN.
- 2026-08-08 **R1.4** (b02f343): trainer `_stack_obs(buffer.next_obs)` full critic forward → `next_values_np.shape == (T,)`; spy-critic episode-leak probe `test_episode_boundary_stops_advantage_leak` hand-computes all 6 advantages independently (a5 = g·7.0−1002.0 plain-cut bootstrap, A_2 = 0.3+g·1000.0−3.0 truncated own-bootstrap, leak probe discriminative >1.0).
- 2026-08-08 **R1.2 GPU identity gate GREEN**: `G5_ACTUAL_BATCH_IDENTITY.json` — rollout_logprob_batch_size=1, policy_cfg_lambda=1.5, ppo_update_batching="official element-level randperm + microbatch=1 likelihood recompute"; B=1 max_abs_diff=**0.0**, 1_randperm=**0.0** (composition gate), B=128=4.57763671875e-4 (informational, kernel dependence recorded not asserted), ratio mean=1.0 all batches, cfg_mixing_effect=152.18 (CFG wiring live), actor_parameter_update_norm=0.0, TF32-off + deterministic. Design §5.7 thresholds unchanged (1e-5, [0.99999, 1.00001]).
- 2026-08-08 **R1.5 review gate**: independent review `review_R1.md` — PASS (thresholds unchanged; actual likelihood path B=1 microbatch real; old logprobs written once at rollout, never overwritten; GAE hand-computed expected independent; vendor zero-diff). R1 completion gate: LOGP-BATCH-ID / GAE-MASK / GAE-EP-BOUNDARY / NEXT-VALUE-PER-T all PASS. **（注：同日独立审计裁决后本 PASS 作废——见下方 R1 rework closure 条目，新审查 review_R1_rework.md 取代）**
- 2026-08-08 **R1 rework closure (independent audit ruling)**: audit declared R1 REWORK REQUIRED (per-minibatch randperm coverage loss; fake microbatch accumulation; unbounded next-value critic forward; CFG λ / B=1 contract gaps; evidence not bound to code) and voided the R1.5 PASS. Fixed in `2ea2a0f`+`61d2b77` (frozen code_result_commit 61d2b77, tree 5af1494, worktree clean at run): per-epoch OFFICIAL randperm + contiguous slicing (vendor :306-311); TRUE per-element forward/backward gradient accumulation (grads /B, one optimizer step, B128-level adv norm once then norm_adv=False); order-preserving chunked next-value critic; cfg_lambda AttributeError hard-gate (no silent fallback); rollout/likelihood B=1 hard-fail. New gates: test_randperm_epoch_coverage / test_microbatch_equivalence / test_next_value_chunking / test_cfg_contract. **Authority run (no cache)**: `python -m pytest -q tests/dppo_parity/ -p no:cacheprovider` → **259 passed / 0 failed / EXIT=0** (`/tmp/r1_authority_final.log`, sha256 ab12f2cb…, 2026-08-08T07:30:55Z→08:25:59Z); warning audit 26/2 classes accepted (std ddof<=0 informational; no-ray-hit no gate bypass); stability: RNG-prefix 19, integration ×5 processes, merged 32, utf-8 all read points, locale regression kept. Evidence resealed: closure commit `ba81ac5` (manifest full audit fields; review_R1.md VOID); fresh independent review `review_R1_rework.md` — **PASS 9/9 (A–I)** (committed `dbf518f`). **R1 = PASS**; next = R2.
- 2026-08-08 **R1 rework #2 closure (second independent audit ruling, narrow scope)**: audit declared R1 narrow-scope REWORK REQUIRED again — (a) CFG override: `sample_dppo_rollout`/`make_get_new_logprobs` accepted an external cfg_lambda, enabling a fake ratio (chain at policy λ=1.5, old logprob at override λ); (b) checkpoint exact config lacked policy_cfg_lambda / rollout_policy_batch_size / likelihood_microbatch_size / next_value_batch_size; (c) `ratio.std()` NaN + ddof warning at B=1 (21 warnings in the sealed run; summary text "value 0.0" inaccurate); (d) doc corrections (Guide v1.1 superseding v1.0 B=128 identity clause; RTM numeric unification; R1_阶段总结 reset--hard removal). RED `4f71798` → fixes `6680cd7` (frozen code_result_commit 6680cd7, tree 17e4e6b6, worktree clean at run): formal interfaces drop cfg_lambda, λ resolved exclusively via `_require_policy_cfg_lambda` (non-finite or != 1.5 → ValueError); pure-cond contrast only via low-level `per_dim_denoising_logprobs`/`per_step_denoising_logp`; loader verifies runtime instance λ==1.5; checkpoint persists policy_contract + next_value_batch_size configured/resolved (round-trip test); TrainerConfig validates batch_size>1 and next_value_batch_size>0; ratio_std=0.0 at numel==1 (informational). **Authority run (no cache)**: → **275 passed / 0 failed / EXIT=0** (`/tmp/r1_rework2_authority.log`, sha256 a34cea96a9…, 2026-08-08T10:09:26Z→11:07:24Z); warning audit 8/1 class accepted (synth no-ray-hit x8 GPU gates only) — ddof/NaN std class eliminated (21→0), zero CFG-related warnings; G5 v2: B=1 and 1_randperm max_abs_diff=0.0, B=128 3.662109375e-4 informational, thresholds unchanged (1e-5, [0.99999, 1.00001]); integration fixture action_scale 0.01→0.0002 (CFG-live cond_proj amplification, test-only). Evidence v2 (sealed v1 preserved as history): `implementation_manifest_R1_v2.json`, `G5_ACTUAL_BATCH_IDENTITY_v2.json`, `R1_CORRECTNESS_REPORT_v2.json`, `review_R1_final_v2.md`; rollback = reviewed `git revert` only.

## Decisions (R1 additions)

- **ASSUMPTION-11** (2026-08-08, user-adjudicated, supersedes the R1.2 arange-chunk fallback decision): **fixed arange chunks are NOT the R1.2 final fix.** Binding rulings: (1) keep TF32-off/deterministic numerical mode; (2) restore the OFFICIAL element-level randperm (randperm(N·K_f) + unravel_index, per epoch and per batch — official train_ppo_diffusion_agent.py line 300-309); (3) the rollout chain is produced by a B=1, CFG λ=1.5 behavior policy, so old/new likelihoods MUST use the same canonical CFG computation path and be computed at the actual behavior batch B=1 — including the policy's own cfg_lambda in the logp path (user-confirmed: "logp 路径接行为策略的 CFG λ"); (4) optimizer batch stays 128 via actor-likelihood microbatch=1 + full B=128 advantage normalization + gradient accumulation — official PPO math preserved; (5) FORBIDDEN: post-hoc B=128 redefinition of buffer old logprobs (deferred pass deleted), deletion of the randperm composition gate, loosening the 1e-5 / [0.99999, 1.00001] thresholds; (6) per-transition GAE (R1.3/R1.4) is wired only after R1.2 passes; (7) no changes this phase to reward, environment, Gazebo/dynamics, role targets, official vendor, or the three capability acceptance criteria. Execution: 85d304e; evidence G5_ACTUAL_BATCH_IDENTITY.json.

## Decisions (R0 additions)

- **ASSUMPTION-10** (2026-08-07, user-adjudicated): **design v1.1 document
  is NOT written.** The R1–R4 implementation guides
  (`program/zhidao/Guide_A_OfficialCore_R{1..4}*.md`) already carry every
  adjudicated errata item that v1.1 was to record (actual-batch identity,
  GAE dual mask, production sampler, dual hash, single/vector staging,
  SAFE_END/no-ray, G4 replacement fixture, unique entrypoint); the user
  directed execution to proceed directly from the guides. The MasterPlan
  v1.0 remains the authoritative execution plan; the design v1.0 (B2
  amended) remains authoritative for algorithm semantics. Final-closeout
  §8 deliverables: the v1.1 item is replaced by this decision record.

## Evidence ledger (R2 rework additions)

- 2026-08-09 **R2 REWORK REQUIRED (independent audit ruling, supersedes the
  R2 closure `a6f9cab`)**: 10 findings — P1×6: (1) G3 gate dead counters —
  `admission_blocked` never enters `gates_ok()`, fallback==0 constant,
  hash mismatch masked by `AdmissionExhausted` (PASS under total block);
  (2) RTM PC-VALID/ROLE-PROP/ADMISSION still OPEN + manifest
  requirement_ids lists 7 of 10; (3) SEED-RESUME lacks production resume —
  checkpoint.py has no sampler_state; resume probe builds a fresh sampler;
  (4) no-ray stress = 9k wrapper-level + 1k adapter frames, not 10k rollout
  frames, harness uncommitted; (5) no-ray implementation lacked design-layer
  adjudication (design v1.0 §4.7 requires hit_count > 0); (6)
  entry_metadata_hash covers only 7 fields (admission_id/generator_version/
  obstacle_json/resolved-role changes invisible). P2×4: (7) step()
  non-atomic (slot0 advances before a later slot errors); (8) episode
  collection over-fill — inactive slots filled ignoring remaining budget;
  (9) reset info episode_id = spec.admission_id non-unique; (10) closure
  evidence unbounded (R2.6 "PENDING", no SHAs, harness uncommitted,
  worktree not clean; ROLE-TRANSITION cites a reset-count-only report with
  no production role-metrics JSON). Correctly-completed parts retained
  (role bucketing/cursor, eval-only hard-fail, dual hash, vector ordering,
  no-ray semantics, regression 313/0/7). All 10 verified RED before fixing
  (details: task_plan_r2_rework.md).
- 2026-08-09 **R2 rework fixes (TDD RED→GREEN)**: P1-1 `af70e2d` (G3 gates →
  11 gates: admission_blocked / manifest_hash_mismatch / dense_path_stretch_fail
  added; gates_ok() consults all; negative tests + non-zero exit); P1-6
  `533e497` (source-entry canonical payload = ALL spec-input fields +
  per-field sensitivity tests); P2-7/8/9 `7d99f08` (step() unified
  preflight atomicity; n_episodes % n_envs != 0 startup hard-fail; unique
  EpisodeConfigRecord episode ids); P1-3 `6e5dd93` (sampler_state persisted
  in checkpoint + forced restore + true two-process continuation JSON — RED
  measured run2 seed reuse [1000,1001], GREEN after fix); P1-4 `c4545b6`
  (versioned harness `scripts/no_ray_stress.py`, all 10,000 frames on the
  adapter/rollout path). Design-layer adjudication P1-5: design v1.1 §4.7
  contract ruling (`hit_count>0 OR (hit_count==0 AND
  analytic_ray_geometry_passes)`; far-dome legitimate) —
  `program/sheji/RouteA_RL_Training_System_A_OfficialCore_Design_v1_1.md`,
  review PASS no P0/P1 (`program/review/...v1_1_Review.md`), registered in
  sheji/README.
- 2026-08-09 **R2 rework evidence suite (all at HEAD `b967036`, tree
  `c77d355`, worktree clean at run)**:
  - G3 10k rerun v3: 11/11 gates PASS; 10,000 resets / 80,000 frames; all
    counters 0 (admission_blocked, clearance_fail, dense_path_stretch_fail,
    eval_leakage, hash_missing, invalid_pc, manifest_hash_mismatch,
    reachability_fail, role_unknown, unvalidated_fallback); role
    0.3055/0.3003/0.2940/0.1002 vs 0.30/0.30/0.30/0.10, max delta 0.6pp ≤
    2pp — `program/reports/routeA_rl_env_v1/g3_gate/G3_GATE_REPORT_v3.json`
    (sha256 `561c2ed3cb1b0bb3d0e1b4e2da241adc1b35071981d985e779d41b0a744f6922`).
  - no-ray stress v2: 10,000 frames, all adapter/rollout path, invalid=0 /
    far-dome=1,970 / report_failures=0 / exit 0 —
    `NO_RAY_STRESS_EVIDENCE_v2.json` (sha256
    `2367d72ae5ec304ae929489bb4e8e8e2f1f0113bf77294a07bb60a87fe0fab3a`).
  - role metrics probe: 120 episodes (4 envs × max_ep_len 30), production
    path (EpisodeTaskSampler → UAVVectorEnvAdapter → PointCloudValidityGate
    → run_e2e_batch role_stats, no_update; CPU FakePolicy action generator
    NOT under evidence): PASS — attempted=completed=120, voided=0,
    transitions=3468==env_steps, gate_invalid_count=0, proportions
    0.3167/0.2833/0.3083/0.0917 — `ROLE_METRICS_EVIDENCE.json` (sha256
    `be6e7a4e6e88140c99498f35e836d3705a3bf7e24bd7df7975b51f7daa347058`).
    Harness `scripts/role_metrics_probe.py` + 3 tests
    `tests/dppo_parity/test_role_metrics_probe.py` (`b967036`).
  - CPU regression: **330 passed / 0 failed / 7 deselected (GPU)**,
    112.13s — `program/reports/routeA_rl_env_v1/g3_gate/regression/cpu_regression_rework1_20260809.log`
    (原 `temp/r2_rework_cpu_suite.log`，2026-08-10 自 temp/ 迁入) (sha256
    `006f22cf9f17d318430339add66b7e47ed83b9a31edfca90233145cae9822070`).
  - vendor `third_party/dppo_official_cc7234a` 401 files zero-diff; E040
    frozen 5 files untouched; no policy/PPO-loss/reward/env changes in R2
    scope.

## Evidence ledger (R2 rework #2 additions)

- 2026-08-09 **R2 closure rework #2 (independent audit ruling, supersedes
  the rework #1 closure as far as the affected evidence claims)**: 3
  findings — P1: (1) role-metrics probe claimed n_envs=4 but
  `build_adapter` was never passed `n_envs` (UAVVectorEnvAdapter default
  1) — the sealed `ROLE_METRICS_EVIDENCE.json` ran a single-slot adapter;
  `max_episode_steps=30` was also hard-coded, ignoring `max_ep_len`; the
  existing tests checked JSON fields only, never the real adapter's
  `n_envs`. P2: (2) G3 report lacked self-binding — no tree_hash/command/
  exit_code/manifest_sha256/world_inputs_hash; `n_frames=80000` read as
  80k independently validated frames but is an 8-row obs window per
  validated reset; (3) manifest changed_files=49 vs actual
  `git diff --name-only c1f7dc1..HEAD` = 52 (missing review_R2_rework.md /
  task_plan_r1_rework2.md / task_plan_r2.md); the fresh review compared
  against 1a5f063, inconsistent with the manifest's declared c1f7dc1
  baseline. All 3 verified [FACT] in code before fixing.
- 2026-08-09 **R2 rework #2 fixes (TDD RED→GREEN)**: P1 `f83c19d`
  (`build_adapter(..., n_envs=n_envs, max_episode_steps=max_ep_len)` +
  hard-fail when resolved env.n_envs != requested + evidence records
  requested_n_envs/resolved_n_envs + completeness invariants
  completed==n_episodes, attempted>=completed+voided); P1 follow-up
  `0fa888e` (truthful multi-slot completeness: the loop stops at the
  collection bound and drops still-running episodes — measured
  attempted=123 > completed=120, tail drop recorded as
  accounting.in_flight_at_collection_stop, never masked; the first
  version's attempted==n_episodes assertion was sequential-loop
  semantics, structurally false under n_envs>1); P2 `6d6a805` (G3 report
  self-binds provenance: tree_hash/command/exit_code/manifest_sha256/
  world_inputs_hash + honest frame accounting validated_reset_frames=
  n_resets, observation_window_rows=n_frames=resets*8; any provenance
  failure hard-fails, no report written). 5 new tests, all RED first.
- 2026-08-09 **R2 rework #2 evidence suite (all at HEAD `0fa888e`, tree
  `7502dfaf`, worktree clean at run)**:
  - G3 10k rerun v4: 11/11 gates PASS; 10,000 validated resets / 80,000
    observation-window rows; all 10 counters 0; role
    0.3055/0.3003/0.2940/0.1002 vs 0.30/0.30/0.30/0.10, max delta 0.6pp ≤
    2pp (identical to v3, seed-0 deterministic) —
    `program/reports/routeA_rl_env_v1/g3_gate/G3_GATE_REPORT_v4.json`
    (sha256
    `fba26bcbd7453d5dd0e8c3b80eccddb2f9e7810e9441fe22bf03c7ef31db711a`).
  - role metrics probe v2: 120 episodes, REAL 4-slot adapter
    (resolved_n_envs=4 == requested, wiring verified at construction):
    PASS — completed=120==n_episodes, attempted=123,
    in_flight_at_collection_stop=3 (multi-slot tail drop, recorded
    truthfully), voided=0, rejected=0, transitions=3532==env_steps,
    gate_invalid_count=0, transition proportions
    0.3208/0.2755/0.3078/0.0960 — `ROLE_METRICS_EVIDENCE_v2.json`
    (sha256
    `be244ce03e6361d1359f777d71ae1c05f39cb4c095d8aa00e423e1ae3a00d6e8`).
    The old `ROLE_METRICS_EVIDENCE.json` is VOIDED (n_envs wiring bug:
    it ran 1 slot while claiming 4; its self-consistent numbers were an
    artifact of the single-slot sequential semantics).
  - CPU regression: **335 passed / 0 failed / 7 deselected (GPU)**, 89.68s
    — `program/reports/routeA_rl_env_v1/g3_gate/regression/cpu_regression_rework2_20260809.log`
    (原 `temp/r2_rework2_cpu_suite.log`，2026-08-10 自 temp/ 迁入) (sha256
    `f68cd090fbabb738716a0170fafb57663ea24bef4a6d2393c1fae60e0572b6fd`).
  - vendor 401 files zero-diff; E040 frozen 5 files untouched; no
    policy/PPO-loss/reward/env changes (train_loop untouched — the
    multi-slot tail-drop is R2.4's existing collection-bound semantics).
  - R3 前置风险（本次未改，冻结环境不触碰）: kinematic_uav_env_v2.py
    payload 仍含进程随机化 `hash(str(world_id))`（注释声称稳定 hash），
    restore_rng_states() 不恢复/验证 PYTHONHASHSEED —— 不推翻 SEED-RESUME
    （sampler seed 不复用已证），但"真实环境跨新进程 observation bitwise
    resume"暂不可宣称，列为 R3 environment RNG / resume 前置门禁。

## Evidence ledger (R2 rework #3 additions)

- 2026-08-09 **R2 rework #3 (user audit ruling, supersedes the rework #2
  closure as far as the affected evidence claims; R2 未完整闭合前不得进入
  R3)**: 3 P1 + 3 P2, all verified [FACT] in code before fixing — P1-1
  refill over-shoot + silent tail-drop (train_loop checked only completed
  and filled ALL inactive slots with no remaining-budget-minus-active
  accounting; the v2 evidence attempted=123 for n_episodes=120,
  in_flight_at_collection_stop=3, is the physical artifact); P1-2
  `reset_at` allowed resetting ACTIVE slots (no slot-state check,
  violating the guide contract 只重置 done/voided); P1-3 source-hash
  whitelist missed 15 real production manifest fields, `default=str`
  masked missing values, obstacle_json resolved to None while the world
  loader uses `<uid>.world.obstacles.json`; P2-a manifest self-exclusion
  invalid (v4 must list all files including itself); P2-b design v1.1
  header still 待审查; P2-c G3 world_inputs_hash hashed a glob superset
  (180 obstacle files) instead of the resolved pool (21).
- 2026-08-09 **R2 rework #3 fixes (TDD RED→GREEN, 11 test functions, all
  RED first)**: P1-1 `6b80587` exact refill budget — `budget = n_episodes
  - completed - active_count` (completed+active ≤ n_episodes at every
  step), attempt_limit gates the refill inner loop, exit invariant
  active=0 and attempted==completed+voided exactly; P1-2 slot lifecycle
  INITIAL/ACTIVE/DONE/VOIDED — reset_at accepts only INITIAL/DONE/VOIDED
  (ACTIVE atomic RuntimeError before any mutation), mark_voided(env_ids),
  step_at(env_ids, actions), new abort_episode(env_ids) explicit
  ACTIVE→DONE discard used by gate_audit/no_ray_stress audit-type
  resets; P1-3 full normalized source entry — ALL raw manifest fixed
  fields + resolved_role/resolved_admission_id/resolved_generator_version/
  resolved_obstacle_path (`<uid>.world.obstacles.json`, matching the
  TaskSpecRegistry default and the loaded path), no whitelist, no
  `default=str`; P2-c g3 hashes ONLY the resolved pool (empty pool
  hard-fails) + resolved_world_count in provenance.
- 2026-08-09 **R2 rework #3 evidence suite (fix commit `6b80587` = tree
  `e303d160`, committed before running; closure HEAD `4eea765`; sealed
  v1-v4 evidence untouched)**:
  - G3 10k rerun v5: 11/11 gates PASS; 10,000 validated resets / 80,000
    observation-window rows; all 10 counters 0; role 0.3055/0.3003/0.2940/
    0.1002 vs 0.30/0.30/0.30/0.10, max delta 0.6pp ≤ 2pp; provenance
    resolved_world_count=21==manifest_world_count=21, world_inputs_hash
    over the resolved pool only (e1539c9a, independently recomputed by
    the fresh review) —
    `program/reports/routeA_rl_env_v1/g3_gate/G3_GATE_REPORT_v5.json`
    (sha256
    `693707efbef38fd647f986e83eb0260cd9bc76f1dcf59e0fc13c96861bdb706e`).
  - role metrics probe v3: 120 episodes, REAL 4-slot adapter
    (requested=resolved_n_envs=4): PASS — completed=120, attempted=120,
    voided=0, in_flight_at_collection_stop=0, transitions=3457==env_steps,
    gate_invalid_count=0, 7 invariants all True, role 38/34/37/11 —
    `ROLE_METRICS_EVIDENCE_v3.json` (sha256
    `b611b04d3c31f258f2b200638f4925272a808c61e1d340b5a283b9fe4550be96`).
    The rework #2 `_v2.json` (attempted=123, in_flight=3) is retained as
    the P1-1 over-shoot artifact.
  - no-ray stress v3: 10,000 frames, all adapter/rollout path, invalid=0 /
    far-dome=1,970 / report_failures=0 / resets=466 (cadence+terminal),
    early_terminations=67 (legitimate disturb) —
    `NO_RAY_STRESS_EVIDENCE_v3.json` (sha256
    `ee3da4f164840cbe26f44001e443175ed9d98a0fa4804f7b725ee1f07a9547d8`).
  - CPU regression: **345 passed / 0 failed / 7 deselected (GPU)**,
    88.10s — `program/reports/routeA_rl_env_v1/g3_gate/regression/
    cpu_regression_rework3_20260809.log` (sha256
    `0cd90fee66422eef274dcd1a5e06f6be07144831945cb255148edb09bc83bb41`).
  - GPU/PPO core untouched (ruling: not required — rework #3 touches no
    GPU/PPO path); vendor 401 files zero-diff; E040 frozen 5 files
    untouched. Fresh independent review `review_R2_rework3.md`: **PASS
    0 P0 / 0 P1 / 4 P2 observations**; manifest v4 changed_files 58 = 57
    diff + itself (P2-a, no self-exclusion).

## Evidence ledger (R2 rework #4 additions)

- 2026-08-09 **R2 rework #4 (user audit ruling, supersedes the rework #3
  closure for the affected claims; R2 stays REWORK REQUIRED until a fresh
  independent review of the rework #4 tree passes — no R3 entry, no
  formal RL training before that)**: 8 of 10 requirements keep
  conclusions; reopened and re-verified this round: VECTOR-ORDER
  (production train loop + adapter lifecycle closure), ADMISSION
  (AdmissionExhausted → sampler switches world/task, never
  fixed-coordinate fallback, design v1.1:823), PC-VALID (reset-invalid
  recovery paths). 3 P1 blockers, all verified [FACT] in code before
  fixing — P1-1 reset-invalid leaves the slot ACTIVE (train_loop.py:862
  reset_at → environment_adapter.py:903 ACTIVE → gate reject →
  train_loop.py:882 only `continue` → next refill hits the ACTIVE
  preflight RuntimeError; mock_uav_env.py:106 lacked the ACTIVE guard so
  tests missed it); P1-2 AdmissionExhausted uncaught (exception ended
  the batch; reset_at did admission then mutation per-slot so a mid-batch
  failure left partial mutations — an outer except is insufficient); P1-3
  loop max_ep_len (train_loop.py:997) only set local `s["done"]=True`,
  the adapter slot stayed ACTIVE, and the final transition's truncated
  could be False. Doc-closure scope: manifest v4 declared 58 files vs
  actual diff 59 (review_R2_rework3.md missed); RTM carried two
  R2-REWORK-RULING-5 entries with conflicting tail-episode rulings
  (rework #3's renumbered to RULING-7); 9aec72a landed after the rework
  #3 fresh review — covered by the rework #4 fresh review below.
- 2026-08-09 **R2 rework #4 fixes (TDD RED→GREEN, 10 lifecycle tests +
  2 p7 eval recovery pins, all RED first)**: fix commit `5419305`
  (tree `ba1f1aac`, committed before running evidence) — per-slot
  admission / two-phase atomic reset_at (Phase 1 builds admission +
  episode-config records READ-ONLY for ALL slots; AdmissionExhausted
  raised before any mutation → zero partial resets; Phase 2 applies
  mutations); AdmissionExhausted caught in the train loop (attempt
  consumed, admission_blocked counted, slot stays resettable, sampler
  advances via next_attempt()); explicit `reject_episode(env_ids)`
  (ACTIVE→DONE) after REJECT_EPISODE; `mark_truncated(env_ids)` +
  `effective_truncated` for loop-level max_ep_len truncation
  (`loop_trunc = (s["step"]+1) >= max_ep_len`, preserving original
  post-increment semantics); EpisodeCollectionIncomplete gains
  admission_blocked_episodes; eval path closes invalid reset frames via
  reject_episode and invalid step frames via mark_voided (is_active()
  guarded, never double-closed); refill budget decrements exactly
  (per-slot refill list; n_envs > n_episodes over-launch fixed);
  gate_audit blanket `except RuntimeError: pass` replaced by explicit
  is_active() queries; g3 script `except Exception` narrowed to
  (SubprocessError, OSError). Mocks mirror the real adapter's semantics
  exactly (ACTIVE guard included).
- 2026-08-09 **R2 rework #4 evidence suite (fix commit `5419305` = tree
  `ba1f1aac`, committed before running; sealed v1-v5 evidence untouched;
  all runs SEQUENTIAL — the earlier 3-script concurrent launch on 7GB
  RAM / 20 cores caused 80x step-cost inflation and was killed and
  cleanly re-run, see task_plan_r2_rework4.md Errors)**:
  - G3 10k rerun v6: 11/11 gates PASS; 10,000 validated resets / 80,000
    observation-window rows; all counters 0; role 0.3055/0.3003/0.2940/
    0.1002 vs 0.30/0.30/0.30/0.10, max delta 0.6pp ≤ 2pp; provenance
    resolved_world_count=21==manifest_world_count=21, world_inputs_hash
    e1539c9a (resolved pool only) —
    `program/reports/routeA_rl_env_v1/g3_gate/G3_GATE_REPORT_v6.json`
    (sha256
    `5cb0c0035eb0045e051b09a68b3f22ee51627d7daa82885792bb215bd0c5897c`).
  - role metrics probe v4: 120 episodes, REAL 4-slot adapter
    (requested=resolved_n_envs=4): PASS — completed=120, attempted=120,
    voided=0, in_flight_at_collection_stop=0, transitions=3457==env_steps,
    gate_invalid_count=0, 7 invariants all True, role 38/34/37/11 —
    `ROLE_METRICS_EVIDENCE_v4.json` (sha256
    `9fe7daba302f7ef97f886cc86846a1c339c68b90fe24fe7ed610b7ae7ecc523e`).
  - no-ray stress v4: 10,000 frames, all adapter/rollout path, invalid=0 /
    far-dome=1,970 / report_failures=0 —
    `NO_RAY_STRESS_EVIDENCE_v4.json` (sha256
    `873480699018551ab9d70cf791f414accc1b9e8f540edd02653f1b4ec5dbd5bf`).
  - lifecycle evidence v1 (new dedicated artifact covering the three
    negative recovery paths A/B/C): 15 tests total PASS in 13.87s — 12 of
    them map to the A/B/C recovery paths (A 5 + B 4 + C 3), the other 3
    are the p7 eval-persistence regressions; A reset-invalid →
    reject_episode → next task reset succeeds; B AdmissionExhausted →
    attempt consumed + admission_blocked counted + sampler advances, zero
    partial mutation; C loop truncation → truncated=True + slot closed, no
    ACTIVE leftover across batches — `LIFECYCLE_EVIDENCE_v1.json` (sha256
    `bcaacffd226ef4f3d69cb00b7dcb4ee96a73c88a3fa1fecda43cbde6ac9873fc`).
  - CPU regression (clean run, no contention): **355 passed / 0 failed /
    7 deselected (GPU)**, 138.58s —
    `program/reports/routeA_rl_env_v1/g3_gate/regression/
    cpu_regression_rework4_20260809.log` (sha256
    `1be7823004c33db186c4443009eb94a5087fa61a75d29e28a9c9f5a0ada218b4`;
    the contended 1620.86s log was replaced by this clean run).
  - GPU/PPO core untouched; vendor 401 files zero-diff; E040 frozen 5
    files untouched. Fresh independent review `review_R2_rework4.md`
    (covers 9aec72a + 5419305 + rework #4 doc commits): **PASS —
    0 P0 / 0 P1 / 7 P2 observations** (sha256
    `d97fa8f7e1dfedb6cf94569bdd3d987bc7cacdbe6cafae89e3e7376f52916df1`).
    P2 dispositions: 2 fixed in the closure commit (manifest version_note
    arithmetic, RTM Evidence-column stale filenames), 2 doc updates
    (task_plan checkboxes, STATUS.md HEAD), 1 skipped with reason
    (gate_audit if/else+pass style — code change after review would
    invalidate the reviewed tree; style-only, deferred to R3 hygiene),
    1 process record retained (transient LIFECYCLE run, documented in
    task_plan Errors), 1 noted (task_plan Status line edited mid-review;
    honest content, committed in closure).

## Evidence ledger (R2 rework #5 additions)

- **Fifth-round ruling** (2026-08-10): R2 = REWORK REQUIRED (narrow
  closure) — paired-eval lifecycle was not fully closed: P1-1 `_eval_episode`
  loop-limit exit leaked an ACTIVE slot and recorded timeout=False; P1-2
  paired eval assumed single-slot semantics on a multi-slot vector env;
  P2-1 design v1.1 had not absorbed the rework #4 lifecycle contracts
  (fixed in design v1.2); P2-2 evidence phrasing (PC-VALID v5→v6,
  lifecycle 15/12 counting, task_plan HEAD sync). Fix commits `7666f71`
  (train_loop.py + 3 test files) + `eee7230` (lifecycle evidence runner
  v2) — evidence tree `9db41a21571eeb4e42ebc9453bc11d8d3d021e67`;
  docs commit `97f3277` (RTM corrections + task plans) does not touch
  tested code, so the evidence remains bound to 9db41a21.
- G3 10k rerun v7: 11/11 gates PASS; 10,000 validated resets / 80,000
  observation-window rows; all counters 0; role 0.3055/0.3003/0.2940/
  0.1002 vs 0.30/0.30/0.30/0.10, max delta 0.6pp ≤ 2pp; provenance
  resolved_world_count=21==manifest_world_count=21, world_inputs_hash
  e1539c9a — `program/reports/routeA_rl_env_v1/g3_gate/
  G3_GATE_REPORT_v7.json` (sha256
  `1c911e9c7550dfec6e86c6304de3085c0ac0d3a9c8db69060f927b8d141573b3`).
- role metrics probe v5: 120 episodes, REAL 4-slot adapter
  (requested=resolved_n_envs=4): PASS — completed=120, attempted=120,
  voided=0, in_flight_at_collection_stop=0, transitions==env_steps,
  gate_invalid_count=0, 7 invariants all True — `ROLE_METRICS_EVIDENCE_v5.json`
  (sha256 `d68659cbe263ea972777ccc1022633c1aa8aa15b5aef4fde89bdbad3b8ef4891`).
- no-ray stress v5: 10,000 frames, all adapter/rollout path, invalid=0 /
  far-dome=1,970 / report_failures=0 — `NO_RAY_STRESS_EVIDENCE_v5.json`
  (sha256 `8a9637c7dedb5585d34284e0915e0a9fc915b3189f6a12c80a67e80566867c1d`).
- lifecycle evidence v2 (rework #5): 17 tests total PASS in 4.72s — 14
  of them map to the recovery paths (A reset-invalid 5 +
  B admission-exhausted 4 + C loop-truncation 4, now incl. the paired-eval
  loop-limit twin, + D dedicated-single-slot-eval-env 1), the other 3 are
  the p7 eval-persistence regressions; paired-eval loop-limit exit closes
  the slot via mark_truncated and records timeout=True; paired eval with
  n_envs!=1 hard-fails (RuntimeError) — `LIFECYCLE_EVIDENCE_v2.json`
  (sha256 `877336d13c30faa8f7af02fa90ffc3632adb7c5bce7a7871e2bf5ef64778b701`).
- CPU regression (clean run): **357 passed / 0 failed / 7 deselected
  (GPU)**, 90.55s — `program/reports/routeA_rl_env_v1/g3_gate/regression/
  cpu_regression_rework5_20260810.log` (sha256
  `3aab1098fd31a020adfb6cdb3f212efd7e46da99ff15a87bb1e9b69f6a962ca7`).
- Fresh independent review `review_R2_rework5.md` (2026-08-10, zero-context,
  covers 0f00f85..HEAD = 7666f71 + eee7230 + 97f3277 + d4ee922): **PASS —
  0 P0 / 0 P1 / 4 P2 observations** (sha256
  `d63ae8daef61c2698c2f542e60bd2c48512c4b06021f04edfa3f182180b95bec`).
  P2 dispositions: 2 fixed in the closure commit (regression-time wording
  unified to the archived 90.55s log across STATUS/IMPLEMENTATION_LOG/
  handoff; the task_plan Status line updated mid-review is committed with
  the closure), 2 accepted with reason (RTM requirement rows keep their
  historical evidence chain — the ledger is the per-round incremental
  authority and ruling (d) is satisfied; e2e_gpu eval_env adaptation is a
  gpu-marked test, statically reviewed, to be covered at R3/G7 GPU
  integration). R2 = PASS (final, after rework #5) per the ruling's gate.

## Evidence ledger (R2 rework #6 additions)

- **Sixth-round ruling** (2026-08-10): R2 downgraded from "PASS (final,
  after rework #5)" to **R2 = REWORK REQUIRED (narrow entry-preflight
  closure)** — the two original fifth-round P1 fixes were confirmed correct
  (not redone), but a new stably reproducible P1 was found: the
  `run_e2e_batch` eval contract check sits inside the `one_update` branch,
  AFTER `ppo_update` + `CheckpointBuilder.save` (train_loop.py:1135):
  `no_update + eval_n_episodes>0` silently returns (no hard-fail, no
  paired eval, `eval_artifact=None`, `eval_json_path=None`); `one_update`
  with a missing `eval_env` raises only AFTER the training side effects
  (rollout + PPO update + checkpoint write). Mandated fixes: eval contract
  lifted to a `run_e2e_batch` ENTRY preflight (mode / eval_env non-None /
  n_envs==1 / dedicated-instance four checks, before any rollout/resume
  restore/PPO update/file write); dedicated eval_env instance enforced
  (`eval_env is env` -> RuntimeError); LoopConfig `eval_n_episodes >= 0`
  assertion (P2: negative count silently disabled eval); 4 new regression
  tests (the 2nd must prove env zero reset/step, policy params unchanged,
  optimizer not stepped, empty output dir, no checkpoint). Also corrected:
  `0f00f85..2bff8b1` = 5 commits / 9 files (not 6/8).
- Fix commits `2441188` (train_loop.py + test_p7_eval_persistence.py) +
  `fad10d4` (lifecycle evidence runner v3, path E) — evidence tree
  `0a067a385fe35b3a587ecda6816b6879dc08755a` (self-binding). Design v1.2
  errata §0.6.3 + §4.17 + header (out-of-repo, not committed here).
- G3 10k rerun v8: 11/11 gates PASS; 10,000 validated resets / 80,000
  observation-window rows; all counters 0; role 0.3055/0.3003/0.2940/
  0.1002 vs 0.30/0.30/0.30/0.10, max delta 0.6pp ≤ 2pp;
  role_proportion_asserted=True; provenance resolved_world_count=21==
  manifest_world_count=21, world_inputs_hash e1539c9a —
  `program/reports/routeA_rl_env_v1/g3_gate/G3_GATE_REPORT_v8.json`
  (sha256 `70ba1248136bf338b6e15f9bd79feaa62c33ffe326e4f785b26d4b7c7f31a374`).
- role metrics probe v6: 120 episodes, REAL 4-slot adapter
  (requested=resolved_n_envs=4): PASS — completed=120==attempted==
  n_episodes, voided=0, in_flight_at_collection_stop=0,
  transitions=3457==env_steps, gate_invalid_count=0, 7 invariants all
  True; role 38/34/37/11 — `ROLE_METRICS_EVIDENCE_v6.json`
  (sha256 `c1b2f149cff837ac0f5751b0ae5d9201001a5a76b921c80a51c1dfbd5af04f5f`).
- no-ray stress v6: 10,000 frames, all adapter/rollout path, invalid=0 /
  far-dome=1,970 / report_failures=0 / resets=466 —
  `NO_RAY_STRESS_EVIDENCE_v6.json`
  (sha256 `d51eac002b08f0b4f14edf65ce1f2135eb542e06275726c00ebe2c04d0200229`).
- lifecycle evidence v3 (rework #6): 21 tests total PASS in 4.40s — 18
  map to the recovery paths (A reset-invalid 5 + B admission-exhausted 4
  + C loop-truncation 4 + D dedicated-single-slot-eval-env 1 +
  E eval-entry-preflight 4: no_update+eval request hard-fails, missing
  eval_env fails before ANY side effect with env zero reset/step + policy
  unchanged + optimizer not stepped + empty output dir + no checkpoint,
  negative eval_n_episodes rejected, training env reuse rejected), the
  other 3 are the p7 eval-persistence regressions —
  `LIFECYCLE_EVIDENCE_v3.json`
  (sha256 `900b0b239ff25571a048af427a5a859d85b9e984f34f92e945413472dc54fca6`).
- CPU regression (clean run): **361 passed / 0 failed / 7 deselected
  (GPU)**, 96.05s — `program/reports/routeA_rl_env_v1/g3_gate/regression/
  cpu_regression_rework6_20260810.log` (sha256
  `8dd93721bb4202a8ac12968c03dd1b505ce099f7b0c9cd670ddc07b8d16532ef`).
- Fresh independent review `review_R2_rework6.md` (2026-08-10,
  zero-context, covers 2441188 + fad10d4 + 1177a0c): **PASS — 0 P0 /
  0 P1 / 4 P2 observations** (all numbers re-measured by the reviewer:
  5 commits / 9 files via rev-list + diff --stat; entry preflight at
  train_loop.py:791-818 before configure_numerical_mode L823 /
  restore_training_state L828 / ppo_update L1173 / CheckpointBuilder.save
  L1214; one_update-branch old check removed; run_paired_eval n_envs==1
  retained L517; LoopConfig assert L165; 4 new tests present with the
  mandated side-effect-free assertions; p7 11 passed / lifecycle 10 passed
  re-run; 4 evidence tree_hashes = 0a067a38 = fad10d4^{tree}; all 5
  sha256 character-exact; regression log 361 passed / 7 deselected;
  sealed evidence untouched; vendor/E040 zero diff; worktree clean).
  P2 dispositions: 1 fixed (range notation 0f00f85..HEAD ->
  0f00f85..2bff8b1 in this ledger), 3 accepted with reason
  (HEAD^{tree} vs evidence tree: evidence tree == generator commit
  fad10d4^{tree}, docs-only commits after — the correct invariant,
  same convention as rework #5; lifecycle duration 4.40s vs 8.45s is
  machine variance, counts verified; reflog resets are 2026-08-08,
  before this rework window). R2 = PASS (final, after rework #6) per
  the ruling's gate.

## Decisions (R2 rework additions)

- **R2-REWORK-RULING-1** (2026-08-09, user audit ruling, supersedes the R2
  closure `a6f9cab`): **R2 = REWORK REQUIRED**; the 10 findings are binding;
  "correctly completed" parts retained; rollback = reviewed `git revert`
  only (no `git reset --hard`); vendor `third_party/dppo_official_cc7234a`
  + E040 frozen 5 files zero modification; R2 scope excludes policy/PPO
  loss/reward/env; role targets 0.30/0.30/0.30/0.10 ±2pp; w4/w8/w11/w12/w18
  never in train specs; all numbers measured, none fabricated; TDD with
  negative tests; fresh independent review required before final closure.
- **R2-REWORK-RULING-2** (2026-08-09, design-layer adjudication P1-5):
  design v1.1 §4.7 contract modified: `hit_count > 0` →
  `hit_count > 0 OR (hit_count == 0 AND analytic_ray_geometry_passes)`.
  Far-dome (no-hit, analytic-verified) is legitimate; single-point all-far
  remains ALWAYS invalid (unique floor + allclose-far double guard).
  Review PASS no P0/P1.
- **R2-REWORK-RULING-3** (2026-08-09, P2-8 adjudication): episode collection
  budget contract, first version = hard-fail on `n_episodes % n_envs != 0`
  (startup validation, exit 1 before any evidence is written). A partial
  final batch can never be reported as pass.
- **R2-REWORK-RULING-4** (2026-08-09, P2-9 adjudication): reset info
  episode_id unified to the EpisodeConfigRecord unique per-episode id,
  replacing the non-unique spec.admission_id.
- **R2-REWORK-RULING-7** (2026-08-09, user audit ruling, rework #3;
  renumbered from RULING-5 by rework #4 审核裁决 — the old RULING-5 number
  collided with rework #2's completeness-contract ruling below, and the two
  conflicted on tail-episode accounting; see that entry): 6
  findings binding (3 P1 + 3 P2, all verified [FACT] in code). Refill
  budget invariant: completed + active ≤ n_episodes at every step, exit
  with active=0 and attempted==completed+voided exactly. Slot lifecycle
  INITIAL/ACTIVE/DONE/VOIDED: reset_at accepts only done/voided slots,
  ACTIVE atomic hard-fail; new `abort_episode(env_ids)` API = explicit
  in-flight discard (no fault) for audit-type repeated resets. Source
  entry hash covers ALL raw manifest fields + resolved defaults
  (role/admission_id/generator_version/obstacle_path
  `<uid>.world.obstacles.json`), no whitelist, no `default=str`.
  Evidence tree-bound to `6b80587^{tree}`=`e303d160` (committed before
  running); rollback = reviewed `git revert` only (manifest v4 rollback
  field), `git reset --hard` forbidden.
- **R2-REWORK-RULING-5** (2026-08-09, rework #2 P1 adjudication): the
  probe's completeness contract is **completed == n_episodes** (full
  collection), NOT attempted == n_episodes. Multi-slot collection stops
  at the n_episodes bound (train_loop `while completed < n_episodes`);
  episodes still in-flight are dropped — they consumed attempts but are
  neither completed nor voided. attempted is a superset
  (attempted >= completed + voided) and the tail drop is recorded as
  accounting.in_flight_at_collection_stop, never masked. The old v1
  evidence's attempted=completed=120 was an artifact of its unintended
  single-slot run (no tail drop exists sequentially).
  (Number re-used by rework #3's refill-budget ruling until rework #4
   renumbered that one to RULING-7 — the two conflicted on tail-episode
   accounting: this entry legalizes the tail drop; RULING-7's
   attempted==completed+voided holds within a full batch where every
   admitted episode runs to completion.)
- **R2-REWORK-RULING-6** (2026-08-09, rework #2 P2 adjudication): sealed
  evidence must self-bind its run origin — tree_hash, command, exit_code,
  input-set fingerprints (manifest_sha256, world_inputs_hash) live IN the
  report; the manifest may only corroborate, never assert them externally.
  Frame accounting must be explicit: validated_reset_frames vs
  observation_window_rows (n_frames = resets × 8 window rows, NOT
  independently validated frames). Provenance failures hard-fail: no
  report is written without binding.
- **R2-REWORK-RULING-8** (2026-08-09, user audit ruling, rework #4): R2 =
  REWORK REQUIRED (not PASS) until a fresh independent review of the
  rework #4 tree; no R3 entry, no formal RL training before that review
  passes. Ten rework-#3 requirements keep their conclusions; reopened and
  re-verified this round: VECTOR-ORDER (train loop + adapter lifecycle
  closure), ADMISSION (AdmissionExhausted → sampler switches world/task,
  never a fixed-coordinate fallback; design v1.1:823), PC-VALID
  (reset-invalid recovery paths). Slot lifecycle gains explicit closures:
  `reject_episode(env_ids)` (ACTIVE→DONE, reset-reject) and
  `mark_truncated(env_ids)` (ACTIVE→DONE, loop-truncation) — no private
  `_done` mutation; `reset_at` is two-phase atomic (all admission/
  preparation read-only before any env mutation; a mid-batch
  AdmissionExhausted leaves ZERO partial resets); read-only
  `slot_state()`/`is_active()` replace blanket `except RuntimeError` in
  audit paths. Train loop: per-slot admission with AdmissionExhausted
  caught (attempt consumed, admission_blocked counted, slot stays
  resettable, sampler advances), explicit reject_episode after
  REJECT_EPISODE, `effective_truncated` (loop truncation recorded as
  truncated=True) + mark_truncated, EpisodeCollectionIncomplete carries
  admission_blocked_episodes. Eval path (PC-VALID): invalid reset frame
  closes the slot via reject_episode, invalid step frame via mark_voided —
  guard-surfaced ACTIVE leak fixed. Refill budget now decrements exactly
  (per-slot refill list; over-launch when n_envs > n_episodes fixed).
  Mock envs mirror the real adapter's semantics exactly (ACTIVE-reset
  guard included); tests: 10 lifecycle tests (6 裁决 scenarios) + 2 p7
  recovery pins; full CPU suite 355 passed / 7 deselected.
- **R2-REWORK-RULING-9** (2026-08-09, rework #4): loop-level max_ep_len
  counts COMPLETED transitions — `loop_trunc` is evaluated as
  `(s["step"] + 1) >= max_ep_len` pre-increment, preserving the original
  post-increment check (max_ep_len transitions per episode, not +1); the
  final transition is written with truncated=True and the slot closed via
  mark_truncated. A frame that both terminates AND faults leaves the slot
  already DONE — explicit closures are guarded by `is_active()`, never
  double-closed.

## Evidence ledger (R3 additions)

- 2026-08-10 **R3.0c 注释清理（task #8，commit c681a1b）**：
  rl/dppo_uav 20 源码文件 114 处 + tests 27 文件 ~72 处阶段性注释清除
  （仅注释变化，代码前缀零改动，标签删除/实质保留/§锚点保留）；全量
  py_compile + 全局 AST 扫描 0 残留（剩余 11 处命中全部为合法保留：
  5 字符串字面量 + 4 guide Step 3 设计锚点 + 1 测试函数名 + 1 assert
  消息）。R3.1 基线 GPU e2e 既存失败（非 R3.1 引入）同步修复：
  fresh-load normalizer 在 cuda:0 → unnormalize 返回 CUDA 张量 →
  train_loop 两处 numpy 消费确定性崩溃（L1049 `.detach().cpu().numpy()` +
  mock env is_tensor 镜像）。
- 2026-08-10 **R3.1（5a5748d）CRITIC-STATE-ONLY 关闭**：
  PolicyObsEncoder 不再把整个 policy 注册为 submodule（非注册强引用，
  weakref GC 过期改 object.__setattr__）；value_state_dict()/
  load_value_state_dict() 仅 mlp.*（strict + 非 mlp key 拒绝）；
  _build_checkpoint_data / restore_training_state 切 value-only API。
  key-set 测试 RED→GREEN。
- 2026-08-10 **R3.2（3ea3bba）RESUME-HASH-GATE + CKPT-AUDIT fresh 关闭**：
  checkpoint schema v2 + pre-load identity validation（8 组 hash 字段 +
  独立 source_file_hashes；mismatch 均在 state load 前 hard-fail，spy
  proof 0 次 load；schema v1 拒绝；空 policy_state hard-fail）。
  test_checkpoint 24 passed、CPU 382 passed、GPU g6 resume 1 passed
  （32:42）。code-reviewer BLOCK 已解（C1 g6_fix_verify numerical mode
  对称、W3 空 policy_state、L1 非 dict payload、W1/W2 记录）。
- 2026-08-10 **R3.3（c05d131 + 5e7c442）LEGACY-FORMAL-GUARD + 唯一
  canonical 入口关闭**：E040 最小 guard（pre-hash `7868d1be…` →
  guarded `c548efc2…`，hash 前后入账于 ENTRYPOINT_CLOSURE_REPORT.md
  §1）；entrypoint_guard.validate_official_environment()（git
  clean/commit/tree + vendor SHA256SUMS 401/401 + BC/manifest hash +
  projection=False + train/eval split + numerical mode）+
  official_source_file_hashes()；scripts/train_dppo_uav_official.py +
  scripts/eval_dppo_uav_official.py（7 代码/资产轴身份校验，config_hash
  轴以 checkpoint 自身保存值为准 [JUDGMENT]）。subprocess RED（E040
  exit 0）→ GREEN 8/8；code-reviewer Block 2H+3W+3L 全部落实（H1
  manifest hash 真实重算、H2 bank world 池可达性 CLI 级等）；提交后
  clean tree 上两个 clean-tree 测试（identity mismatch + positive
  preflight）由 skip 转 live；当时记录的 closure 计数 "47 passed / 0
  skipped" 在 R3.6 独立审查实测中不可复现（HEAD collect-only 28、
  5e7c442 时点 15 个测试函数），保留为历史记录、不作计数引用；CPU
  suite 406 passed / 2 skipped / 7 deselected。
- 2026-08-10 **R3.4（e30d835）EVAL-STD + EVAL-AUDIT fresh 关闭**：
  _eval_episode 双 floor 从 LoopConfig 透传（SpyPolicy RED：eval 静默
  走 policy default 0.01 → GREEN）；冻结 paired bank v2（PairedBankEntry
  14 字段 + (spec,seed) 确定性 episode_config_hash 冻结契约 +
  read_paired_bank_file 单一 schema 源）；episode trace（每步 13 键含 step+elapsed，
  validate 两侧非空全键）；唯一文件名
  `paired_eval_envstep_<9d>_ppoupd_<6d>_<ckpt12>.json` +
  tmp+os.replace 原子写（已存在 hash 不同 → RuntimeError 禁覆盖）。
  code-reviewer 7M/5L 全部落实（mock s9 布局镜像真实 frozen 几何、
  28 处阶段性注释清除、expected_contract_hash 真值化）。
- 2026-08-10 **R3.5（77e2fa2 + 7c9a9af）CAPABILITY-AUDIT 关闭**：
  capability_metrics.py（d_soft^H H=8/β=8.0/ω decay、onset、trap、
  side_flip、support cell + OfflineDensity frozen dataclass：
  counts/bin_size/tau_D=p20（p20=0→p50）/空或全零 counts → ValueError）
  + compute_capability_metrics（空记录/density None/三能力 role 任一
  缺失 → ValueError）+ evaluator validator 升级 + train_loop 删
  _CAPABILITY_PLACEHOLDER + offline density 缺失 hard-fail（eval 前置
  RuntimeError / CLI exit 2）。RED 先行 test_capability_metrics.py
  14+2 tests；code-reviewer 终审 2W+6L 全部落实（W1 onset NaN clearance
  跳过、W2 yaw onset 改 per-step |yaw_rate| 相对 pre-shell 基线、L1 eval
  脚本 artifact build 入 eval FAIL 处理器、L5 全 NaN 窗口无
  RuntimeWarning 等；8 新测试 RED 先证）；CPU suite 460 passed / 2
  skipped / 7 deselected。
- 2026-08-11 **R3.6 仓库闭合**：Step 1 全仓旁路搜索 9 axes（旧 DPPO
  loss/observation slicing/action slicing/formal mode/checkpoint writer/
  paired evaluator/RoleSampler caller/sigma_clip_min/projection flags）
  无 bypass；FM 脚本 = 独立 stack 不 import dppo_uav；Step 2 CPU suite
  **462 passed / 7 deselected / 174.08s**（`temp/r3_cpu_suite.log`；
  较 R3.5 460+2 skipped 差异 = 两个 clean-tree live 测试（identity
  mismatch + positive preflight）在脏/净树间切换：R3.5 在 7c9a9af
  提交前脏树跑（2 skip）、R3.6 Step 2 在 clean tree 跑（2 live））；
  Step 3 fresh-process checkpoint 测试 **24 passed / 14.60s**
  （`temp/r3_checkpoint_fresh_process.log`：六类 hash mismatch +
  source_file_hashes + spy proof 0 次 state load + matching passes +
  v1 拒绝 + 空 policy_state 拒绝）；Step 4 文档去陈旧 +
  implementation_manifest_R3.json + review_R3.md +
  `program/reports/routeA_rl_env_v1/official_core_r3/R3_AUDIT_REPORT.json`；
  Step 5 commit `d88ed73` `docs(dppo): close checkpoint entrypoint and
  evaluation audit`。
  移交注记：R2 rework #3 标注的 R3 前置风险（kinematic_uav_env_v2.py
  `hash(str(world_id))` 进程随机化 + restore_rng_states() 不恢复
  PYTHONHASHSEED）不在 R3 guide 范围、R3 未触碰冻结环境——状态保持
  OPEN，移交 R4 或独立环境工作包处理（不推翻 SEED-RESUME）。
- **R3 完成门（2026-08-11，同日 external-review REWORK REQUIRED 裁决
  后作废 — 本 PASS 不成立，见下方 Evidence ledger (R3 rework additions)
  与 Decisions (R3 rework additions)）**：CRITIC-STATE-ONLY /
  CKPT-AUDIT / RESUME-HASH-GATE / LEGACY-FORMAL-GUARD / EVAL-STD /
  EVAL-AUDIT / CAPABILITY-AUDIT 自评全 PASS；RTM 仅 EQUIV-REG（BLOCKED，
  ASSUMPTION-4 语义：状态非 gate 输入）/ GATE-ORDER（R4 关闭）及后续
  训练 gates（G5/G6/G7/G8）保留 OPEN。自评缺陷：v1 证据链以
  review_R3.md 自审为准，未覆盖 empty-LoRA、config_hash 8 轴、sampler
  校验时机、bank 绑定、NaN 掩码五类负向路径 —— 均为 external-review
  4 FAIL 行（CKPT-AUDIT / RESUME-HASH-GATE / EVAL-AUDIT /
  CAPABILITY-AUDIT）所针对，修复后各行状态见上表 REWORK 注记。
  **external-review #2（2026-08-11，HEAD `3a69725`）再作废**：
  rework #1 后仍不能定为终局 PASS——当前 HEAD 完整 CPU suite 实际失败
  （1 failed / 500 passed / 7 deselected；`test_eval_official_identity_mismatch`
  的 `_fake_checkpoint()` 仍为 schema v2 fixture 缺 v3 必填字段，构造即
  TypeError；归档 499/2 skipped 为含非 temp/ untracked 工作树运行，两个
  clean-tree 测试被 skip 掩盖），另 2 P0 + 3 P1 + bank 口径未闭合（详见
  Evidence ledger (R3 rework additions) external-review #2 条目与
  R3-REWORK-RULING-5）。**rework #2（2026-08-11，a1d8e96 + 36e30c1 +
  67fa369）全部关闭，R3 = PASS（终局，rework #2 后）** — 见下方
  Evidence ledger (R3 rework #2 additions) 与 R3-REWORK-RULING-6。

## Evidence ledger (R3 rework additions)

- 2026-08-11 **External-review REWORK REQUIRED 裁决（独立审查，非自审；
  作废 R3 完成门 PASS）**：7 requirement 行中 4 FAIL（CKPT-AUDIT /
  RESUME-HASH-GATE / EVAL-AUDIT / CAPABILITY-AUDIT）+ 3 PASS 属实
  （CRITIC-STATE-ONLY / LEGACY-FORMAL-GUARD / EVAL-STD）；2 P0 +
  3 P1 + checkpoint 缺口 C1/C2 + 事实声明 F1–F3。逐条三方验证
  （代码 + 设计 v1.2 §4.16/§4.17 + R3 guide）全部成立，裁决摘要与
  证据见 `temp/task_plan_r3_verdict_rework.md` Decisions 段。
  成立项汇总：
  - P0-1 LoRA 身份未闭合：eval 空 lora_state 跳过恢复 + 随机 LoRA(B=0)
    → 输出==BC；sha256(b"") 通过 validator 非空检查；config_hash 无
    rank/alpha；ckpt 不存 alpha。
  - P0-2 sampler 校验太晚：恢复顺序 policy→critic→opt→RNG→sampler，
    sampler 身份校验发生在所有 state load 之后；config_hash 缺
    base_seed/n_envs/lora/lr/optimizer 7 参数。
  - P1-1 eval-only world 被排除：eval 只建 rl_train pool，设计 v1.3
    边界 = "只评估不训练" 的 w4/w8/w11/w12/w18 不可达（manifest 无
    这些 world）。
  - P1-2 config_hash 自比恒等：eval 用 data.config_hash 与自身比较。
  - P1-3 bank provenance 装饰：无 bank_id；source_hash 只查非空；
    无 bank/density SHA-256 入 artifact。
  - P1-4 NaN 被掩码：capability 测试允许 NaN 跳过/排除，np.isfinite
    掩码生成有效指标。
  - C1 无独立 environment_rng_state：rng_states["env"]=PYTHONHASHSEED
    不可恢复。
  - C2 4 个 state 字段存而不验：lr_scheduler/sampling_std/reward_norm/
    numerical_mode（"字段存在证明内容真实"反模式）。
  - F1 42d652d..HEAD = 12 commits（非 13）；F2 工作树非 clean；
    F3 145/96.37s + 462/195.08s 可复现。
- 2026-08-11 **Rework 修复（TDD RED→GREEN，负向测试先行）**：
  RED 25 functions / 39 collected tests 落
  `tests/dppo_parity/test_r3_rework_negative.py`
  （8 负向路径：empty-LoRA train/eval hard-fail、LoRA alpha/rank
  mismatch hard-fail、sampler mismatch 时所有 state-load spy 均为 0、
  config hash 对 base_seed/n_envs/rank/alpha/lr 任一变化都改变、
  canonical eval 正向接受 rl_eval bank、eval-only world 仍被训练
  sampler 拒绝、bank/density 任一字节变化导致绑定变化、任一必需
  trace 字段 NaN/Inf 均拒绝 artifact）。GREEN 实现（修复文件：
  checkpoint.py / train_loop.py / evaluator.py / capability_metrics.py /
  environment_adapter.py / integration.py / scripts（train/eval
  official）/ 4 个既有测试文件升级）：
  - **schema v3**：CheckpointData 30 fields（+lora_identity
    +environment_rng_state +exact_resolved_config 增 config_layer 精确
    hashed dict）；v2 checkpoints 显式 audit-only（guide 契约）。
  - **training_identity 8 轴 config layer**：base_seed/n_envs/lora_rank/
    lora_alpha/lr_actor/lr_value/actor_optimizer/value_optimizer；
    `_config_dict` 被 save 侧 `_build_checkpoint_data` 与 restore 侧
    `resume_identity_for` 共享；base_seed 对无该属性 sampler（scripted/
    test double）记录 None（getattr + None-safe int coercion）。
  - **restore 顺序闭合**：validate_resume_identity →
    validate_lora_identity → policy-non-empty →
    validate_checkpoint_lora_usable → _validate_sampler_identity →
    _validate_structured_state → validate_environment_rng_state →
    THEN loads → task_sampler.load_state_dict → device anchor；
    sampler 校验在任何 state load 前（mismatch 时所有 state-load spy
    均为 0）。
  - **structured-state 一致性验证**：lr_scheduler/sampling_std/
    reward_norm/numerical_mode 逐项验证（C2）。
  - **environment_rng_state 三键**（C1）。
  - **bank v3**：bank_id = canonical_sha256(内容, sort_keys=True)，
    `_bank_content` 顶层只排除 bank_id（bank_version 在内容内）；
    paired_bank_from_dict 校验顺序 = schema gates FIRST（精确消息）、
    bank_id binding LAST（schema-valid 篡改的 catch-all）。
  - **empty-LoRA hard-fail**：_live_lora_state() 空 lora_params → {}，
    EMPTY_LORA_CHECKSUM=sha256(b"")，evaluator/validator 拒绝。
  - **eval pool**：build_routea_eval_pool = rl_train ∪ rl_eval；
    manifest eval entries 准入；CLI 精确消息。
  - **NaN/Inf hard-fail**：capability_metrics + evaluator validator
    拒绝任一必需 trace 字段 NaN/Inf（不再掩码）。
- 2026-08-11 **Rework 验证（三档，均在修复后重跑）**：
  - 定向：test_checkpoint.py + test_r3_rework_negative.py +
    test_paired_bank.py + test_p5_contracts.py + test_train_loop.py +
    test_p7_eval_persistence.py + test_canonical_entrypoints.py →
    **128 passed**。
  - 全量 CPU suite：`python -m pytest -q tests/dppo_parity/ -m "not gpu"
    -p no:cacheprovider` → **499 passed / 2 skipped / 7 deselected in
    167.08s**（2 skipped = clean-tree 测试在脏树跳过，R3.6 同款语义；
    commit 后 clean tree 将 live）—
    `program/reports/routeA_rl_env_v1/official_core_r3/regression/
    cpu_regression_r3_rework_20260811.log`。
  - fresh-process（`-p no:cacheprovider` 独立进程）：
    test_checkpoint.py + test_r3_rework_negative.py → **63 passed**。
  - AST + import 检查 21 个文件全过；Py3.8 AST clean。
- 2026-08-11 **Rework 期间修复的 11 个全量-suite 回归（定向 128 全绿
  掩盖，full-suite 暴露）**：① `task_sampler.base_seed` 无条件访问 →
  ScriptedTaskSampler 无该属性 → AttributeError（train_loop L1278/
  L1666）→ getattr(..., None)；② `int(None)` 崩溃 → None-safe；
  ③ mock `lora_info={}` 被新 empty-LoRA gate 拒绝 → 改用 plain-dict
  axes 形式（`_live_lora_axes` 文档化的测试替身形式；insert_lora 的
  TARGET_PATTERNS 只匹配 UNet 层名，FakeModel 上插 0 adaptors）；
  ④ p5 测试 build_eval_artifact 缺 bank_id/offline_density → 补
  provenance；⑤ canonical entrypoint 测试 `_real_bank_payload` 仍是
  v2 → 升 v3 + 重签；⑥ 重签初始错误排除 bank_version（`_bank_content`
  顶层只排除 bank_id，bank_version 在内容内 —— 通过打印 mine vs
  theirs hashes + keys diff 诊断）→ 过滤改 `k != "bank_id"`；
  ⑦ 期望消息随 P1-1 eval-pool 新语义更新。
- 2026-08-11 **Rework 证据文件**：`implementation_manifest_R3_v2.json`
  （v1 保留历史）、`R3_AUDIT_REPORT_v2.json`（
  `program/reports/routeA_rl_env_v1/official_core_r3/`）、
  `review_R3_rework.md`（零上下文 fresh 独立审查，取代 review_R3.md
  的 PASS 效力）；`review_R3.md` v1 保留为历史记录、标注 VOID 于
  fresh 审查通过时。
- 2026-08-11 **External-review #2 裁决（独立审查，非自审；作废
  rework #1 的 PASS 结论，R3 恢复 REWORK REQUIRED / NO-GO，rework #2
  pending）**：裁决 11 项 claim 经本地逐条验证（代码 + 运行时探针
  `temp/r3_rework2_probe.py`）全部成立：
  - **完整 suite 失败（P1-3 根因）**：HEAD `3a69725` 全量 CPU suite
    `1 failed / 500 passed / 7 deselected in 215.28s`；失败项
    `test_eval_official_identity_mismatch` 的 `_fake_checkpoint()` 仍按
    schema v2 构造（缺 lora_identity/environment_rng_state 两个 v3 必填
    字段），`CheckpointData(...)` 构造即 TypeError，测试从未走到
    subprocess identity 拒绝路径（探针实测 `TypeError: __init__() missing
    2 required positional arguments`）；归档 499/2 skipped 在含非 temp/
    untracked 的工作树下运行（`_preflight_clean` 只允许 `?? temp/`），
    两个 clean-tree 测试被 skip 掩盖，数字不代表当前 HEAD。
  - **P0-1 LoRA checksum 未绑定实际张量**：`validate_lora_identity` 只查
    四键存在；`validate_checkpoint_lora_usable` 只比 rank/alpha/adaptor
    数，不重算 `lora_checksum_of(lora_state)`（探针
    `LORA_BAD_CHECKSUM_ACCEPTED`：saved=deadbeef vs actual=ffebaaf7
    通过）；identity 宣称 wired（rank=4/n_adaptors=2）但 `lora_state={}`
    被 train resume 接受（`TRAIN_EMPTY_LORA_ACCEPTED`；现有
    test_restore_hard_fails_wired_lora_with_empty_state 实为 rank
    mismatch 用例，未覆盖「宣称 wired + 空 state」）；eval 侧直接重算实际
    checksum 可拒绝，train resume 侧未闭合。
  - **P0-2 environment_rng_state 值级验证缺失**：
    `validate_environment_rng_state` 只查三键存在（探针
    `BAD_ENV_RNG_VALUES_ACCEPTED`：seed_source=garbage /
    python_hash_seed=different / restorable=True 通过）；生产多 world
    路径（world_uid 字符串）`abs(hash(str(world_id))) % (2**31)` 进入
    `_episode_rng_seed` → 点云合成 RNG（kinematic_uav_env_v2.py
    L238/L489），跨进程 PYTHONHASHSEED 不同则点云不同（单 world int
    world_id 走稳定 blake2b 分支，不触发 hash fallback）；checkpoint
    只记录环境变量值 + restorable=False，不记录实际 episode RNG 种子。
  - **P1-1 reward_norm 只比 version**：`_validate_structured_state`
    checks 中 reward_norm 仅比较 `reward_norm_state.version`（探针
    `BAD_REWARD_NORM_SEMANTICS_ACCEPTED`：source=evil/method=ema/
    params={mean:99.0} 通过）。
  - **P1-2 episode outcome 无类型/finite 检查**：bc/rl 聚合记录只查五键
    存在（evaluator.py L386-392），探针 `NAN_EPISODE_RETURN_ACCEPTED`
    （return=NaN 通过；trace 层 NaN 双层 hard-fail 已正确，outcome 层
    未闭合）。
  - **bank/density 口径**：bank_id/offline_density_sha256 为规范化内容
    哈希（canonical_sha256 sort_keys + compact；探针
    RAW_BYTES_DIFFER=True / BANK_ID_SAME=True，仅缩进/键序变化不改变
    bank_id；语义变化 seed+1 仍敏感）；rework #1 报告/代码注释
    “任一字节变化 → 绑定变化”表述不准确，应改称 canonical content
    hash 或另存 raw file SHA-256。
  - **正确部分确认**：critic value-head 隔离 / LoRA rank-alpha-adaptor
    pre-load gate / sampler 身份任何 load 前校验 / config hash 8 轴 /
    eval 池 rl_train∪rl_eval / 训练 sampler 拒 rl_eval / trace NaN 双层
    hard-fail / std 显式透传 / legacy guard / 原子写唯一命名——本次无
    新问题；canonical rl_eval 完整正向 CLI e2e 仍未用真实 schema v3
    checkpoint 跑通（仅池层测试）。

## Decisions (R3 rework additions)

- **R3-REWORK-RULING-1**（2026-08-11，外部审查裁决，binding）：R3 =
  REWORK REQUIRED（非 PASS）；4 FAIL 行 + 2 P0 + 3 P1 + C1/C2 全部
  成立并已修复；v1 PASS 证据链作废；rollback = reviewed `git revert`
  仅；修复必须 TDD 负向先行；修复后必须重建 manifest/RTM/audit
  report/fresh review 四件套；审查八项负向要求 verbatim 为验收标准
  （empty LoRA train/eval hard-fail；LoRA alpha/rank mismatch
  hard-fail；sampler mismatch 时所有 state-load spy 均为 0；config
  hash 对 base_seed/n_envs/rank/alpha/lr 任一变化都改变；canonical
  eval 正向接受 rl_eval bank；eval-only world 仍被训练 sampler 拒绝；
  bank/density 任一字节变化导致绑定变化；任一必需 trace 字段 NaN/Inf
  均拒绝 artifact）。
- **R3-REWORK-RULING-2**（2026-08-11，实现裁决）：schema v3 契约 —
  v2 checkpoints 显式 audit-only（不拒绝读取，拒绝作为恢复证据）；
  config_layer 是精确 hashed dict（非 config 对象引用）；8 轴
  training_identity 是恢复前 hard-fail 的最小集；sampler 身份校验
  必须在任何 state load 之前（"先校验后加载"顺序不可回退）。
- **R3-REWORK-RULING-3**（2026-08-11，bank 契约）：bank_id 绑定
  完整内容（顶层只排除 bank_id；bank_version 留在内容内参与哈希）；
  校验顺序 schema-first/binding-last —— 绑定是 schema-valid 篡改的
  catch-all，不是 schema 检查的前置替代。
- **R3-REWORK-RULING-4**（2026-08-11，数值契约）：NaN/Inf 必须
  hard-fail（capability_metrics + evaluator validator），禁止
  np.isfinite 掩码把非有限值变成有效指标；能力指标只在真实有限
  值上计算。

- **R3-REWORK-RULING-5**（2026-08-11，external-review #2 裁决，binding）：
  R3 恢复 **REWORK REQUIRED / NO-GO**，rework #2 pending，**暂不进入 R4**。
  rework #1（RULING-1..4）修复方向正确、多数已接入真实调用链，但：① 当前
  HEAD 完整 CPU suite 实际失败（clean-tree 语义下两个测试真正执行，陈旧
  schema v2 fixture 暴露）；② LoRA checksum 不绑定实际张量、wired identity
  + empty state 被 train resume 接受（P0-1）；③ environment_rng_state 值级
  验证缺失、跨进程点云 RNG 不可复现风险未闭合（P0-2）；④ reward_norm 只比
  version（P1-1）；⑤ episode outcome NaN 接受（P1-2）；⑥ fixture 未迁移
  schema v3（P1-3）；⑦ bank 口径 = canonical content hash 非原始字节。
  返工清单（验收标准）：LoRA checksum 强一致 + wired-identity-empty-state
  hard-fail + key 集/参数数/shape 校验；env RNG 值级验证 + PYTHONHASHSEED
  显式设置/记录/逐字符比较；reward_norm 完整 canonical dict 对比；episode
  outcome 类型/finite/范围验证；schema v3 fixture（clean-tree 两测试必须
  实际执行，不得以 dirty-tree skip 作终局证据）；bank 口径修正或 raw file
  SHA-256。证据重建：review_R3_rework.md 再 VOID、阶段总结改 rework
  pending、clean-tree 全量 suite 重跑、真实 schema v3 GPU resume + canonical
  paired eval、fresh reviewer 增加“字段齐全但内容错误”negative matrix。

## Evidence ledger (R3 rework #2 additions)

- 2026-08-11 **Rework #2 修复（TDD RED→GREEN，commit `a1d8e96` + `36e30c1` +
  `67fa369`）**：RULING-5 六项返工清单全部实现——
  ① **LoRA checksum 强一致 + wired-empty-state + key/shape**（checkpoint.py
  `validate_checkpoint_lora_usable`）：重算 `lora_checksum_of(lora_state)`
  与 declared 逐字符比较（declared is None 的 skip 分支在 restore 路径不可达
  ——`validate_lora_identity` 先硬失败缺 checksum 轴）；wired identity
  (rank>0) + 空 lora_state hard-fail；key 必须恰好 `["lora_0",...,
  "lora_{n-1}"]`、count == n_adaptors、逐张量 shape vs live lora_params
  （review P2-1：live wired 无 lora_params 亦 hard-fail，不再静默跳过）。
  ② **env RNG 值级验证**（`validate_environment_rng_state`）：seed_source ∈
  ('sampler_state',) 枚举、restorable 真 bool（`isinstance(..., bool)` 拒
  int 0/1 与 str）、python_hash_seed 与 `os.environ.get("PYTHONHASHSEED","")`
  逐字符比较（本环境未设置 → 记录 "" 必须匹配）。③ **reward_norm 完整
  canonical dict**（train_loop.py）：`StructuredStateIdentity.reward_norm_state:
  Dict[str, Any]` 取代 reward_norm_version，live 与 save 两侧都写
  `RewardNormState(version=...).to_dict()` 全量 {version, source, method,
  params}，`_validate_structured_state` 全 dict 比较。④ **episode outcome
  类型/finite/范围**（evaluator.py）：success/collision/timeout 严格 bool、
  return 非 bool 数值 + finite（review P2-3：int 恒有限，不做 float 转换）、
  steps 非 bool int ≥ 0。⑤ **schema v3 fixture**（test_canonical_entrypoints.py
  `_fake_checkpoint`）：全 30 字段 v3 + config_layer 自洽（config_hash 与
  exact_resolved_config 一致），identity mismatch 在真实 hash 上表现为
  resume identity mismatch。⑥ **bank 口径实证**：canonical content hash
  （sort_keys+compact），`test_bank_id_survives_formatting_changes` 证明纯
  格式变化不破坏绑定（RTM L17 已改口径表述）。
- 2026-08-11 **Rework #2 验证（三档）**：负向矩阵 `test_r3_rework2_negative.py`
  14 tests 全 TDD RED 先行（修复前 13 项 DID NOT RAISE）→ GREEN；
  探针 `temp/r3_rework2_probe.py` **5/5 REJECTED**（P0-1a checksum
  ffebaaf7 vs deadbeef、P0-1b wired rank=4 + empty state、P0-2 seed_source
  'garbage'、P1-1 reward_norm full dict、P1-2 return NaN）；clean-tree
  **全量 CPU suite 518 passed / 0 skipped / 7 deselected in 178.61s,
  EXIT=0**（两个 clean-tree 测试实际执行未跳过；`cpu_regression_r3_rework2_
  final_20260811.log` sha256 `7b10c06f…`）；回归安全子集 113 passed。
- 2026-08-11 **fresh 独立审查 `review_R3_rework2.md` PASS（0 P0 / 0 P1 /
  4 P2 观察）**：独立 code-reviewer 零上下文核验 6 项 checklist 全 [FACT]
  （负向矩阵 14/14、探针 5 输出、clean-tree 对 2 passed in 13.51s、归档
  log sha256 逐字符、adversarial 手动验证 np.bool_/key-naming 违规）。
  P2 处置（commit `67fa369`，find-and-fix）：P2-1 shape 检查不再静默跳过
  （live wired 无 lora_params → hard-fail，新测试）；P2-2 key-naming 直接
  单测（`test_lora_state_key_naming_mismatch_hard_fails`）；P2-3 超大 int
  return 不做 float 转换（新测试 `test_eval_artifact_accepts_huge_int_`
  `return`）；P2-4 `_wired_lora_data` fixture 命名契约注释。P2 修复后重跑：
  负向矩阵 17/17、探针 5/5、回归子集 111 passed / 2 skipped（脏树语义）、
  clean-tree 全量 518 passed / 0 skipped（终局 log 同上）。
- 2026-08-11 **R3 阶段总结（rework #2 后）**：`program/reports/routeA_rl_env_v1/
  official_core_r3/` 阶段总结已更新为 resolved（R3 = PASS 终局，rework #2
  后）；remaining evidence（RULING-5 原证据重建第 ⑤ 项）：真实 schema v3
  GPU resume + canonical paired eval 需真实资产（needs_real_assets 门控），
  R4/G7 集成覆盖。

## Decisions (R3 rework #2 additions)

- **R3-REWORK-RULING-6**（2026-08-11，fresh 独立审查 `review_R3_rework2.md`，
  binding）：RULING-5 六项返工清单全部关闭，R3 = **PASS（终局，rework #2
  后）**。验收证据：TDD RED 先行（负向矩阵 14 RED→GREEN）、探针 5/5
  REJECTED、clean-tree 全量 suite 518 passed / 0 skipped / 7 deselected
  （`cpu_regression_r3_rework2_final_20260811.log` sha256 `7b10c06f…`）、
  fresh 独立审查 PASS（0 P0 / 0 P1 / 4 P2 全处置，含“字段齐全但内容错误”
  negative matrix）。保留项（不阻塞 PASS，R4/G7 集成覆盖）：真实 schema v3
  GPU resume + canonical paired eval 需真实资产。**[SUPERSEDED 2026-08-11
  external-review #3 —— 见 R3-REWORK-RULING-7]**

## Decisions (R3 rework #3 additions)

- **R3-REWORK-RULING-7**（2026-08-11，external-review #3 裁决，binding）：
  **RULING-6 作废，R3 恢复 REWORK REQUIRED / NO-GO，rework #3 pending，
  暂不进入 R4**。裁决 claim 全部经代码 + 实跑验证成立：
  - **P0-1（结构级）**：`lora_state` 按参数张量保存（`extract_lora_state`
    policy_loader.py:161-178，每适配器 A+B 两张量 → n 适配器 = 2n keys），
    `n_adaptors` 是适配器模块数（lora_adapters.py:172）；`validate_checkpoint_
    lora_usable` 却要求 `len(state_keys) == n_adaptors`（checkpoint.py:419）
    → 任何真实 LoRA 正式 checkpoint 在 restore 必被拒。实跑复现：4 适配器 →
    8 keys vs n_adaptors=4 → `CheckpointError: lora_state key count 8 !=
    declared n_adaptors 4`。现有负向矩阵全绿只因 fixture 手工构造"1 参数/
    适配器"语义（test_r3_rework2_negative.py:109-144 等），与生产实现不一致；
    全仓无一条真实 insert_lora 正路径 save→restore 测试。
  - **P0-2（值级）**：两 env 三处仍调用 `hash(str(world_id))`
    （kinematic_uav_env_v2.py:238/266、kinematic_uav_env.py:273）——string
    world_id（Route A RL v1 生产路径）进程级随机（实测三 fresh process 对
    同字符串输出三不同值）；空 PYTHONHASHSEED 时 checkpoint.py:502
    空==空比较恒等通过；builder 恒写 `restorable=False`（checkpoint.py:458）
    而校验只查 isinstance bool，formal restore 实际接受不可恢复状态。
  - **P1**：Guide Step 4/5 要求的计数器链测试（fresh→update1→resume
    update2→resume update3）与 GPU resume 测试未执行；`review_R3_rework2.md`
    自身写明完成真实 schema-v3 GPU resume + canonical paired eval 才可正式
    PASS，交接将其降级为"不阻塞"无设计依据。
  - **P2**：fresh review 覆盖 `HEAD=36e30c1`，实际最终生产代码为
    `67fa369/7e0fa22`；工作树 3 个 `temp/` untracked（preflight 豁免 ≠
    字面 clean）。
  **返工清单（验收标准）**：① schema v4 —— `lora_identity` 分别保存
  `n_adaptors`（模块数）与 `n_lora_param_tensors`（参数张量数），校验
  `len(lora_state) == saved n_lora_param_tensors == len(live lora_params)`，
  `n_adaptors` 只与真实适配器模块数比较，不硬编码 2×n_adaptors；v3 降
  audit-only；② 真实链正路径测试（真实 `insert_lora()` → save → fresh
  model → 完整 `restore_training_state()` → LoRA 张量 bitwise 恢复 +
  optimizer state 恢复 + base UNet 未改变）；保留 checksum/key/shape 负向
  测试但修正适配器计数假设；③ env RNG —— 两 env 删除全部
  `hash(str(world_id))`，稳定字符串 `world_id` 直接入 canonical JSON 再
  BLAKE2b 派生 seed；formal 入口要求固定且非空 `PYTHONHASHSEED`（未设置或
  `random` → preflight hard-fail）；`restorable=True` 仅在环境序列确实可恢复
  时记录，formal restore 拒绝 `restorable=False`；两 fresh process 对同一
  task 生成相同初始 observation/point cloud 的测试；④ 正式集成门禁 —— 真实
  schema-v4 GPU one-update → save → fresh-process resume → next update +
  canonical paired eval（冻结 bank / LoRA checksum / artifact hash / episode
  trace）+ 计数器/optimizer/sampler cursor/下一 update 数值一致；GPU/真实
  资产不可用时 R3 必须保持 **BLOCKED**，不能 PASS；⑤ 状态与证据重建 ——
  STATUS/TODO/RTM 恢复 NO-GO（本 RULING 即记录），旧 closure/总结/PASS
  review 标注 superseded/VOID，修复后重跑 CPU + GPU + fresh-process
  正向/负向门禁，最终 result commit 必须由新的独立审查覆盖。
  **closure（2026-08-13）**：五项清单全部关闭（见 Evidence ledger (R3
  rework #3 additions)）；fresh 独立审查 `review_R3_rework3.md` 覆盖
  最终 result commit `3054eee`（tree `641bd439`）= **PASS，0 P0 / 0 P1 /
  5 P2 观察全处置**；R3 = PASS（终局，rework #3 后），R4 解禁。

## Evidence ledger (R3 rework #3 additions)

- 2026-08-13 **R3 rework #3 RED 证据（GPU 门禁 run1）**：
  `gpu_gate_rework3_20260813.log`（sha256 `6638a77c…`）— 2 failed in
  3099.49s：gate 1 `KeyError: 0` @ test_r3_rework3_gpu_gate.py:412
  （per_denoise_* int 键 vs JSON str 键，harness 缺陷）+ gate 2
  `TypeError: can't convert cuda:0 device type tensor to numpy` @
  train_loop.py:782（`normalizer.unnormalize` 返回 policy 设备张量——
  canonical eval GPU 正路径首次执行暴露的生产缺陷，RULING-7 P1 命中的
  "正向路径从未在 GPU 执行"类别）。
- 2026-08-13 **修复**：`3a367d7`（test：_compare_metrics 两侧 _jsonable
  规范化 + Guide R3.2 Step 4 第二次 resume/update3 扩展，outer_iteration
  [1,2,3]）+ `3054eee`（fix：_eval_episode unnormalize 后 CUDA→host）。
- 2026-08-13 **GPU 门禁 run2（终局）**：`gpu_gate_rework3_20260813_run2.log`
  （sha256 `fdff7d9e…`）— **2 passed in 4405.12s (1:13:25)**：gate 1
  双 resume parity（25 metric groups，计数器/optimizer/sampler cursor
  与连续世界逐项一致）+ gate 2 canonical paired eval（6 paired
  episodes，bank `11ff61c8e8f7`，ckpt LoRA `bdb2b68ad2a0`）。
- 2026-08-13 **定向 CPU**：test_r3_rework3_negative(10) + positive(4) =
  **14/14 PASS**（fresh-process 测试实际执行，无 skip；fresh 审查独立
  复跑 14 passed in 8.75s 复核）。
- 2026-08-13 **clean-tree 全量 CPU suite（终局）**：
  `cpu_regression_r3_rework3_20260813_final.log`（sha256 `515329a2…`，
  session header 含 HEAD `3054eee` / tree `641bd439`）— **532 passed /
  9 deselected in 175.96s，PYTEST_EXIT=0**（541 = 532 + 9，9 个
  gpu-marked 测试经 --collect-only 复核）。
- 2026-08-13 **fresh 独立审查** `review_R3_rework3.md`：**PASS，0 P0 /
  0 P1 / 5 P2**（P2 处置：1 fixed = RED 日志归档；2 fixed = CPU 日志
  session header + 正确 exit 捕获；3 fixed = STATUS/TODO 终局更新；4/5
  accepted with reason）。~~该 PASS 终局被 external-review #4（2026-08-13）作废（P1 live 计数三方一致缺失），审查文件已加 SUPERSEDED 标注（正文保留）。~~

## Evidence ledger (R3 rework #4 additions)

- 2026-08-13 **external-review #4 裁决本地核验**：对抗探针复现 P1——
  saved=4/live=2 → `BAD_LIVE_PARAM_COUNT_ACCEPTED`；完整 restore 在
  `restore_lora_state` 抛 ValueError 且 STATE_LOAD_CALLS_BEFORE_FAILURE
  = `['policy']`；反向 saved=2/live=4 → `KeyError`。裁决成立，R3 降级
  REWORK REQUIRED / NO-GO，R4 frozen。
- 2026-08-13 **RED**：`4b947a0` `test_r3_rework4_negative.py`（direct
  validator + full-restore spy × 两方向）— **4 failed in 3.45s**，
  失败模式与裁决逐条对应。
- 2026-08-13 **修复**：`1f0e5bb` `checkpoint.py`（+22/−1）— live 参数
  张量数 vs declared `n_lora_param_tensors` 校验置于任何 shape 遍历前；
  探针重跑两方向 CheckpointError、spy `[]`。
- 2026-08-13 **定向电池**：`targeted_r3_rework4_20260813.log`（sha256
  `d64e6c45…`，HEAD `6bd3e91`）— **98 passed in 11.08s**（新 4 +
  rework #2/#3 全部 checkpoint 测试 + test_checkpoint.py）。
- 2026-08-13 **clean-tree 全量 CPU**：`cpu_regression_r3_rework4_20260813.log`
  （sha256 `6a57a1fe…`，HEAD `8cfee02`）— **536 passed / 9 deselected
  in 171.04s，PYTEST_EXIT=0**（545 = 536 + 9，--collect-only 复核）。
- 2026-08-13 **GPU 双 gate（checkpoint 核心修改后强制重跑）**：
  `gpu_gate_rework4_20260813.log`（sha256 `515f7bbe…`，HEAD `8cfee02`）
  — **2 passed in 4380.17s (1:13:00)**。
- 2026-08-13 **fresh 独立审查** `review_R3_rework4.md`：
  **PASS-WITH-CONDITIONS**（0 P0 / 1 P1 文档级 / 3 P2），7 项强制验证
  全 PASS；条件全处置（`6bd3e91`：env_v2 freeze-lift 记账 pre
  `d1168142…`→post `5b7b1651…` + 报告过时声明修正 + rework #3 计划
  清单矛盾解析 + banner 路径修正）。
- 2026-08-13 **external-review #5 证据修补（`dc3c239`）**：程序 PASS /
  证据 CONDITIONAL PASS → P1 补终局 `R3_AUDIT_REPORT_v3.json`（7/7
  verdict + 三档证据全 SHA256 + review4 条件处置 + 明示 R4 未完成）+
  RTM 7 行 evidence 列改引 v3 + 报告两处历史句加时间范围 + train_loop
  注释-only 修正（diff 证实零可执行语句变化）→ fresh
  `review_R3_rework4_evidence.md` **PASS**（0 P0 / 0 P1 / 3 P2 观察，4
  日志 SHA256 复算逐字符吻合）→ **R3 无保留关闭**。


## R5.8 ENGINE_CORRECT 新增行（2026-08-17，裁决 §4.2 NEW/OPEN → fresh PASS）

| 需求 | 语义 | 来源 | 实现文件 | oracle | 证据 | 状态 |
|------|------|------|---------|--------|------|------|
| VECTOR-GAE-TIMEAXIS | 交错 vector-env buffer 按每个 episode 自身时间轴递归 GAE，不退化为 one-step TD | 裁决 §4.2 NEW | `gae.py` + `rollout_buffer.py` + `test_next_value_chunking.py` | per-slot 手算 oracle | pytest log | **PASS (`3182874`)** — `compute_gae` 按 episode 分组倒序递归；交错 A0,B0,A1,B1,A2 与 per-slot oracle 逐元素一致（atol 1e-12）；n_envs=1 既有 hand-computed a0..a5 逐字节保持（leak 测试）；B-1 同包：next values 采集末从 rollout critic 填充（单一 value function，官方 values_trajs[t+1] 语义），未填充 hard-fail |
| OUTCOME-REWARD-IDENTITY | env 产出唯一 EpisodeOutcome；reward/终止/eval 三方同一；collision 胜出；early-death 独立桶 | 裁决 §4.2 NEW | `env/episode_outcome.py` + `kinematic_uav_env_v2.py` + `reward_v14/v16.py` + metrics/evaluator | C8 四反例（真实 env + 真实障碍几何） | pytest log | **PASS (`8a8e15c`)** — 固定 done_reason enum + 不变量；reward reached = env 权威三条件（z 容差+速度门），r_reach terminal-gated；collision+goal 同步 collision 胜出不付 bonus；early-death 独立桶+−5；reset 清 done_reason；四反例全过（高速掠过/低速到达/碰撞+goal/timeout 无陈旧泄漏）；eval reward_version 硬拒非 v14/v16 |
| FINITE-UPDATE-TRANSACTION | 梯度→参数→optimizer-state 全链 finite；异常即事务边界（不发布 ckpt/最新指针/计数） | 裁决 §4.2 NEW | `trainer.py` + `test_r58f_numeric_transaction.py` | 故障注入（Inf param / NaN 表 / 无梯度参数） | pytest log | **PASS (`619cbdf`)** — step 前全 trainable grad finite、step 后 param+exp_avg/exp_avg_sq finite；3/3 负向测试（Inf param post-step、NaN kernel 表、grad-less actor param）全部 FloatingPointError；kernel alpha 守卫 NaN-aware |
| VARIANCE-PROVENANCE | BC/RL variance 同物理空间；bc ckpt 内容 hash；seed 可重推导；GPU identical-policy sentinel | 裁决 §4.2 NEW | `trainer.py` + `measure_bc_variance.py` + `evaluator.py` + `test_r58e_paired_sentinel_gpu.py` | 真 GPU 同 policy 双端 sentinel | GPU log | **PASS (`92aa0f4`)** — trainer 经 normalizer 输出物理空间 variance（action_variance_space 记录）；bc_checkpoint_hash 改内容 SHA256；floors/max_ep_len 从 --rl_ckpt erc 同源；eval 双侧 disturb off（frozen bank 精确重放）；GPU sentinel per-episode delta 全 0 |
| CANONICAL-COMPOSITION | 生产入口（canonical train/eval CLI）端到端被真实验证，不绕过 caller | 裁决 §4.2 NEW | `test_r3_rework3_gpu_gate.py`（train/eval subprocess）+ 链 D | CLI subprocess + 门禁报告 | GPU log | **PASS（`cd04e93` 起连续干净树, 3 gate 77min）** — canonical train 入口（真 preflight/identity/release）→ canonical eval 入口（真 bank_id/outcome/扰动关闭）→ artifact 断言；resume-chain + paired-eval + sentinel 三 gate 全过 |

## R6 训练准入修复新增行（2026-08-18，裁决 Guide_A_OfficialCore_R6_Training_Admission_v1_0.md §12.1；R5.9 TRAINING_ADMITTED 被 R6.0 裁决 NO-GO 回退，见 STATUS.md）

| ID | Finding | 工作包 | 必改生产边界 | 必须存在的反例/测试 | 关闭证据 | G8 阻断 | 状态 |
|----|---------|--------|--------------|---------------------|----------|---------|------|
| R6-CTRL-1 | F-01 target-only 每批 release | A+B | `LoopConfig`、`compute_batch_target`、入口 batch driver、publication/retention/preflight | missing horizon；25/8 composition；100 updates release count；disk oracle | exact batch sequence、release=1、resume≤2、preflight 明细 | 是/P0 | **IMPLEMENTED_AWAITING_REVIEW**（A+B 包侧） |
| R6-CTRL-2 | F-02 episode 抢先截断 horizon | A | `run_e2e_batch` collection/refill/active-slot | 1-step short eps 收满 horizon；long ep 跨 update；n_env hard pin | actual horizon=target 每批；slot continuity hash | 是/P0 | **IMPLEMENTED_AWAITING_REVIEW** |
| R6-METRIC-1 | F-03 NaN gate | C | `metrics.py`、`trainer.py`、writer、`g7_gate_report.py` | zero/nonzero conditioned samples；nested NaN；missing denoise index | standard JSON；count/value/valid；counter invariants | 是/P1 | **IMPLEMENTED_AWAITING_REVIEW** |
| R6-EVAL-2 | F-04 capability 无信息 | D | bank schema、`capability_metrics.py`、统计器 | zero opportunity/exposure；missing replica；policy-dependent opportunity rejection | opportunity/exposure/event/validity；G8 NOT_INFORMATIVE 诚实输出 | G8 engine 否；G9 是 | **OPEN** |
| R6-EVID-1 | F-05 非 final-tree fresh | G | strict identity/evidence orchestration | source drift、commit drift、incomplete seed/matrix | 同一 final HEAD R6.4 双 seed fresh artifact hashes | 是/P1 | **OPEN** |
| R6-LOAD-1 | F-06 strict load | E | `policy_loader.py` 与 train/eval preflight | missing、extra unexpected、wrong BC hash、allowlist missing key | exact allowlist audit 在 rollout 前 PASS | 是/P1 | **OPEN** |
| R6-TUNE-1 | F-07 scheduler/scaler 缺失 | F | `training_schedules.py`、`reward_norm.py`、checkpoint/resume | vendor parity；none identity；state/config tamper；cold resume | 逐步 oracle、state hash、resume continuation | 是/P1 | **OPEN** |
| R6-OPT-1 | F-08 single group 假设 | E | optimizer creation/resume、`trainer.py` helper | second actor/critic param group | preflight reject；ordered parameter-name hash | 否/P2，须关闭处分 | **OPEN** |
| R6-DISK-1 | F-09 transaction 命名/边界 | B+E | `ppo_update` error propagation、artifact publication | post-step non-finite；crash before metrics | 无新 committed ckpt/metrics；报告改称 fail-before-publish | 否/P2，须关闭处分 | **IMPLEMENTED_AWAITING_REVIEW**（B 侧）；E 侧（ppo_update non-finite 传播）待 R6-E |
| R6-CTRL-3 | F-10 timeout 双计 | C+E | outcome/category mapping、evaluator aggregation | reached+truncated、collision+truncated、reached+collision、timeout | raw events + exclusive category 手算一致 | 否/P2，须关闭处分 | **IMPLEMENTED_AWAITING_REVIEW**（C 侧）；E 侧（evaluator aggregation 分类）待 R6-E |
| R6-EVAL-1 | BC 伪重复（18/24） | D | `eval_statistics.py` 与 report aggregator | duplicated 12-record BC 输入两次 | raw=24、unique/effective=12、SR=9/12 | 是/P1 evidence | **OPEN** |

### R6.0 证据冻结注记（2026-08-18）

- HEAD `52ce1ebfd32b65f7588f9f95dbcaa6277203a09f`，branch `a-official-core-r5`，tree：`?? temp/smoke_e5_rebuild.py`（E5 用户侧，保留）。
- 裁决冻结 fresh CPU/non-GPU：608 passed / 11 deselected（本行待本表当前 HEAD 实测复核后更新为 [FACT] 实测值）。
- 旧 R5.8/R5.9 相关行与本表无冲突，保留为 historical；TRAINING_ADMITTED 声明已被 R6.0 回退（见 STATUS.md 2026-08-18 段）。

### R6-A 包注记（2026-08-18，工作树 52ce1eb + 未提交 R6-A 变更）

- **契约**：`LoopConfig.collection_mode`（train_loop.py:203）在 `__post_init__` 解析/验证（:235-271）：派生（total set → "transition" 否则 "episode"）；episode 禁 budget 字段；transition 要求 total+horizon+target 三全，target-only 显式拒绝。`compute_batch_target` 拒绝 total-without-horizon（:295-300，"target-only state — forbidden (R6-A §11.2.2)"）；`is_budget_complete` 纯终局 oracle（:316-333，`TransitionBudgetOvershoot` :310-313）。
- **batch driver**：`checkpoint_kind` 形参（:1437）；入口 preflight（:1535-1550）——transition 必须带派生 target + `n_envs==1`（G8 hard pin）；attempt_limit transition=`max(2*n_episodes,4)+4+int(target)`（:1621-1627）；`_collection_complete()` 内联求值（:1735-1740）；refill transition=`n_envs-active_count`（:1749）；finality/effective_kind 后置终局判定（:2006-2027，`checkpoint_kind=="release"` 但非终局 → RuntimeError，零物化）；result `"checkpoint_kind"`（:2201）。
- **CLI**：train_dppo_uav_official.py:362-377 成对检查（exit 2，命名缺失 flag）；:379-387 n_envs==1 pin（exit 2）；主循环 transition 分支 `checkpoint_kind=None`（:539 起），episode 分支保留 batch_idx 终局。
- **哈希轴**：`_config_dict` 新增 `"collection_mode"`（train_loop.py:765）——模式是 resume identity 契约轴。
- **测试**：`tests/dppo_parity/test_canonical_budget_controller.py`（新增，5 个 RED 反例）：red1 CLI exit 2 + 缺失 flag 命名、red2 [8,8,8,1] + 恰一次 release + counter==25、red3 短 episode 收满 horizon、red4 显式模式跨 update 连续（obs 位级）、red5 demanded-release 非终局 hard-fail 零物化；`test_exact_transition_budget.py` 增补：target-only 拒绝 oracle、`is_budget_complete` oracle、collection_mode resolution、B-harness horizon 契约化。
- **证据**：RED `02_test_runs/r6a_red_52ce1eb.log`（10 FAILED 行，5/5 失败模式与预测一致）；GREEN `02_test_runs/r6a_green_52ce1eb_ws.log`（19 passed）；回归 `02_test_runs/r6a_regression_52ce1eb_ws.log`（141 passed, 2 skipped，skip=entrypoints 脏树设计内）。
- §11.2.5 验收：unit（[8,8,8,1]/overshoot/missing horizon）、composition、short eps、long eps、counters 五项全 PASS。
- R6-B 前置（§11.3）尚未动工：publication 事务（先 resume 后终局物化）、retention 不变量、磁盘预检估计式 —— 均归 R6-CTRL-1 的 B 包侧，本注记仅覆盖 A 包侧。

### R6-B 包注记（2026-08-18，工作树 52ce1eb + 未提交 R6-A/B 变更）

- **artifact_audit**（scripts/train_dppo_uav_official.py `artifact_audit`）：§11.3.2 三项 hard-fail —— metrics 指向缺失 checkpoint（**所有** record，不只 release）、同一 run 多个 final release、transition 模式无 release 或 release counter != run total；torn trailing line 记为 problems（crash residue 可报告可忽略），绝不物化为 release；返回 `{"release_paths", "problems"}`。
- **入口发布验证**（`_validate_batch_publish`，§11.3.1.2）：批次必须物化 checkpoint 文件；transition 模式 release ⟺ 累计计数 == total（提前 release = F-01 复发，loud）；kind 必须合法。主循环在 `persist_batch_metrics` **之前**调用（fail-before-publish 事务边界）。
- **磁盘预检重写**（`_disk_preflight`，§11.3.3）：旧 `per_ckpt×(max_updates+2)` 单文件模型废除；新显式估计 = `2×resume + 1×release + 1×tmp + metrics/eval 预留(1GiB) + safety(20GiB)`；来源优先级 = metrics 分类历史实测最大值（release 集来自 `_release_paths_from_metrics`）> 空目录 G7 实测 oracle（`DISK_ORACLE_RESUME_BYTES≈7.5MB` / `DISK_ORACLE_RELEASE_BYTES≈293MB`，§11.3.4）；明细 dict 输出（source/各分量/free/projected），来源未知取保守上界并标注。旧签名 `(output_dir, max_updates, reserve_bytes=20GiB)` 保持兼容（test_s4 回归通过）。
- **终局 audit 接线**：主循环结束后 `artifact_audit(output_dir, total_transitions=args.target_transitions)` —— transition 模式强制恰好一个 release + counter==total；episode 模式查一致性 + 单 release。
- **测试**：`tests/dppo_parity/test_r6b_artifact_audit_disk.py`（新增 12 项）：red1 缺失文件、red2 多 release、red3 counter 失配、red4 无 release、red5 空目录 oracle 明细、red6 历史实测最大值、red7/red8 入口验证拒绝、green1 合法终局 PASS、green2 孤儿 release 不保护（crash window）、green3 torn line 不物化 release、green4 合法入口验证。
- **证据**：RED `02_test_runs/r6b_red_52ce1eb_ws.log`（11 FAILED / 1 PASSED）；GREEN `02_test_runs/r6b_green_52ce1eb_ws.log`（12 passed）；回归 `02_test_runs/r6b_regression_52ce1eb_ws.log`（176 passed, 2 skipped）。
- **§11.3.4 验收映射**：batch1/100 纪律 → red2 + R6-A RED-1/2；actual<planned → red4 + R6-A RED-5 + `_validate_batch_publish`；crash window → green2（孤儿不保护）+ green3（torn 不物化）；空目录 oracle → red5；100-update dry-run 形态 → green1（100 文件 1 release audit PASS）+ retention 既有 S3。
- **R6-DISK-1 E 侧未动**：`ppo_update` post-step non-finite 的错误传播归 R6-E（F-09 分类侧）。

### R6-C 包注记（2026-08-18，工作树 52ce1eb + 未提交 R6-A/B/C 变更）

- **OptionalStatistic 两态契约**（metrics.py:44-67，§11.4.1）：`{sample_count, value, valid, invalid_reason}`；count>0 → 有限 value/valid=True/reason=None；count==0 → value=None/valid=False/frozen reason（`no_success_terminal`/`no_collision_terminal`/`constant_value_target`/`zero_executed_gradient_denominator`/`capability_opportunity_absent`）。非法 reason 与 non-finite mean 均 ValueError（工厂内拦截，第三态不可能产生）。statistics 工厂在 :53，cond-adv 在 :148（succ/coll → OptionalStatistic，role → 普通 dict），explained variance :186（constant target → null + 保留 `target_variance=0`，对官方 diffusion_ppo 的 NaN 语义做了 §11.4.1 化），executed/unexecuted ratio :219（unexecuted≤1e-8 → null）。
- **per-denoise 非可选**（metrics.py:238-268，§11.4.1）：每个 fine-tune index 必须有样本（`ft = contract.ft_denoising_steps`，contracts.py:30；trainer 侧累积 trainer.py:473-477 `if mask.any(): per_denoise_kl[k].append(...)`）；空集或 non-finite → **RuntimeError ENGINE_HARD_GATE**（训练停），绝不写 NaN。
- **trainer 接线**（trainer.py:568-635）：`compute_outcome_counts`（:270，10 个计数 dict + 动态 role map）→ `verify_outcome_invariants`（:326，每 ep ≤1 terminal、cut 必须 truncated 且非 terminated、role sums==totals，违规 RuntimeError 带 episode ids）→ `cross_check_terminal_rewards`（:403，§11.4.2 交叉校验：success terminal reward ≥ `0.5×reach_reward`、collision terminal reward ≤ 0，EpisodeOutcome 为真、reward 仅为交叉校验；`:399-400` 常量）→ 组装 `"per_denoise_approx_kl"` / `"executed_vs_unexecuted_grad_norm"`（executed/unexecuted/ratio）/ `"explained_variance"` / `"outcome_counts"`。
- **严格 JSON**（train_dppo_uav_official.py:113-176，§11.4.3）：`SCHEMA_VERSION="routeA_rl_env_v1_r6c"`（:115）；`_json_safe_metrics` 递归检查 non-finite（None 合法 = 显式 null 通道，float NaN 必拒，:118）；`persist_batch_metrics`（:136）`json.dumps(record, allow_nan=False, sort_keys=True)` + flush + fsync（:176）。
- **三层门禁**（§11.4.4）：ENGINE_HARD_GATE（训练内 RuntimeError）> UPDATE_HEALTH_GATE（g7_gate_report.py，R6.5 校准）> CAPABILITY_INFORMATION_GATE；**capability BLOCKED 独立通道**（g7_gate_report.py:569-585：`capability_blocked` 列表 + `capability_verdict="BLOCKED"|"PASS"`，:618-619 入报告；status 仅由 engine+health+structural 决定，BLOCKED 不抹掉 engine 结论）。
- **测试**：`tests/dppo_parity/test_r6c_standard_metrics_gates.py`（新增 24 项 RED-1..RED-24）：两态契约、cond-adv null/oracle、explained variance、ratio、per-denoise（ENGINE hard-fail）、outcome counts（动态 role）、invariants、交叉校验、persist 严格 JSON/schema_version、gate 检查（第三态拒绝、字段缺失、conditioned mismatch、缺 index、legacy 标记）、**RED-24 GPU 两态比较回归保护**（双侧 None 相等 / 单侧 None 双向 mismatch / 有限值 isclose / 顶层 child-None 崩溃守卫）。迁移：`test_metrics.py` 4 项（`.value`/sample_count/valid 断言）、`test_g7_metrics_persistence.py`（null+valid OptionalStatistic roundtrip、`"nan" not in raw`、inf-in-OptionalStatistic ValueError）、`test_checkpoint_retention.py`（torn fixture 带 schema_version）、`test_r3_rework3_gpu_gate.py`（`_compare_metrics` None-aware）、`test_g6_real_one_update_gpu.py`（ratio 两态迁移 + per-denoise 注释 + **gate-9 对比提取为模块级 `_compare_metrics`**）、`test_trainer.py`（batch_size = n×ft 鸽笼保证全 index 覆盖）、`test_train_loop.py`（METRIC_KEYS + `outcome_counts`）。
- **GPU 真缺陷 RED→GREEN（2026-08-18，run2 捕获）**：g6 gate-9 同 seed resume 对比 `float(None)` TypeError（`test_g6_real_one_update_gpu.py` 原 :412）——两态契约下 count==0 → value=None，旧循环只跳过 NaN 未跳过 None。修复：g6 内联循环提取为 `_compare_metrics`（两态正确：双侧 None 相等、单侧 None 进 mismatch 不崩溃），r3 的 `_compare_metrics` 补同一反向守卫（v2 有限 got None → mismatch 而非 float(None)）；RED-24 单元测试 import 两 helper 全语义覆盖（3.20s）。focused 99/99 未捕获原因：该测试 GPU-marked 被排除，仅真训练路径可暴露——验证 GPU 门禁必要性。
- **证据**：RED 见会话记录（test_target_kl_early_stop 空 index RuntimeError 捕获 → 鸽笼修复；METRIC_KEYS 缺 outcome_counts AssertionError 捕获 → 修复；capability 通道 typo 捕获 → 修复；**g6 gate-9 float(None) GPU RED round 2 → helper 提取 + 双向守卫 + RED-24；g6 gate-9 float(dict) GPU RED round 3 → 递归结构遍历 + RED-25**）；GREEN 全量 CPU focused 25/25 R6-C 文件（含 RED-24/25）+ 修复前 99/99（16.20s）；回归修复前 `02_test_runs/r6c_regression_52ce1eb_ws.log`（662/649/0/2/11，174.28s）+ `02_test_runs/r6c_regression_52ce1eb_fix_ws.log`（663 collected / 650 passed / 0 failed / 2 skipped / 11 deselected，194.67s，exit 0，两次一致）+ **递归修复后 `02_test_runs/r6c_regression_52ce1eb_fix2_ws.log`（664 collected / 651 passed / 0 failed / 2 skipped / 11 deselected，183.52s，exit 0——当前冻结参考）**；2 skipped = entrypoints 脏树设计内，同 R6-A/B；11 deselected = gpu marker 单列；GPU 受影响组：run1 60 min 超时被杀 + `-q` 块缓冲 → 0 证据（死因记录于 00_scope_and_head.json gpu_gate_runs）；run2 逐测试无缓冲重跑 → fresh_process_resume_chain EXIT 124（预算不足，>60 min）、paired_eval/budget_composition EXIT 1（**环境性：train CLI preflight 拒绝 dirty tree**，与 2 个 skipped entrypoints 同源，非代码回归）、**g6_one_update_mechanism EXIT 1（真缺陷：float(None)，见上）→ 修复 → run3 EXIT 1（2520.39s，第二真缺陷：同一循环对嵌套 dict 叶 float(dict)——executed_vs_unexecuted_grad_norm.ratio / outcome_counts per_role 映射；str/list 叶同类潜伏）→ 递归修复 → run4 PASS（2547.17s，`G6 resume parity OK: 32 metric groups + GAE hashes match`，`02_test_runs/r6c_gpu_run4_g6_fix_52ce1eb_ws.log`——递归比较器真训练验证通过，证实两次崩溃均为比较器结构假设问题、无指标真分歧，**GPU 真缺陷链闭环**）**；**处置**：真 GPU 全量验证按 §12.4 Gate D 在 R6-A~F final-commit clean tree 从零重跑（§8.1 DoD "current final HEAD 真 GPU composition/resume PASS"）。
- **§11.4.5 验收映射**：空 succ/coll 合法 null → RED-5/6/7 + 迁移 test_metrics（no_success/no_collision 两态）；2 个 success terminal count=2 手算 oracle → RED-8/9 + test_metrics（.value==2.0）；count>0 但 conditioned=0 → gate FAIL → g7 `validate_optional_statistic` + RED-20；`allow_nan=False` 嵌套拒收 → RED-13/14 + test_g7_metrics_persistence（nan/inf 字符串 + inf-in-OptionalStatistic）；任一 index count=0 ENGINE hard-fail → RED-10 + per_denoise_mean_dict RuntimeError；constant target EV null + target_variance=0 → RED-15 + test_metrics（constant_value_target）；旧 G7 legacy 标记 → RED-18/19 + g7 :238 legacy 行标记；formal metrics.jsonl 每行标准 JSON 可读 → RED-12/16/17 + persist roundtrip + schema_version。
- **R6-CTRL-3（F-10）C 侧**：outcome/category mapping 侧的计数（reached+truncated 等互斥分类由 `compute_outcome_counts` + `verify_outcome_invariants` 支撑，见上）；evaluator aggregation 分类侧归 R6-E，未动。

### R6-C REWORK 注记（2026-08-19，外部审查裁决 → 修复闭环）

- **裁决**：REWORK（fresh review）——run2/run3 已证属比较器缺陷、run4 真 GPU parity PASS 成立，但 4×P1 假阴性门禁路径 + 1×P1 证据冻结缺口 + 1×P2 可观测性问题不允无保留关闭。5×P1 + P2 全部代码核验确认（[FACT 代码]）后逐条修复。
- **P1-1 无条件双向 count 门禁**（g7_gate_report.py `check_outcome_invariants`）：旧实现只在 `events>0` 时比较 sample_count → phantom（events=0/sample_count=1）被接受；bool/负数/缺字段静默跳过。修复：无条件 `sample_count == events` 双向相等 + plain-int/非负/缺字段拒收。RED-26。
- **P1-2 completed-episode 跨层闭合**（同函数）：签名改收完整 record——`terminal_transition_count == record.completed_episodes`、`per_role_completed_episode_count` 汇总 == terminal、void/fault/rollout_cut 单列且绝不进入 completed 映射、字段类型/负数拒收。生产形状误报验证：按 `persist_batch_metrics` 原样构造 record（`compute_outcome_counts` + cond-adv 真实语义）→ 0 误报；篡改 `completed_episodes` → 检出 cross-layer 不变量。RED-27。
- **P1-3 schema 严格 allowlist**（`_load_metrics`）：旧 `if not schema_version` 把任意非空版本当 current 读。修复：`GATE_SCHEMA_VERSION="routeA_rl_env_v1_r6c"` 严格相等；未知/伪造/旧版非空版本 hard-fail（绝不按当前规则静默读取）；缺失/空串仍 legacy 标记。版本迁移必须显式实现。RED-28。
- **P1-4 GPU 比较器非有限 hard-fail**（`test_g6_real_one_update_gpu.py` / `test_r3_rework3_gpu_gate.py` `_compare_value`）：旧 parent 非有限 → 跳过（NaN 当无数据）。R6-C 两态契约下无数据 = OptionalStatistic count==0，NaN 任何位置 = engine violation。修复：任一侧非有限（含双侧同为 NaN/Inf）→ mismatch，dict/list 任意深度递归生效。RED-29 + RED-25 语义更新（双侧 NaN 也 mismatch）。
- **P1-5 证据冻结可重建**（MANIFEST.json + 00_scope_and_head.json）：完整 `git status --porcelain`（19 行）+ `git diff --binary` SHA256 + 逐文件 SHA256（15 modified + 4 untracked）+ untracked 排除原因（`temp/smoke_e5_rebuild.py` = 用户 E5 track、preflight 不计 temp/，此前漏记——untracked 3 更正为 4）+ 精确 producer 命令 + 退出码真实（EXIT marker）/推断（pytest summary，显式标注）+ HEAD identity 与 worktree identity 分离。同一 HEAD 不同未提交内容可区分。
- **P2 explained_var 日志契约**（trainer.py `ppo_update` 尾部日志）：`explained_variance()` 返回 dict（metrics.py:186）后 `%.4f` 在 logging handler 格式化抛 TypeError（被 handleError 吞 → 日志行丢失 + stderr 污染；RED-30 在旧代码上精确复现 `TypeError: must be real number, not dict`）。修复：显式记录 `value/valid/sample_count/invalid_reason/target_variance`。RED-30。
- **证据**：REWORK RED 7/7（`02_test_runs/r6c_rework_red_52ce1eb.log`，EXIT 0）；focused 51/51（r6c gates + g7 persistence + trainer + checkpoint retention，`02_test_runs/r6c_rework_focused_52ce1eb.log`，EXIT 0）；回归 **669 collected / 656 passed / 0 failed / 2 skipped / 11 deselected，190.57s，EXIT 0**（`02_test_runs/r6c_rework_regression_52ce1eb.log`——REWORK 后当前冻结参考，651→656 = RED-26..30 新增）；MANIFEST.json 重生成（20 artifacts 实时 SHA256 + worktree identity，round-trip 验证通过）。生产训练语义零改动（trainer 仅日志行；gate/比较器属证据层），GPU 结论由已核验哈希的 run4 artifact 承载；真 GPU 全量验证仍按 §12.4 Gate D 在 final-commit clean tree 承担。R6-METRIC-1 / R6-CTRL-3（C 侧）保持 IMPLEMENTED_AWAITING_REVIEW。
