# Guide 一致性验证报告：R2 程序 vs 两份 Guide（含程序实测）

- 日期: 2026-08-10
- 依据: 用户要求——"使用 superpowers 检查程序是否完全按照 Guide_A_OfficialCore_R2_Sampler_Env_Contracts_v1_0.md 和 Guide_A_OfficialCore_Completion_MasterPlan_v1_0.md 一致；不能只检查文字，还要检查程序"
- 角色: 审核者（只检查，不写代码）
- 检查对象: R2 程序（代码 + 测试 + 证据），HEAD=`dd94c22`（branch a-official-core-rework）
- 方法: 读代码（environment_adapter / train_loop / gate_audit / integration / routea_rl_env_adapter_v1 / pointcloud_gate / validated_synth_adapter / metrics / evaluator）+ **实测运行 Guide 指定测试** + 复核证据 JSON（G3 v8 / stress v6 / role v6 / RTM）

---

## 结论

**R2 程序与两份 Guide 完全一致：[PASS] 44 项 + [偏差] 1 项（I7，非阻塞）+ [FAIL] 0 项。**

程序实测证据（本次运行，非引用归档）：

| 实测 | 结果 |
|------|------|
| 12 个 Guide 指定测试文件（R2.1–R2.6） | **141 passed / 1 deselected（GPU）/ 33.59s / exit 0** |
| 全量 CPU 回归（tests/dppo_parity） | **361 passed / 7 deselected / 95.26s / exit 0** |
| G3 10k v8（gates/deltas/自绑定） | 11/11 PASS；deltas 0.0055/0.0003/−0.006/0.0002 ≤2pp；tree `0a067a38` |
| no-ray stress v6 | 10,000 帧 invalid=0 / far-dome=1,970 / report_failures=0 |
| git diff `0f00f85..2bff8b1` | 5 commits / 9 files，零 vendor/reward/policy 改动 |

---

## A. R2 Guide Global Constraints（8 项）— 8/8 PASS

| ID | 判定 | 证据（代码位置 / 实测） |
|----|------|--------------------------|
| A1 | PASS | R1 = PASS（终局）STATUS 顶部 + TODO L14 [x] |
| A2 | PASS | 数据侧：生产 manifest 53 条目 world_uid 全 `routeA_rl_v1__*` 零 w 系列；代码侧：`HISTORICAL_EVAL_WORLD_IDS={"w4","w8","w11","w12","w18"}`（routea_rl_env_adapter_v1.py:45）+ `check_historical_contamination`（L54-97，rl_train/rl_val 时 world_uid/world_file/metadata_file 命中即返回错误串）4 调用点（L317/663/672/683） |
| A3 | PASS | G3 v8：targets 0.3/0.3/0.3/0.1，proportions 0.3055/0.3003/0.2940/0.1002，deltas 最大 0.6pp ≤ 2pp，role_proportion_asserted=True，counts 3055/3003/2940/1002 |
| A4 | PASS | G3 v8 gates 11/11 PASS（admission_blocked/clearance/dense_path_stretch_fail/eval_leakage/hash_missing/invalid_pc/manifest_hash_mismatch/reachability/role_proportion/role_unknown/unvalidated_fallback） |
| A5 | PASS | `manifest_record_key(world_uid, role, admission_id)` 复合键（environment_adapter.py:287-297）；G3 10k 内同 world 多 role 正常并存（3055/3003 等均非互斥） |
| A6 | PASS | G3 生产脚本 `scripts/g3_10k_reset_gate_audit.py` 调 `run_reset_gate_audit`（gate_audit.py 注入同 factory `EpisodeTaskSampler`，L135-136）；provenance：`resolved_world_count=21==manifest_world_count`、tree_hash 自绑定 |
| A7 | PASS | stress v6：10,000 帧 invalid=0 / far_dome=1,970 / report_failures=0 / 路径 adapter reset/step→info→PointCloudValidityGate |
| A8 | PASS | `git diff --name-only 0f00f85..2bff8b1`：9 文件（2 文档 + train_loop.py + 1 证据脚本 + 2 task_plan + 3 测试），无 vendor/reward/policy 模块；`git log --all -- vendor/` 空 = 全分支 vendor 零改动 |

## B. R2.1 EpisodeTaskSampler（5 项）— 5/5 PASS

| ID | 判定 | 证据 |
|----|------|------|
| B1 | PASS | SampledEpisodeTask(frozen) + next_attempt/mark_completed/state_dict/load_state_dict（environment_adapter.py L326-443） |
| B2 | PASS | `episode_seed = self._base_seed + attempt`（L378-393）；`test_episode_seed_determined_by_base_seed_and_attempt` 实测通过 |
| B3 | PASS | 6 类 RED 测试全在位（13 个测试）：比例 `test_10k_attempts_within_2pp` / 同 world 多 role `test_same_world_multiple_roles_all_sampled` / save-load `test_save_load_resumes_identical_sequence` / 不重复 `test_two_batches_never_repeat_ids_or_seeds` / 空 bucket `test_empty_role_bucket_hard_fails` / eval-only `test_eval_only_spec_hard_fails` |
| B4 | PASS | state_dict 7 项（schema/base_seed/RNG base64+cursors/attempt/episode/completed/spec-set hash，L400-417）+ 硬校验 load（L419-443） |
| B5 | PASS | gate_audit.py 注入同 factory（L135-136 不独立重建）；AST 测试 `test_gate_audit_constructs_shared_sampler` + `test_gate_audit_has_no_independent_role_rng` 实测通过 |

## C. R2.2 train loop 接入（5 项）— 5/5 PASS

| ID | 判定 | 证据 |
|----|------|------|
| C1 | PASS | `role_stats_report` 7 字段：attempted/completed/voided/transition × counts/proportions（metrics.py:46-63） |
| C2 | PASS | `task_sampler` 为 `run_e2e_batch` 必需参数（train_loop.py:746）；采样仅经 `task_sampler.next_attempt()`（L957）；无内部 attempt-index seed / `task_specs[(attempt-1)%len]` / `seeds=[attempt]` 模式（AST 测试 `test_train_loop_injects_sampler_without_rebuilding` 实测通过） |
| C3 | PASS | episode id = `f"{spec.world_uid}:{spec.role}:{sm.episode_seed:09d}"`（train_loop.py:971-974） |
| C4 | PASS | `EpisodeCollectionIncomplete(RuntimeError)`（L102-136）6 计数域 + attempt_limit；attempt 耗尽即 raise（L1136），绝不把部分 batch 当成功 |
| C5 | PASS | `test_two_consecutive_loops_continue_sampler_sequence`（test_train_loop.py:382）：seeds [5,6]→[7,8] 严格递增、episode ids 跨 batch 不重复、`min(idx2)==max(idx1)+1`、`:09d` 格式断言——实测通过 |

## D. R2.3 双 hash（5 项）— 5/5 PASS

| ID | 判定 | 证据 |
|----|------|------|
| D1 | PASS | `EpisodeTaskSpec.manifest_entry_hash` 语义为 source-entry hash，不声称覆盖 start/goal（环境适配器 L85-97 docstring；start/goal/yaw 由 episode_config_hash 覆盖） |
| D2 | PASS | `EpisodeConfigRecord` 14 字段（L107-133，含 manifest_entry_hash + episode_config_hash） |
| D3 | PASS | `canonical_sha256`（L270-284）：sort_keys/separators/ensure_ascii=False/allow_nan=False，无 default=str（非 JSON 字段硬失败）；RTM META-SOURCE-HASH 记录逐字段 sensitivity 测试 |
| D4 | PASS | `manifest_record_key(world_uid, role, admission_id)` 复合键（L287-297），同 world 多 role 不互相覆盖 |
| D5 | PASS | 全链路双 hash：reset info `_wrap_info`（L1122-1134，注释明示 R2.3 buffer trace 来源）+ buffer trace（rollout_buffer.py:48-49 Transition 双字段，train_loop.py:1106-1107 写入）+ checkpoint（`episode_metadata=env.current_episode_config()` + `sampler_state`，train_loop.py:1208-1213）+ eval artifact（train_loop.py:547-548 双 hash） |

## E. R2.4 vector adapter（7 项）— 7/7 PASS

| ID | 判定 | 证据 |
|----|------|------|
| E1 | PASS | `reset` 断言 `len(task_specs)==n_envs` 且 `len(seeds)==n_envs`（environment_adapter.py:826-834） |
| E2 | PASS | `reset_at` 仅接受 INITIAL/DONE/VOIDED：ACTIVE 槽在任何 mutation 前原子 RuntimeError（L853-860）；两阶段原子（admission 全槽只读→全过才 mutation）；结果按 env_ids 输入顺序 |
| E3 | PASS | `step_at` 严格三重断言：shape[0]==len(ids)、steps==exec_steps、dim==action_dim=11（L1031-1041） |
| E4 | PASS | StepBatch：reward np.float32 (n_envs,)、terminated/truncated np.bool (n_envs,)、info List[dict] 含 env_id（`_wrap_info` L1123） |
| E5 | PASS | 每 slot 独立实例：`[KinematicUAVEnv(**kwargs) for _ in range(self.n_envs)]`（L673-675）；`_env` 为只读 slot-0 访问器；静态几何只读共享（synth/checker cache） |
| E6 | PASS | done slot 禁 step：`step_at` 原子预检 `done_slots` raise（L1053-1058）；`self._done[env_id]=bool(terminated or truncated)`（L1065）；无自动 reset |
| E7 | PASS | `n_envs=1` 默认合法（L655），断言 `n_envs >= 1` |

## F. R2.5 no-ray 语义（4 项）— 4/4 PASS

| ID | 判定 | 证据 |
|----|------|------|
| F1 | PASS | DEGENERATE_ENCODING 实现于 validated_synth_adapter.py：物理 far endpoint `endpoints_world = cam_world + far·dirs_world`（L229）+ 四重认证 all_finite / unique_passes（≥FAR_DOME_UNIQUE_FLOOR=1024）/ all_at_far（==far±5e-3）/ direction_consistent（≤1e-3），四者全过才 passes（L250-253） |
| F2 | PASS | 单点 all-far 永远 invalid：gate unique 下限（pointcloud_gate.py:141-142 `unique_count < n_unique_min → invalid`）+ allclose-far 双保险（L118-119 docstring：collapse 保持 ALWAYS invalid） |
| F3 | PASS | `git log -- diffusion_policy_3d/rl/synth_pointcloud.py` 空 = 冻结零修改 |
| F4 | PASS | stress v6：10,000 帧 invalid=0 / far_dome=1,970 / report_failures=0 / unique_min=max=4096 / dist_err_max=1.9e-6 |

## G. R2.6 production G3 + 闭合（4 项）— 4/4 PASS

| ID | 判定 | 证据 |
|----|------|------|
| G1 | PASS | test_r2_6_contracts.py 5 个静态 AST 测试（同 factory / 无独立 role RNG / train loop 注入不重建 / CLI 不采样 / CLI 生产默认路径）——实测通过 |
| G2 | PASS | G3 v8：11/11 gates、8 个零计数类 gate 全 0、role delta ≤0.6pp、n_frames=80,000=n_resets×8 口径如实 |
| G3 | PASS | **实测全量 CPU 回归 361 passed / 7 deselected / 95.26s / exit 0**（与归档 `cpu_regression_rework6_20260810.log` 96.05s 一致） |
| G4 | PASS | RTM 更新（"Evidence ledger (R2 rework #6 additions)" 段 + 计数修正 `0f00f85..2bff8b1`）+ `review_R2_rework6.md`（fresh 审查 PASS 0 P0/0 P1/4 P2）+ admission_manifest.json（53 条目）存在 |

## H. R2 完成门（10 requirement）— 10/10 PASS

RTM 十行全 **PASS**：SAMPLER-LIVE / SEED-RESUME / ROLE-TRANSITION / META-SOURCE-HASH / META-EPISODE-HASH / VECTOR-ORDER / NO-RAY-SEMANTICS / PC-VALID / ROLE-PROP / ADMISSION（RL_REBUILD_TRACEABILITY.md L21-27 + L9/L11/L12）。

## I. MasterPlan（8 项）— 8/8 PASS，1 记录偏差

| ID | 判定 | 证据 |
|----|------|------|
| I1 | PASS | 全局约束：vendor 401 零修改（A8 全分支零 diff）；w4/w8/w11/w12/w18 禁入训练（A2）；λ=0.5 DDPM BC + Action 11d（MasterPlan FACT 记录 + adapter step 断言）；禁 obstacle projection —— 三层防护：构造默认 False + `fm_base_policy.py:34` 强制 False + evaluator.py:182-185 `EvalArtifactError` hard-fail（"formal eval hard-fails on True"）；E040 冻结一次 formal guard —— RTM L30 LEGACY-FORMAL-GUARD 标注 OPEN（R3.3 关闭，正确待办）；GPU 512 OOM 禁（batch 128 记录）；禁放宽阈值（各 rework 明确"阈值未放宽"） |
| I2 | PASS | 文件包串行：TODO L14 R1 [x] PASS → L15 R2 [x] PASS（终局 rework #6 后）→ L16 R3 [ ] pending；依赖链 R2 PASS + G3 PASS（v8 11/11）满足，R3 未提前 |
| I3 | PASS | RTM 扩展行在位：LOGP-BATCH-ID（L18）… GATE-ORDER（L33）16 个新 ID；原始 10 行保留未删 |
| I4 | PASS | 强制节奏 §5：R2 六个任务独立 commit（5458504→4eb696c→dece2d9→b8e7227→3db5778→1759805）+ 每轮 rework RED 先失败记录 + 独立审查文件系列（review_R2_rework*.md，最后一轮 0 P0/0 P1/4 P2） |
| I5 | PASS | §6.1 硬故障 × R2 相关项全闭合：invalid obs 入 buffer ↔ invalid_pc gate 0；task/metadata/admission 失效 ↔ hash_missing/manifest_hash_mismatch/admission_blocked 0；eval-only leakage ↔ eval_leakage 0（G3 v8） |
| I6 | PASS | 验收矩阵 DoD B/C/D/F（R2 负责项）：B（no-ray 正确 + invalid buffer=0 + 10k calibration）✅；C（projection=False hard-fail）✅；D（live sampler + 10k 比例 + clearance/reachability/stretch + 无 leakage/fallback）✅；F（vector ordering + fault=0）✅ |
| I7 | [偏差] | §3 状态重置六件套：task_plan 系列 / RL_REBUILD_TRACEABILITY.md / ENTRYPOINT_CLOSURE_REPORT.md 在位；CLAUDE.md 在 repo 父目录 `/home/cjj/AUV/CLAUDE.md`（会话已加载）；**AGENTS.md 不存在**（find 全仓无结果）。性质：§3 为一次性开工动作的"读取清单"且无后续交付义务，MasterPlan 未为 AGENTS.md 定义产出物，不阻塞 R2/R3；如实记录 |
| I8 | PASS | §8 交付物：implementation_manifest_R2 v1-v5（commit_binding 自绑定 c1f7dc1 起）+ review_R2_rework*.md 系列 + g3_gate/ 证据目录（v1-v8 全保留，sealed 零覆盖）+ jindu 同步（STATUS 顶部 R2=PASS 终局 / TODO R2 [x] R3 [ ] / 交接文档 rework6） |

---

## 偏差明细（唯一项）

**I7-AGENTS**：MasterPlan §3（L101）"读取仓库根 AGENTS.md、CLAUDE.md、…"——AGENTS.md 不存在于 `/home/cjj/AUV`（maxdepth 3 搜索无结果）。不构成 [FAIL]：该文件是开工重置的读取清单项，无对应交付物；项目实际以 CLAUDE.md（父目录，会话注入）为准。若需消除偏差：可在 R3 开工前新建仓库根 AGENTS.md（内容为 A-Official-Core 角色/约束摘要），或由用户裁决降级该行。

## 验证边界

- 本次检查为审核者行为：零代码修改（`git status` 验证 worktree clean，除本报告与 task_plan 更新外无改动——该两文件为检查产出物，已列明）
- GPU 测试（7 deselected）未运行——按 Guide §5 "GPU 测试仅在对应 gate 执行"（G7 GPU 阶段），符合规则
- G4 EQUIV-REG 保持 BLOCKED（fixture 未定位）——属 R4 范围，与 R2 一致性无关
