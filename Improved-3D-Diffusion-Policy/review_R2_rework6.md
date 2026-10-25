# R2 rework #6 独立审查报告（fresh review，零上下文）

## 1. 审查信息

- 日期：2026-08-10
- 审查者：独立审查者（未参与任何此前轮次；全部结论来自亲自读取文件、命令输出与实测）
- 覆盖 commits（按序）：`2441188`（fix: rework #6 eval entry preflight）→ `fad10d4`（test: lifecycle evidence v3）→ `1177a0c`（docs: rework #6 evidence ledger + task_plan sync）
- 基线：`2bff8b1`（rework #5 closure commit）；分支 `a-official-core-rework`
- 审查方法：git 命令实测（log/rev-list/diff/rev-parse/reflog/status）；`train_loop.py` 与 `test_p7_eval_persistence.py` 全文相关段读取；conda idp3 CPU 环境重跑 2 组测试；4 份证据 JSON 用 `python3 -c` 逐字段读取；5 个文件 sha256 用 `sha256sum` 实测并与 RTM ledger 逐字符比对；文档闭环与约束合规逐项核验

## 2. 逐项核验结果

### A. 裁决 claims 与修复代码核验

**[FACT] A.1 commit 序列与计数**：`git rev-list --count 0f00f85..2bff8b1` = **5**；`git diff --stat 0f00f85..2bff8b1` = **9 files**（455 insertions / 21 deletions）。裁决的计数修正（5 commits / 9 files，非 6/8）**正确**。`git log --oneline 0f00f85..HEAD` 确认 rework #6 三 commit（2441188 → fad10d4 → 1177a0c）按序追加在 `2bff8b1`（rework #5 closure）之后，`HEAD=1177a0c`。文件归属实测：2441188 = train_loop.py + test_p7_eval_persistence.py；fad10d4 = scripts/r2_rework4_lifecycle_evidence.py；1177a0c = 仅 3 个 docs 文件（RL_REBUILD_TRACEABILITY.md、task_plan_r2_rework5.md、task_plan_r2_rework6.md）。

**[FACT] A.2 入口 preflight 位置与四检查**（train_loop.py）：
- preflight 位于函数第一句 `device = loop_config.device`（L790）之后、`configure_numerical_mode`（L823）之前；`restore_training_state`（L828）、rollout、`ppo_update`（L1173）、`output_dir.mkdir`（L1182）、`CheckpointBuilder.save`（L1214）、eval JSON 写入（L1250）全部在其后。顺序满足"任何 rollout / resume restore / PPO update / 文件写入之前"。
- 四检查齐备（L799-818，`if loop_config.eval_n_episodes > 0:` 内）：(a) `mode != "one_update"` → RuntimeError；(b) `eval_env is None` → RuntimeError；(c) `getattr(eval_env, "n_envs", None) != 1` → RuntimeError；(d) `eval_env is env` → RuntimeError（"dedicated instance"）。
- one_update 分支内旧检查已移除（diff 实测：原 6 行 None 检查被删除），仅保留运行调用 `run_paired_eval`（L1220-1223）+ 注释说明校验已前移。
- `run_paired_eval` 内部 n_envs==1 检查保留（L517-521，defense-in-depth）。
- LoopConfig `__post_init__` 新增 `assert self.eval_n_episodes >= 0`（L165，P2 落实）。

**[FACT] A.3 四个新测试**（test_p7_eval_persistence.py，rework #6 段 L137-272）：
- `test_no_update_with_eval_request_hard_fails`：`pytest.raises(RuntimeError, match="one_update")` ✓
- `test_missing_eval_env_fails_before_rollout_update_and_checkpoint`：断言含 `env.reset_calls == 0 and env.step_calls == 0`、policy state_dict 逐参数 `torch.equal`、`not actor_optimizer.state_dict()["state"]`、`list(tmp_path.iterdir()) == []`（输出目录空、无 checkpoint）✓
- `test_negative_eval_n_episodes_rejected`：`pytest.raises(AssertionError)` ✓
- `test_training_env_cannot_be_reused_as_eval_env`：`pytest.raises(RuntimeError, match="dedicated")` ✓
- `_CountingMockUAVEnv` 覆写 `reset_at`/`step` 计数 ✓

**[FACT] A.4 计数修正**：见 A.1，5 commits / 9 files 实测无误。

### B. 测试重跑（本人实测，idp3 CPU）

```
$ python -m pytest tests/dppo_parity/test_p7_eval_persistence.py -q -p no:cacheprovider -m "not gpu"
11 passed in 4.40s
$ python -m pytest tests/dppo_parity/test_lifecycle_rework4.py -q -p no:cacheprovider -m "not gpu"
10 passed in 4.05s
```
- p7 实测 11 passed（= 7 旧 + 4 新 preflight），lifecycle 实测 10 passed，合计 21/21 全绿，与 lifecycle v3 证据声明一致。
- `--collect-only` 确认 p7 文件 11 个测试，含 4 个新测试名。

### C. 证据核验（逐字段实测）

**[FACT] C.1 tree_hash 自绑定**：4 份证据 JSON 的 tree_hash 全部为 `0a067a385fe35b3a587ecda6816b6879dc08755a`；实测 `fad10d4^{tree}` = 同一值（证据生成 commit 的树）。**注意**：`git rev-parse HEAD^{tree}` = `d783a41b2ba909906e45e45fdcf7c55c7d44de0e` ≠ 0a067a，因为证据生成后追加了 docs commit `1177a0c`（仅 3 个 docs 文件，已用 `git show --stat` 实测无代码改动）。此模式与 rework #5 完全一致（rework #5 证据 tree `9db41a21` = `eee7230^{tree}`，closure docs commit 在其后；RTM ledger 明确记载 "docs commit ... does not touch tested code, so the evidence remains bound"）。证据绑定关系完整且被 ledger 明示，无篡改风险。见 P2-2。

**[FACT] C.2 证据内容**：
- G3 v8：11/11 gates 全 PASS（admission_blocked/clearance/dense_path_stretch_fail/eval_leakage/hash_missing/invalid_pc/manifest_hash_mismatch/reachability/role_proportion/role_unknown/unvalidated_fallback）；n_resets=10000；n_frames=80000；role_proportion_asserted=True；resolved_world_count=21==manifest_world_count=21 ✓
- role v6：pass=True；completed=120==attempted==n_episodes；voided=0；in_flight_at_collection_stop=0；transitions=3457==env_steps；gate_invalid_count=0；7 invariants 全 True ✓
- stress v6：pass=True；total_frames=10000；invalid=0；far_dome_frames=1970；report_failures=0 ✓
- lifecycle v3：pass=True；exit_code=0；内嵌 pytest 输出含 "21 passed in 4.40s" 且 PASSED 行计数=21；recovery_paths A 5/B 4/C 4/D 1/E 4 全 PASSED；missing=[]；test_files 两文件 ✓；log_sha256 用内嵌输出重算一致（630a4f16...，自洽）✓

**[FACT] C.3 sha256 逐字符比对**（`sha256sum` 实测 vs RTM ledger）：
- G3_GATE_REPORT_v8.json：`70ba1248136bf338b6e15f9bd79feaa62c33ffe326e4f785b26d4b7c7f31a374` ✓ 一致
- ROLE_METRICS_EVIDENCE_v6.json：`c1b2f149cff837ac0f5751b0ae5d9201001a5a76b921c80a51c1dfbd5af04f5f` ✓ 一致
- NO_RAY_STRESS_EVIDENCE_v6.json：`d51eac002b08f0b4f14edf65ce1f2135eb542e06275726c00ebe2c04d0200229` ✓ 一致
- LIFECYCLE_EVIDENCE_v3.json：`900b0b239ff25571a048af427a5a859d85b9e984f34f92e945413472dc54fca6` ✓ 一致
- cpu_regression_rework6_20260810.log：`8dd93721bb4202a8ac12968c03dd1b505ce099f7b0c9cd670ddc07b8d16532ef` ✓ 一致；内容实测 **361 passed, 7 deselected in 96.05s** ✓

### D. 文档闭环核验

**[FACT] D.1 RTM ledger**：`RL_REBUILD_TRACEABILITY.md` L625 起新增 "Evidence ledger (R2 rework #6 additions)" 段：第六轮裁决原文要点、5 个 sha256（与实测逐字符一致）、361 passed/7 deselected/96.05s、5 commits/9 files 修正、证据 tree 绑定说明（2441188+fad10d4，tree 0a067a）✓
**[FACT] D.2 task_plan_r2_rework6.md**：Phase 0-5 勾选完成、Phase 6 进行中、Phase 7 待启动；Status 段更新（Phase 6 证据链状态 + tree 绑定值 + sha256）✓
**[FACT] D.3 task_plan_r2_rework5.md**：末尾有第六轮重开注记（明确本文件为 rework #5 历史记录，rework #6 以新 plan + STATUS 为准）✓
**[FACT] D.4 设计文档 v1.2**（repo 外 `/home/cjj/AUV/program/sheji/RouteA_RL_Training_System_A_OfficialCore_Design_v1_2.md`）：§0.6.3（L129）完整记录第六轮裁决四检查 + dedicated instance + LoopConfig 非负断言 + 4 回归测试；§4.17 第 2 条强化（L1166，含 (d) `eval_env is env` → RuntimeError）；header（L8/L11）引用 rework #6 ✓
**[FACT] D.5 STATUS.md**：顶部 `R2 = REWORK REQUIRED（第六轮审计裁决 2026-08-10，窄范围入口 preflight）`，rework #6 进度 Phase 0-6 完成、Phase 7 进行中 ✓

### E. 约束合规核验

**[FACT] E.1 sealed 证据零覆盖**：g3_gate 目录实测 G3 v1-v8、LIFECYCLE v1-v3、STRESS (未版本化)+v2-v6、ROLE (未版本化)+v2-v6 全系列旧文件存在；抽查 3 个旧文件 sha256 与 rework #5 ledger 逐字符一致（G3 v7=`1c911e9c...`、role v5=`d68659cb...`、lifecycle v2=`877336d1...`）——旧证据未被改动/覆盖 ✓
**[FACT] E.2 vendor 零改动**：`git diff 2bff8b1..HEAD --stat -- third_party/` 无输出 ✓
**[FACT] E.3 E040 冻结零改动**：`git diff 2bff8b1..HEAD --stat -- | grep -i e040` 无输出 ✓
**[FACT] E.4 工作区状态**：`git status --short` 干净（写本报告前实测）✓
**[FACT] E.5 reset 检查**：reflog 有两条 `reset: moving to HEAD`（HEAD@{40}/@{42}，2026-08-08 21:04 / 20:30），早于 rework #5/#6（2026-08-10）；rework #6 窗口（13:43-13:54）无任何 reset 痕迹 ✓

### 补充核验（超出要求，主动执行）

**[FACT]** `run_e2e_batch` 全部调用方（scripts/role_metrics_probe.py、scripts/g6_one_update_resume_probe.py）均未设置 `eval_n_episodes>0`，不受新增 dedicated-instance 检查影响，无回归风险。
**[FACT]** lifecycle v3 内嵌 pytest 输出与 `log_sha256` 重算自洽；"21 passed in 4.40s" 来自单进程合并运行（两文件一次 pytest 调用），计数与独立重跑一致（11+10）。

## 3. 测试重跑实际输出

```
$ conda activate idp3 && python -m pytest tests/dppo_parity/test_p7_eval_persistence.py -q -p no:cacheprovider -m "not gpu"
...........                                                              [100%]
11 passed in 4.40s

$ python -m pytest tests/dppo_parity/test_lifecycle_rework4.py -q -p no:cacheprovider -m "not gpu"
..........                                                              [100%]
10 passed in 4.05s
```

## 4. 发现的问题列表

- **P0（阻塞 PASS，必须修复）**：无
- **P1（严重但可说明）**：无
- **P2（观察项）**：
  1. **RTM ledger 范围记号不精确**（RL_REBUILD_TRACEABILITY.md L644）：写 "`0f00f85..HEAD` = 5 commits / 9 files"。实测 `0f00f85..2bff8b1` = 5/9（rework #5 范围，裁决所指）正确，但写成 `..HEAD` 时该范围实为 8 commits / 10 files（含 rework #6 三 commit）。建议改为 `0f00f85..2bff8b1`。不影响结论——数字本身对应关系已核实无误。
  2. **HEAD^{tree} ≠ 证据 tree_hash（记录在案，非缺陷）**：审查要求中的 "`git rev-parse HEAD^{tree}` 必须等于 0a067a" 在 HEAD（`d783a41b`）不成立；实测证据 tree_hash `0a067a` 精确等于证据生成 commit `fad10d4^{tree}`，且其后唯一 commit `1177a0c` 仅改 3 个 docs 文件（已实测）。该"证据后追加 docs commit"模式与 rework #5 完全一致并被 ledger 明示（"docs commit does not touch tested code, so the evidence remains bound"）。正确不变量为 `证据 tree_hash == 证据生成 commit^{tree}`，成立。建议未来审查模板将判定式改为生成 commit^{tree}。
  3. **lifecycle 时长口径**：证据内嵌输出 "21 passed in 4.40s"（单进程合并运行）与本人两次独立运行的 4.40s+4.05s 不同，属机器负载/进程启动差异，计数（21/21）与全部字段已独立验证，无事实不符。
  4. **历史 reflog 存在两条 2026-08-08 的 reset 记录**：早于本 rework 窗口（2026-08-10），为历史遗留；rework #6 期间无 reset。仅记录备查。

## 5. 最终结论

**PASS（0 P0 / 0 P1 / 4 P2 观察项）**

第六轮裁决全部四项（P1 入口 preflight、P2 LoopConfig 非负断言、dedicated instance 强制、4 个必须测试）已在代码中正确落实并实测验证；测试 21/21 全绿（本人重跑）；5 个 sha256 逐字符一致；证据 tree 自绑定完整；文档闭环（RTM/两份 task plan/设计 v1.2/STATUS）齐备；sealed 证据、vendor、E040 冻结零改动。P2 均为观察项，不阻塞。
