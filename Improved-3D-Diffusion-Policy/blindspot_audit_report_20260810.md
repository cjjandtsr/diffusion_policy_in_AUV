# 盲点模式审计报告：R2 六轮审查反复忽略点 + 针对性复检

- 日期: 2026-08-10
- 依据: 用户要求——"按照多次审查报告的思路总结经常忽略的地方在哪，按着这思路继续检查"
- 输入: review_R2_rework.md ~ review_R2_rework6.md（6 份独立审查）+ 第六轮审计裁决记录（STATUS/IMPLEMENTATION_LOG）
- 方法: 提炼 P1/P2 发现 → 归类 → 对当前 HEAD（`dd94c22`）程序逐类针对性复检

---

## 一、六轮审查反复忽略的模式（按出现频次排序）

### P0 级（出现 ≥3 次）：必须根治的类别

**模式 1 — 状态机泄漏 / 双状态不同步**（rework #4 P1-1/P1-3、rework #5 P1-1）
- 形态：loop 内联变量（`s["done"]`/`s["ep_has_fault"]`）与 adapter slot 生命周期是两套状态；异常/超时/故障退出路径只改了一套。同型问题在 train 路径修完后又出现在 eval 路径（`_eval_episode` loop 超时泄漏 ACTIVE）。
- 根因：每个"退出路径"（continue/raise/timeout/fault/reject/voided/truncated）都要逐一核对 slot 状态，人工枚举容易漏。

**模式 2 — 证据 ≠ 生产路径**（rework #1 P1-2/P1-4、rework #2 P1）
- 形态：证据脚本用默认/非生产参数（probe 未传 n_envs → 单槽运行冒充 4 槽）、只测合成/子路径（no-ray 只测合成不测 adapter 全链路）。
- 根因：证据脚本与生产 launch 的参数没有强制对齐机制，靠人工保持。

### P1 级（出现 2 次）：每次修复都不该再犯的类别

**模式 3 — hash 覆盖面**（rework #1 P1-6、rework #3 P1-3/P2-c）
- 形态：白名单 = 漏配（source hash 漏 15 生产字段）；默认值 = 幻影文件（obstacle_json 默认与实际加载路径不符）；hash 范围超集/子集错误（glob 180 vs pool 21）。

**模式 4 — 边界超额静默处理**（rework #1 P2-8、rework #3 P1-1）
- 形态：refill 超额启动 + tail 静默丢弃；episode budget 超出部分不报错。

**模式 5 — 原子性 / 部分执行**（rework #1 P2-7、rework #4 P1-2）
- 形态：多槽操作中途失败留下部分 mutation（step 逐槽半推进、reset_at 逐槽先 admission 后 mutation）。

**模式 6 — 验证时序**（rework #6 P1）
- 形态：验证嵌套在分支内、副作用（rollout/update/写文件）之后才执行；错误路径静默返回。

### P2 级（单次出现，但形态在别处可迁移）

**模式 7 — 输入校验缺失**（rework #6 P2）：LoopConfig 负数静默当作"关闭"。
**模式 8 — 状态前置校验缺失**（rework #3 P1-2）：reset 未检查 ACTIVE 就静默覆盖 in-flight episode。
**模式 9 — 实例隔离**（rework #5 P1-2、rework #6 dedicated）：eval 复用训练 env。
**模式 10 — 数字口径 / 文档-证据版本脱节**（rework #1 P2-10、rework #2 manifest、rework #4 P2-1/P2-3）：changed_files 计数、旧版本证据引用、未勾选 checkbox。

---

## 二、按盲点类别对当前程序的复检结果

### 模式 1（状态机泄漏）— ✅ 闭合，已全路径核对

train_loop 所有退出路径的 slot 关闭（代码位置实证）：

| 退出路径 | slot 处理 | 位置 |
|----------|-----------|------|
| reset 帧无效（REJECT_EPISODE） | `reject_episode([env_id])`（ACTIVE→DONE） | train_loop.py:993 |
| reset AdmissionExhausted | 两阶段原子未触槽，保持可 reset | train_loop.py:977-982 |
| rollout 首 fault（SAFE_END） | `mark_voided([env_id])`（ACTIVE→VOIDED） | train_loop.py:1060 |
| rollout HARD_FAILURE | fail-fast raise（batch 整体失败，可接受） | train_loop.py:1062-1065 |
| loop max_ep_len 截断 | `mark_truncated([env_id])` | train_loop.py:1119 |
| EpisodeCollectionIncomplete | fail-fast raise（checkpoint 未写，无残留） | train_loop.py:1136 |
| eval reset 帧无效 | `reject_episode([0])` | train_loop.py:432 |
| eval obs fault | `mark_voided([0])`（is_active 守卫） | train_loop.py:473 |
| eval loop 超时 | `mark_truncated([0])`（is_active 守卫防 double-close） | train_loop.py:487 |
| G3 audit 同槽复用 | `abort_episode(range(n_envs))` | gate_audit.py:155 |
| stress 同槽复用 | `abort_episode([0])` | no_ray_stress.py:159 |

adapter 四关闭方法（mark_voided/abort_episode/reject_episode/mark_truncated）均全量 ACTIVE 预检后设置，零部分 mutation（environment_adapter.py:935/955/974/992）。

### 模式 2（证据≠生产）— ⚠️ 发现 1 项参数偏差（记录，非契约 FAIL）

逐脚本对比构造参数 vs 生产：

| 脚本 | max_episode_steps | 生产默认 | 其他参数 |
|------|-------------------|----------|----------|
| g3_10k_reset_gate_audit.py | **60** | **100** | manifest 真实 + resolved 21；reset-only 审计（不 step，ep_len 无影响） |
| no_ray_stress.py | **30** | **100** | curriculum C2 / reward v16 / disturb 默认一致 |
| role_metrics_probe.py | **30**（--max-ep-len） | **100** | curriculum C2 / reward v16 / n_envs=4 已接线（rework #2 P1） |

生产来源：`train_ddpm_ppo_meanlogp_3_e040.py:735,1039`（`max_episode_steps=args.max_ep_len`，`--max_ep_len` 默认 100）。
影响评估：三个脚本验证的对象（role 比例 / 帧有效性 / 状态机契约）均与 episode 长度无关——不构成契约 FAIL，但与 rework #2 P1（probe 参数未接线）同类，属于"证据参数对齐"盲点。已记录的修复方向：R3 阶段统一三个脚本的 max_ep_len 到生产值，或在证据 JSON 中如实声明 ep_len 差异。

### 模式 3（hash 覆盖面）— ✅ 闭合

- `episode_config_hash`：8 字段 {world_uid/role/split/admission_id/episode_seed/start/goal/start_yaw}，admission 指标明确不入 hash（设计意图，environment_adapter.py:569-572）
- `manifest_entry_hash` 侧（entry_metadata_hash，integration.py）：全字段无白名单 + resolved 四字段（rework #3 已修）
- `_spec_set_hash`：spec 全 7 字段覆盖（environment_adapter.py:300-323）——sampler 恢复校验无盲区

### 模式 4（边界超额）— ✅ 闭合

- refill per-slot 精确预算（completed+active ≤ n_episodes，rework #3 已修，`test_multi_slot_precise_budget_no_active_at_return` 实测通过）
- attempt_limit = max(2×n_episodes, 4) + 4（train_loop.py:866）——正常训练不可能误触发
- LoopConfig 全部计数非负断言（模式 7 见下）

### 模式 5（原子性）— ✅ 闭合

- reset_at 两阶段（admission 全槽只读 → 全过才 mutation，AdmissionExhausted 零部分 reset）
- step_at / step 全量预检（done 槽在任何 step 前 raise）
- 四关闭方法全量 ACTIVE 预检（见模式 1 表）

### 模式 6（验证时序）— ✅ 闭合

- eval 合同入口 preflight 四检查在 configure_numerical_mode / restore_training_state / ppo_update / CheckpointBuilder.save 之前（rework #6，train_loop.py:791-818）
- restore 顺序固定（policy→LoRA→critic→optimizer→RNG→sampler 强制恢复），CheckpointBuilder.load 抛 CheckpointError
- eval 循环内 invalid 帧 gate 在 transition 入 buffer 之前（PC-VALID）

### 模式 7（输入校验）— ✅ 闭合

LoopConfig `__post_init__` 6 项断言：mode 枚举 / n_episodes≥1 / max_ep_len≥1 / eval_n_episodes≥0 / gate_phase∈PHASE_GATES / 两个 denoising std>0（train_loop.py:164-174）

### 模式 8（状态前置校验）— ✅ 闭合

reset_at 拒绝 ACTIVE（任何 mutation 前）+ 关闭方法拒绝非 ACTIVE + is_active() 守卫防 double-close

### 模式 9（实例隔离）— ✅ 闭合

- eval 强制 dedicated n_envs=1 实例，`eval_env is env` → RuntimeError（rework #6）
- bc_policy = `_bc_snapshot(policy)` 冻结行为快照，非共享可变实例
- 每 slot 独立 KinematicUAVEnv；synth/checker cache 只读共享

### 模式 10（数字口径）— ✅ 闭合（上一轮 I 维已查）

计数修正 `0f00f85..2bff8b1` = 5 commits / 9 files；manifest v1-v5 commit_binding 自绑定；evidence 版本引用 v8/_v6/v6/v3 一致。

---

## 三、结论

**9/10 盲点类别在程序层面闭合；1 项偏差（模式 2：证据脚本 max_episode_steps 30/60 vs 生产默认 100）为记录项，非契约 FAIL。**

按盲点模式复检未发现新的状态机泄漏、hash 盲区、边界静默、部分执行或验证时序问题——主要原因是 rework #1-#6 把模式 3/4/5/6 的修复都做成了带原子预检/两阶段的结构，且测试以 RED-first 钉住。

## 四、建议（R3 开工前）

1. **[P1] 证据参数对齐**：R3 证据脚本统一 `max_episode_steps` 到生产值（100），或证据 JSON 增加 `ep_len_note` 字段如实声明差异（与 rework #2 的 resolved_n_envs 同机制）
2. **[P2] 模式 1 预防**：R3 的 EVAL-STD 若新增 eval 退出路径，沿用 `is_active()` 守卫 + mark_* 显式关闭的既定模式
3. **[P2] 模式 2 制度化**：把"证据脚本参数 vs 生产 launch 参数"的对比检查写成 R3 证据清单项（当前靠人工保持，正是本报告发现的偏差来源）
