# Task Plan: Route A A1 val20 与同状态机制验证执行（指导 v1.2，2026-10-21）

## Goal
按指导 v1.2 完成 val20 两次同入口 BC/RL 配对评价 + A1 同状态恢复诊断 + B1280 早期固定状态窗口，rc=0 全验收，回传 Windows wsl_snapshot/routea_a1_mechanism_val20_run03/。不做能力判读、不重训、不碰 test12；run01/run02 证据不改。

## 权威输入
- 指导：《Route A A1：val20 与同状态机制验证执行指导 v1.2》（用户粘贴，2026-10-21）——v1.1 到达边界阻塞的裁决回应
- v1.1 阻塞报告：run02/execution/CODE_BLOCKER_REPORT_v1_1.md

## Phases
- [x] Phase 0: 交付核验 —— 六文件 WIN=WSL=v1.2 表 hash 逐位一致（已确认，无需复制）；入口字节保持 A1 13455ae9
- [x] Phase 1: 机制审查 —— 新调度脚本逐行审查（窄边界接纳/四拒/审计文件/身份不动）；新测试 6 用例覆盖
- [x] Phase 2: 身份预检 —— 闭包 133、调度脚本在闭包外、四 checkpoint 身份重建全 MATCH
- [x] Phase 3: 指导 §1 准备段（pytest 15 passed rc=0）
- [x] Phase 4: 指导 §2 val20（u128/u256 rc=0、边界审计各 25 条、BC 重复 0 差异、VAL_PAIR_PASS 80）
- [x] Phase 5: 指导 §3 same_state_main（rc=0，18/3/1080/162）
- [x] Phase 6: 指导 §4 same_state_early（rc=0，18/12/4320/648）
- [x] Phase 7: §5 回传 Windows（394 文件 0 FAILED，无 state.pt）+ 归档（§4.7/jiaojie/记忆）

## Key Facts（2026-10-21）
- 六交付文件 hash：entry=13455ae9…（不变）/ dispatch=2f2ade59…（v1.2 边界接纳）/ recovery=cffcbead… / bank=9527f70b… / test_recovery=76acf968… / test_val_only=1dbfe908…（6 用例）
- v1.2 语义：`Val20EpisodeRecord` 仅在到达错误串 + reached/raw=reached + 无安全标志 + goal 存在时，按 `max(1e-6m, 0.5·‖float32半ULP‖+1e-9m)` 接纳 `0.5 < d ≤ 0.5+界`；success 一致覆写；审计文件含调度代码 hash 与逐条原始坐标/超界量

## Status
**DONE（2026-10-21 19:17→20:50，MASTER_EXIT rc=0）** — 四段全过、§5 回传核对完成（394 OK/0 FAILED），能力判读待 Codex
