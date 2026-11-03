# 阻塞报告：F5 六策略统一评价在 v3 臂上被公共环境门禁拒绝

- 时间：2026-11-02 02:43（评价启动 01:04，已运行 1 h 39 min）
- 触发指导条款：`F5任务分布与reward v3执行指导 v1.0` 失败协议 —— “任何身份、输入hash、观测、路径、数值门禁或分母不符即停步，保存现场；不得改hash、删困难样本、换seed或关校验”
- 处置：**已停步**（`step_04_eval.sh` 的 `set -euo pipefail` 在第三个任务失败后中止全链），未改任何校验、未改任何 checkpoint、未重跑。

## 1. 现象

`eval_dev_P8` 返回码 1，`EVAL_MASTER_RC=1`，其余两个已完成任务正常。

```
Traceback (most recent call last):
  File "scripts/run_routea_xk_evaluation.py", line 632, in <module>  main()
  File "scripts/run_routea_xk_evaluation.py", line 628, in main      globals()[args.command...](args)
  File "scripts/run_routea_xk_evaluation.py", line 383, in worker
    adapter=make_adapter(system,config,world,t['role'])
  File ".../scripts/routea_paired_support/runtime.py", line 193, in make_adapter
    raise ValueError("common environment is not v2/H4/8")
```

失败发生在**第一个回合之前**（`eval_dev/full/P8/` 下只有 `policy_provenance.json`，无任何回合记录）。

## 2. 原因链（代码证据）

| 步骤 | 证据 |
|------|------|
| 1. 评价系统由 checkpoint 自带 `training_config` 构建 | `scripts/run_routea_xk_evaluation.py:287` `config = OmegaConf.create(state['training_config'])` |
| 2. P8/P10 的 `training_config.reward_version = v3` | `state.pt` 直接读取：P8=v3(task_bank=True)、P9=v2、P10=v3(task_bank=True)、P11=v2 |
| 3. 系统按 v3 构建 reward | 诊断输出：`system.reward = rewards.v3.facade.RewardV3Facade`，`.version = 'v3'` |
| 4. 环境 reward 版本取自系统 reward | `scripts/evaluate_dppo_uav_formal_endpoint.py:429` `reward_version=getattr(system.reward,"version","v13")` |
| 5. 评价侧门禁要求 v2 | `scripts/routea_paired_support/runtime.py:193`：要求 `reward_version=="v2" and n_exec_steps==4 and rl_action_steps==8 and traj_dt==.3 and max_episode_steps==60 and collision_thresh==.1` |

即：**指导给的 `worker` 逐臂流程，对 v3 训练臂必然撞门禁**；v2 臂（P9/P11）不受影响。这与策略好坏无关，是"评价环境必须是 v2/H4/8 公共环境"这一契约与"逐臂按自身训练配置构建环境"这一流程之间的矛盾。

诊断脚本与输出：`diagnose_p8_env.py`（只读，未改仓库）。

## 3. 已完成部分（现场保留，可用）

| 任务 | rc | 记录数 | 任务数 × 重复 | bank_hash | 结果 |
|------|----|--------|----------------|-----------|------|
| dev/bc | 0 | 420 | 84 × 5 | `863bf1fb…` | reached 110 / collision 150 / time_limit 160 |
| dev/P6 | 0 | 420 | 84 × 5 | `863bf1fb…` | reached 203 / collision 143 / time_limit 74 |
| dev/P8 | 1 | 0 | — | — | 撞门禁，未产生回合 |

- 两任务的 420 条记录 `status` 全为 `valid`，`initial_hash` 与冻结 dev420 bank 逐 (task, repeat) 完全一致（420/420）。
- dev/P6 的 reached 203 / collision 143 / time_limit 74 与 2026-10-31 的 #E2 dev420 评价数值完全相同 —— 本次评价流程本身可复现。
- 未跑：bc val、P6 val、P8/P9/P10/P11 的全部 dev 与 val。

## 4. 事实补充（供裁决参考）

- `scripts/routea_paired_support/runtime.py`（门禁所在）**不在** 137 文件执行源身份闭包内；闭包 `e5f9c80a…` 含 `scripts/evaluate_dppo_uav_formal_endpoint.py`。即：改前者不会破坏既有 checkpoint 加载，改后者会。
- 仓库内存在"公共环境"设计通道：`runtime.py:100 common_system(root, cat)` 用**目录（catalogue）里的 BC 配置**构建唯一系统，再 `load_actor` 逐策略换 actor（`scripts/routea_paired_support/workflow.py:106`、`mechanism.py:196`、`ego_workflow.py:56`）。这条路的评价环境天然是 v2 公共环境。其现有目录 `scripts/routea_paired_support/policy_catalog_v1.json` 只登记 bc/a0/r1 七个策略，`evaluation_config` 为 null（回落到 bc 策略配置）。
- 本工具的 `matrix` 子命令不是上述通道：它只是对 `worker` 逐策略 fan-out（`run_routea_xk_evaluation.py:467-489`），同样按各臂训练配置构建环境。
- 本次交付 23 个文件**未修改** `run_routea_xk_evaluation.py` 与 `routea_paired_support/runtime.py`，门禁是既有代码。

## 5. 待裁决选项（执行端不自行选择）

**A. 改用目录公共环境通道评价六策略**
- 做法：为 F5 建一份 catalogue（bc/P6/P8/P9/P10/P11 的 checkpoint、update、state_sha256、runtime_identity_hash），`common_system` 用 BC 配置构建唯一 v2 环境，逐策略 `load_actor`。
- 优点：与仓库既有"公共环境"设计一致，六策略环境严格一致，不动任何门禁。
- 代价：`workflow.run` 需要的 bank 选择/BC 基线参考机制面向 xk/paper schema，dev420 bank（`routea_train_domain_paired_dev_v1`）是否直接可用需设计侧确认；产出记录的 provenance 字段与 `worker` 不同，与历史 dev420 数字需说明可比性。

**B. 授权在评价层对 v3 臂做窄放行**
- 做法：门禁保留其余 5 项，仅把 `reward_version` 判据改为"v2 或 v3"（v3 = v2 + `risk_close`，回合动力学与结局定义不变）。
- 优点：改动最小，六策略仍走同一 `worker` 流程，产出字段与既往 dev420 评价完全同构；`runtime.py` 不在身份闭包内，不影响 checkpoint 加载。
- 代价：动了校验本身，需要设计侧明确"reward 只影响训练、不影响评价结局"这一前提。

**C. 评价时用 v2 配置构建环境、只载入 v3 臂的 actor**
- 做法：给 `worker` 增加"用指定 v2 配置构建系统"的入参，actor 仍取自 v3 checkpoint。
- 优点：不碰门禁，六策略同环境。
- 代价：需要新增参数与配套 provenance 记录，属于代码变更。

## 6. 执行端状态

- 已停步，等待裁决。未启动任何后续任务（分层冻结、分析、回传包均未开始）。
- 现场文件：`results/routea_f5_run01/eval_dev/full/{bc,P6,P8}`、`results/routea_f5_run01/execution/eval_dev_*.{log,rc}`、`eval_master.log`、`eval_master.rc`。
- 训练四个臂的 checkpoint 完整未动（各 9 个）。
