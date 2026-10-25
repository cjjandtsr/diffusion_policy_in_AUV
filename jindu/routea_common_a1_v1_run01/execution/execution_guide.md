# Route A：共同协议与 A1 两臂训练、统一评价执行指导

| 字段 | 内容 |
|---|---|
| 日期 / 交付版本 | 2026-10-18 / v1.1（保留原文件入口） |
| 状态 | Windows源码已实现、本地验证完成；尚未在WSL启动本轮训练 |
| 本轮问题 | 相同交互与pair预算下，2560/20组织能否比共同对照1280/40更好保留、增加安全到达能力 |
| 主要能力目标 | 密集且真实受阻任务的绕行、必要退让后恢复；同时要求总体安全到达净收益 |
| 实施授权 | 用户明确要求“直接开始实现”；实现不等于训练能力已通过 |
| 设计 | `sheji/2026-10-16_RouteA_更新组织与共同任务协议_实现前设计_v1_0.md` |

## 1. 交付范围、位置与协议

Windows源码根：`C:/Users/cjj/Desktop/IDP3/chengxu/Improved-3D-Diffusion-Policy/`。
WSL目标根：`/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy/`。
以下所有源码/测试/配置都保持根内相对路径一致。按相邻文件 `2026-10-17_RouteA_共同协议A1_交付sha256.txt` 复制37个文件；它同时是完整修改与新增清单。没有删除活动入口或历史证据。该清单的每一行给出文件SHA256，不用历史版本hash代替本轮hash。

v1.1在原A1交付上合并同状态恢复诊断与轨迹行为分析；训练配置、预算及H4执行不变。追加交付的8文件为统一评价入口、reference兼容检查、recovery辅助模块、冻结24状态JSON、已有行为分析器及其测试、恢复测试、实际环境测试；原runtime同步登记这些评价来源。恢复诊断仅是该评价入口的可选阶段，不另起一套模型评价程序。

修改集中在现有环境与reward、critic上下文、buffer/GAE、pair schedule、配置/训练接线、checkpoint、审计和统一评价预算识别。保留同一训练入口 `scripts/train_dppo_uav_rebuild.py` 和同一评价入口 `scripts/run_routea_paired_comparison.py`。`scripts/smoke_routea_r1.py`也支持两种协议；原迹成本分析程序是独立离线分析职责，不是另一个训练/评价控制器。

| 数量 / 协议 | 共同对照 B1280 | A1 B2560 |
|---|---:|---:|
| 配置文件名（位于`diffusion_policy_3d/rl/dppo_uav_rebuild/config/`） | `train_e5_dppo_common_b1280_seed202609203_v1.yaml` | `train_e5_dppo_common_b2560_seed202609203_v1.yaml` |
| seed / 轮数 / 每轮环境transition | 202609203 / 256 / 1024 | 相同 |
| 去噪位置 / pair遍数 / micro | 10 / 5 / 128 | 相同 |
| 每轮pair / 每轮actor、critic各提交 | 51200 / 40 | 51200 / 20 |
| 最终transition / pair计算 | 262144 / 13107200 | 相同 |
| 最终actor、critic各累计提交（零拒绝时） | 10240 | 5120 |
| actor LR | u1=2e-5，线性到u256=1e-5 | 相同 |
| critic LR / weight decay | .001 / .01 | 相同 |
| checkpoint | u0、u1、u8至u256每8轮，共34个 | 相同 |

共同新协议为 `task_protocol.name=open_space_finite_v1`：训练关闭人工x/y/z空间终止，保留碰撞/触地；实际H4执行路径按真实经过时长收取Cs=1、72s有界渐进空间成本；60决策真实期限不bootstrap，1024收集边界仍使用reset前下一状态bootstrap；critic使用6维任务上下文，actor仍为原三帧576条件。H16/H8/H4、LoRA、原reward其余分项、任务池与扰动保持本轮设计。它是新的共同组合，不冒称旧R1单项消融。

新checkpoint的`state.pt`保存resolved training_config，统一评价据此核实40/20实际预算；旧checkpoint继续读取原S2语义。评价只加载actor，采用共同v3环境；评价分数不用于重建新训练reward。切换权重只更改同一评价命令的`--checkpoint`地址。

## 2. 输入、复制与最小准备

沿用已有`idp3`环境：`/home/cjj/miniconda3/envs/idp3/bin/python`，Python3.8.20。历史回传为torch2.4.1+cu121、NumPy1.23.5；以本轮环境记录为准，不自行升级。E5权重、官方DPPO vendor、admission及train世界按两份配置原路径存在；eval-only、val20/test12不进入训练。本轮不推进尚未验收的目录迁移。

用户将37个文件复制到WSL源码对应位置后，把hash清单和本文复制到下面LOG目录（均为交付资料，不是新增执行源码）。所有日志、测试临时文件、源码/配置快照、轨迹和回传清单位于program；模型权重仍位于ROOT/data/outputs。先在一个独立Bash会话执行：

```bash
set -euo pipefail
ROOT=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
PROGRAM=/home/cjj/AUV/program
PY=/home/cjj/miniconda3/envs/idp3/bin/python
LOG="$PROGRAM/routea_common_a1_v1_run01/execution"
CONFIG_ROOT=diffusion_policy_3d/rl/dppo_uav_rebuild/config
test ! -e "$LOG"
mkdir -p "$LOG/delivery" "$LOG/tmp"
export PYTHONPATH="$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$LOG/tmp"
export MPLCONFIGDIR="$LOG/matplotlib_cache"
cd "$ROOT"
```

把清单保存为 `$LOG/delivery_sha256.txt`，本文保存为`$LOG/execution_guide.md`，然后：

```bash
set +e
sha256sum -c "$LOG/delivery_sha256.txt" > "$LOG/delivery_verify.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$LOG/delivery_verify.rc"
set -e
test "$rc" = 0
while read -r digest path; do
  install -D "$ROOT/$path" "$LOG/delivery/$path"
done < "$LOG/delivery_sha256.txt"
"$PY" -c 'import sys,torch,numpy; print(sys.version); print(torch.__version__,numpy.__version__); print(torch.cuda.get_device_name(0)); assert sys.version_info[:3]==(3,8,20); assert torch.cuda.is_available()' > "$LOG/environment.log" 2>&1
test -f data/outputs/2026.08.16/08.53.26_e5_lambda0p5/checkpoints/latest.ckpt
test -f "$PROGRAM/reports/routeA_rl_env_v1/calibration/admission_manifest.json"
test -d /home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train
set +e
"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/pytest" \
  --junitxml "$LOG/tests.xml" \
  tests/dppo_uav_rebuild/engine tests/dppo_uav_rebuild/rewards \
  tests/dppo_uav_rebuild/entrypoints tests/dppo_uav_rebuild/checkpointing \
  tests/dppo_uav_rebuild/audit/test_training_audit.py tests/routea_paired tests/test_routea_trajectory_behavior.py \
  > "$LOG/tests.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$LOG/tests.rc"
set -e
test "$rc" = 0
```

预期复制核验37文件0差异、测试退出0。Windows本地CPU回归不能代替接下来真实E5/GPU执行。简单依赖或兼容问题按第5节自修权限处理；不自行升级既定计算环境。

## 2.1 训练前插入：旧BC/fresh u32/u384同状态诊断

先跑该诊断，再进入第3节。只对24个已冻结状态做比较，不重跑旧420矩阵；E5由同一评价入口自动加载。固定状态来自已有v3原迹：绕障后直连畅通、三帧纯地面且可直达、有障碍可见但直连畅通，各取成功/失败来源各3个（18个）；另含6个已复查机制案例。来源成功/失败仅说明原轨迹结局，不是对该状态所有策略的预设判决。几何不一致世界不作机制样本，正式420仍完整保留。

程序从原reset seed及实际H8动作前缀重放，核对位置/速度/yaw及三帧输入hash，不以地图推断代替真实策略输入。每状态每策略50次配对初噪，保存实际初噪hash、完整16×11动作、未来14点及H4/H8进展/首次事件；前5次采样继续最多12个H4周期（14.4秒上限），沿用原60决策的剩余期限、历史与环境随机状态。首段复用对应开环动作，后续重新观测；不执行旧tail、不延长H4。

```bash
REF=
for candidate in \
  "$PROGRAM/routea_outputs_migration_v2_1_run02/migrated_outputs/diagnostics/routea_bc_a0_r1_paired_dev_20261010_py38fix_run02" \
  "$PROGRAM/results/routea_paired_evaluation/routea_bc_a0_r1_paired_dev_20261010_py38fix_run02" \
  "$ROOT/data/outputs/diagnostics/routea_bc_a0_r1_paired_dev_20261010_py38fix_run02"; do
  if test -f "$candidate/bank/bank.json"; then REF="$candidate"; break; fi
done
test -n "$REF"
printf '%s\n' "$REF" > "$LOG/reference_path.txt"
RECOVERY_BANK="$ROOT/scripts/routea_paired_support/recovery_bank_v1.json"
BASE_DIAG="$PROGRAM/results/routea_paired_evaluation/routea_common_a1_recovery_baseline_v1_run01"
OLD="$ROOT/data/outputs/dppo_uav_rebuild/routea_r1_s2_seed202609203_fresh384_v1"
test ! -e "$BASE_DIAG"
set +e
"$PY" scripts/run_routea_paired_comparison.py \
  --checkpoint "$OLD/checkpoint_update_000032" \
  --checkpoint "$OLD/checkpoint_update_000384" \
  --reference "$REF" --out "$BASE_DIAG" --recovery-only --recovery-bank "$RECOVERY_BANK" \
  > "$LOG/recovery_baseline.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$LOG/recovery_baseline.rc"
set -e
test "$rc" = 0
(cd "$BASE_DIAG" && sha256sum -c pack_sha256_001.txt) > "$LOG/recovery_baseline_verify.log" 2>&1
"$PY" -c 'import json,sys; from pathlib import Path; r=json.load(open(Path(sys.argv[1])/"run_complete_001.json")); assert r["status"]=="complete" and r["full"]==0; c=r["recovery"]["counts"]; assert c["states"]==24 and c["policies"]==3 and c["open_loop_samples"]==3600 and c["continuations"]==360; print(c)' "$BASE_DIAG" > "$LOG/recovery_baseline_check.log" 2>&1
```

预期72份策略×状态原始记录、3600次开环采样、360条短窗续飞；`full/smoke/comparison_full=0`。初噪与策略输入须逐draw配对；加载RL后禁用LoRA的BC动作必须与本批BC一致。三帧重放或身份不符是实质故障，不能放宽容差绕过。诊断的坏表现不是训练中止条件；无合同故障便按原两臂预算继续，并及时回传此小批结果供Codex分析。

24状态是有目的的开发诊断集，不估计总体失败率。`window_end`表示短窗结束，不标为超时失败；只有原任务期限触发才是`time_limit`。地图/FOV只用于保守纯地面证据及正射线障碍可见证据，不能保证下采样后4096点含有该障碍。所有因果解释仍待同状态行为证据。

## 3. 首轮检查与256轮续跑

两臂串行运行，同一GPU不并发训练/评价。先确认下列两个新权重目录和结果目录均不存在。不要加载旧R1/fresh的actor、critic、优化器或随机状态。

```bash
for B in 1280 2560; do
  test ! -e "$ROOT/data/outputs/dppo_uav_rebuild/routea_common_b${B}_seed202609203_v1"
  test ! -e "$PROGRAM/results/routea_training/routea_common_b${B}_seed202609203_v1"
done
test ! -e "$LOG/train_all.log"
test "$(cat "$LOG/tests.rc")" = 0
test "$(cat "$LOG/recovery_baseline.rc")" = 0
```

下面后台命令对每臂先运行正式u1、核对首轮，再从同一u1续跑到256。u1是正式预算的一部分；没有丢弃式训练或额外LR重置。检查仅针对执行有效性，不用首轮reward/SR决定去留。

```bash
export ROOT PROGRAM PY LOG CONFIG_ROOT
nohup bash -c '
set -euo pipefail
trap '\''rc=$?; printf "%s\n" "$rc" > "$LOG/train_all.rc"'\'' EXIT
cd "$ROOT"
for B in 1280 2560; do
  CFG="$CONFIG_ROOT/train_e5_dppo_common_b${B}_seed202609203_v1.yaml"
  RESULT="$PROGRAM/results/routea_training/routea_common_b${B}_seed202609203_v1"
  WEIGHTS="$ROOT/data/outputs/dppo_uav_rebuild/routea_common_b${B}_seed202609203_v1"
  set +e
  "$PY" scripts/train_dppo_uav_rebuild.py --config "$CFG" --stop-after-update 1 > "$LOG/b${B}_u1.log" 2>&1
  rc=$?
  printf "%s\n" "$rc" > "$LOG/b${B}_u1.rc"
  set -e
  test "$rc" = 0
  "$PY" -c '\''import json,sys,math; from pathlib import Path; b=int(sys.argv[1]); d=json.load(open(Path(sys.argv[2])/"metrics_update_000001.json")); f=d["formal_training"]; assert d["counters"]["env_steps"]==1024; assert d["counters"]["committed_updates"]==51200//b; assert f["rejected_optimizer_steps"]==0; assert f["pair_evaluations"]==51200; assert all(math.isfinite(f["outer_parameter_delta"][k]["l2_norm"]) and f["outer_parameter_delta"][k]["l2_norm"]>0 for k in ("actor","critic")); assert d["rollout_audit"]["altitude_termination_enabled"] is False; assert d["rollout_audit"]["reward_schema_version"]=="v2"; print(json.dumps(f,indent=2))'\'' "$B" "$RESULT" > "$LOG/b${B}_u1_check.log" 2>&1
  test -f "$WEIGHTS/checkpoint_update_000001/COMMITTED.json"
  set +e
  "$PY" scripts/train_dppo_uav_rebuild.py --config "$CFG" --resume-from "$WEIGHTS/checkpoint_update_000001" > "$LOG/b${B}_u2_256.log" 2>&1
  rc=$?
  printf "%s\n" "$rc" > "$LOG/b${B}_u2_256.rc"
  set -e
  test "$rc" = 0
done
' > "$LOG/train_all.log" 2>&1 &
printf '%s\n' "$!" > "$LOG/train_all.pid"
```

首轮应有1024转移、51200 pair、40/20次有效提交、0拒绝。`formal_training.outer_parameter_delta`中actor与critic变化应为有限非零；完整数值和上下文记录回传供Codex独立核对。续跑日志应显示`checkpoint_resume`、`start_update=2`、`total_updates=256`；恢复后u2累计2048转移，提交80/40，LR按原256轮计划继续，不从u1学习率重新开始。

程序已有连续无有效更新等硬故障停止规则。若出现NaN/Inf、无有效actor更新、观测或配对合同错误、首轮计数失败，停止该次执行并保留全部现场。普通reward、loss、SR波动照既定预算完成，不筛seed、不删除困难样本、不临时改参数或以最好中间权重替代终点。

## 4. 终点数量与统一评价

`train_all.rc`生成且为0后执行（换shell时先恢复第2节变量，不再次执行“目录不存在”准备块）：

```bash
test "$(cat "$LOG/train_all.rc")" = 0
for B in 1280 2560; do
  RESULT="$PROGRAM/results/routea_training/routea_common_b${B}_seed202609203_v1"
  WEIGHTS="$ROOT/data/outputs/dppo_uav_rebuild/routea_common_b${B}_seed202609203_v1"
  "$PY" -c 'import json,sys; from pathlib import Path; b=int(sys.argv[1]); p=Path(sys.argv[2]); d=json.load(open(p/"metrics_update_000256.json")); assert len(list(p.glob("metrics_update_*.json")))==256; assert d["counters"]["env_steps"]==262144; assert d["counters"]["committed_updates"]==256*51200//b; assert abs(d["formal_training"]["actor_learning_rate"]-1e-5)<1e-12; assert len(list(Path(sys.argv[3]).glob("checkpoint_update_*/COMMITTED.json")))==34; print(json.dumps(d["counters"],indent=2))' "$B" "$RESULT" "$WEIGHTS" > "$LOG/b${B}_endpoint_check.log" 2>&1
  for checkpoint in "$WEIGHTS"/checkpoint_update_*; do
    for name in header.json SHA256.json COMMITTED.json; do
      install -D "$checkpoint/$name" "$RESULT/weight_identity/$(basename "$checkpoint")/$name"
    done
  done
done
```

两臂各256份metrics、262144条原始transition记录（每轮1024，记录嵌在metrics内）、1份r1_plan_probe和34个checkpoint；权重仅留WEIGHTS。训练计数门槛检查的是有效更新预算，若有拒绝则回传实际差异，不能改日志凑数。

评价固定u32/u128/u256，共6份RL权重。同一入口一次采集2520条RL full、24条RL smoke，另采420条当批BC full用于该入口的分支与共同基线核对，合计2940 full。此前v3全14策略矩阵不重跑。现有程序会在加载每份RL后检查禁用LoRA的BC完整动作，再走实际RL分支；新两臂只在加载权重时读取不同训练预算，评价物理协议一致。

先定位原已验任务bank，不重新采样：

```bash
REF=
for candidate in \
  "$PROGRAM/routea_outputs_migration_v2_1_run02/migrated_outputs/diagnostics/routea_bc_a0_r1_paired_dev_20261010_py38fix_run02" \
  "$PROGRAM/results/routea_paired_evaluation/routea_bc_a0_r1_paired_dev_20261010_py38fix_run02" \
  "$ROOT/data/outputs/diagnostics/routea_bc_a0_r1_paired_dev_20261010_py38fix_run02"; do
  if test -f "$candidate/bank/bank.json"; then REF="$candidate"; break; fi
done
test -n "$REF"
printf '%s\n' "$REF" > "$LOG/reference_path.txt"
EVAL="$PROGRAM/results/routea_paired_evaluation/routea_common_a1_v1_run01"
test ! -e "$EVAL"
ARGS=()
for B in 1280 2560; do
  for U in 000032 000128 000256; do
    ARGS+=(--checkpoint "$ROOT/data/outputs/dppo_uav_rebuild/routea_common_b${B}_seed202609203_v1/checkpoint_update_${U}")
  done
done
set +e
"$PY" scripts/run_routea_paired_comparison.py \
  "${ARGS[@]}" --reference "$REF" --out "$EVAL" --plots \
  --recovery-bank "$ROOT/scripts/routea_paired_support/recovery_bank_v1.json" \
  > "$LOG/evaluate.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$LOG/evaluate.rc"
set -e
test "$rc" = 0
cd "$EVAL"
sha256sum -c pack_sha256_001.txt > "$LOG/eval_return_verify.log" 2>&1
cd "$ROOT"
```

预期完成文件`run_complete_001.json`为status=complete、full=2940、smoke=24、comparison_full=2940、anchors=0、weights_included=false；`paired_index.json`为420条件×7策略引用。绘图仅呈现repeat1，不代替五次重复分析。默认名称`rl_<identity前12位>_u32/u128/u256`来自实际checkpoint；两臂身份不同，不能手改名称来混同。

此外，`recovery.counts`应为24状态×7策略、8400开环采样、840条短窗续飞；`behavior.episodes=2940`。`recovery/`保存168份策略×状态记录、24份原始观测NPZ、冻结bank、协议、BC实际分支检查与汇总；`behavior.json`复用现有分析器记录全部2940原迹的阶段行为、纯地面/障碍可见证据、H4与tail接近、跨周期tail取消和严格连续推迟。数量不得加入420主终点分母。两批合计12000采样、1200续飞，BC重复一次用于当批基线核对；独立策略共9份。

```bash
"$PY" -c 'import json,sys; from pathlib import Path; r=json.load(open(Path(sys.argv[1])/"run_complete_001.json")); c=r["recovery"]["counts"]; assert c["states"]==24 and c["policies"]==7 and c["open_loop_samples"]==8400 and c["continuations"]==840; assert r["behavior"]["episodes"]==2940; print(c)' "$EVAL" > "$LOG/recovery_a1_check.log" 2>&1
```

## 5. 回传、失败现场与范围

将两个训练结果目录、BASE_DIAG、EVAL和LOG完整回传；不要把模型state.pt和BC大权重装入回传包。训练结果内已另存全部checkpoint的header/SHA/COMMITTED身份，模型仍在ROOT/data/outputs。BASE_DIAG完成后可先回传，不必等待训练。完成后生成训练及执行资料清单：

```bash
for DIR in \
  "$PROGRAM/results/routea_training/routea_common_b1280_seed202609203_v1" \
  "$PROGRAM/results/routea_training/routea_common_b2560_seed202609203_v1" \
  "$LOG"; do
  (
    cd "$DIR"
    find . -type f ! -name return_sha256.txt -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > return_sha256.txt
    sha256sum -c return_sha256.txt > /dev/null
  )
done
```

Windows目标为 `C:/Users/cjj/Desktop/IDP3/wsl_snapshot/routea_common_a1_v1_run01/`，其下分别放`training_b1280/`、`training_b2560/`、`recovery_baseline/`、`evaluation/`、`execution/`。保留全部原始日志、退出码、配置/源码快照、metrics内原始transition/上下文/GAE/实际更新诊断、原始计划H8与执行H4数组、提交清单和图。源码/参数若改变，不能继续使用同身份的旧输出。

**简单运行问题允许WSL自行解决（用户2026-10-18授权）。** 可修正已确认的文件位置、目录权限、指定环境缺失依赖，以及不改变计算含义的小范围导入/命令兼容问题。保留失败原命令、rc、完整日志、原因、修复命令；如改源码，另存修改前后文件、diff和新hash至`$LOG/runtime_repairs/`，不覆写原交付清单或原证据。运行相关测试后可以重试；无源码/配置/权重变化的评价可用原命令追加`--resume`，新attempt日志与清单编号递增，核验最新编号，不假定仍为001。

涉及大量源码修改、原理/算法错误、reward/GAE/采样/观测/动作/H4-H8/训练预算或实验矩阵改变时暂停回传。NaN/Inf、无有效更新、hash/三帧重放/配对不一致不能靠跳过检查或放宽阈值“修好”。受身份保护的计算源码变更、训练中断后的checkpoint连续性不明，也须停下说明；不冒用原身份续跑、不丢失已完成样本。Codex收到小修补丁后同步Windows源码。上述规则替代旧版“任何错误都不许执行端处理”，不把简单故障变成额外审批。

## 6. 本轮如何裁决、如何推进节点

1. 实现有效性：核对两臂真实E5分支、采集/期限掩码、当前/下一上下文、实际空间成本、非零有限actor/critic变化、u1恢复连续性和全部有效预算；不从reward曲线直接判能力。
2. 能力：同期共同对照与A1比较总体420条件，以及既有划分中密集且真实受阻125条件的安全到达、救回/丢失配对、必要退让后真实返回和到达。任务来源标签不能代替几何受阻证据。u32/u128说明过程，u256为两臂同预算终点。
   同时比较失败行为组成及同状态反应：BC和早期能恢复、后期不能，支持已具备能力流失；仅早期RL能，优先研究RL新增能力保持，不能直接主张BC锚；三者都不能，表征、目标信息利用、状态分布与规划前缀仍是竞争解释；H4不接近而tail反复接近，支持推迟现象，但尚不能证明改H8执行会提高安全到达。concat/FiLM结构或参数范数不能单独证明根因。
3. 总体与目标能力净收益成立，即进入有限复现；A1无额外收益而共同协议有效，保留共同协议；两者均无效则依据本轮实际信用/保留证据，在A2或B中最多选择一个后继，否则转C1。不追加batch/成本系数网格，不围绕唯一根因无限验证。
4. 137/420、18/125仍为研究筛选建议，没有因代码交付变成已批准硬门槛。实用改进与统计显著性分开陈述；固定开发池与训练使用同21几何，不能据此宣称未见世界或Gazebo/PX4能力。

节点为北京时间2026-10-31结束前至少一项目标取得实质进展。已核历史fresh384指标合计约36.01h，按旧速度256轮约24h/臂只是粗估；新上下文、成本、分组和运行条件可能改变耗时。首轮及前8轮记录真实rollout/update/审计耗时后修订排程；优先在本周完成两臂和终点证据整理，留出一次有依据的修正/有限复现时间，不把整个剩余时间花在扩展验证。收益不是保证，若实测预算明显挤压节点，应及时缩减后继范围并说明依据。

新增诊断耗时目前未实测，不能承诺“几分钟”。`07_recovery.log`逐策略×状态记录累计秒数，汇总保留elapsed_seconds；先据首批数据估计后批成本。原迹行为分析仅做一次，支持同身份恢复。不得为了赶节点临时减少50次配对/5条续飞或筛掉坏样本；若实测超预算，回传耗时并由Codex调整后继范围。
