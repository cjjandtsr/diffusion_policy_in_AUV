# Route A R1：最小 rollout 与三 seed 训练执行指导

## 元信息

- 对应设计：`sheji/2026-09-28_RouteA_RL_循环规划闭环目标与计划风险训练候选设计_v1_4.md`，审核结论 `APPROVED FOR IMPLEMENTATION`。
- 指导版本：v1.1，2026-09-29；程序版本：R1 v1.1，以下 SHA-256 表为交付身份。v1.0 留存溯源，**执行仅用本版**。
- 状态：实现审查的 1 个 P0、2 个 P1 已修正；Windows 26 项 R1 局部测试和 A0 官方 loss/逻辑累积回归通过；真实 WSL 资产的最小 rollout、实际 optimizer update 和三 seed 训练**尚未运行**。
- 分工：Codex 已写程序。用户复制交付文件到 WSL；WSL 智能体只核验、运行、回传原始结果，不改代码、系数、seed、任务或输出。

## 目标、工作目录与输入

R1 从同一 E5 BC checkpoint 按三个 A0 seed 从 update 0 开始训练。每轮策略输出仍为 `16×11`，传给 RL 环境的 policy 为 `2:10` 的 `8×11`，实际执行前 4 步；R1 给完整 8 步联合 PPO 信用，reward 只来自首次物理事件前缀和未终止时的计划尾部风险。三 seed 各 `256 updates × 1024 transitions`，S2 每 update 40 次 optimizer commit。A0 已在用的 WSL checkout 不得在训练或可能恢复期间被 R1 文件覆盖；下述 canonical `ROOT` 只在 A0 不再依赖它时使用。若 A0 仍在运行，用户应先准备完整且独立的 R1 WSL checkout，并把命令中的 `ROOT` 设为其实际绝对路径；R1 只读使用配置指定的 checkpoint、vendor、admission manifest 和 worlds/train，缺失则停下回传，不临时改配置。

Windows 源码根 `WIN_ROOT=C:\Users\cjj\Desktop\IDP3\chengxu\Improved-3D-Diffusion-Policy`；WSL 默认工作目录 `ROOT=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy`。下表每一行的**完整 Windows 绝对路径**是 `WIN_ROOT\` 加第一列并将 `/` 改为 `\`；第二列就是 WSL `ROOT/` 下须保持一致的目标相对路径。用户原样复制这 30 个完整文件，不让 WSL 端另写程序。

| Windows 根目录下相对路径＝WSL 目标相对路径 | SHA-256 |
|---|---|
| `diffusion_policy_3d/env/kinematic_uav_env.py` | `c35935e0375e036b40155db633ef9a02d17b99bf723e93fe7a2a305176ffc349` |
| `diffusion_policy_3d/env/reward_v2_support/__init__.py` | `ca33c382b86597c3bdd08462285ca4e04ab2e155c48ed2c90d5329f267819e91` |
| `diffusion_policy_3d/env/reward_v2_support/first_event.py` | `883b6609b29d9ff0777199b75b46018a361839b3907269342b6058614547c1c7` |
| `diffusion_policy_3d/env/reward_v2_support/reward.py` | `50ee00b54613545ed8fcd4a0c325db88f5e7404dcf48f488a55b5386117a2d60` |
| `diffusion_policy_3d/env/reward_v2_support/environment.py` | `5a6c3ba3a2e35bfe0e71f145203a80fddfd682aece378cf6ad3499f0628ab8ae` |
| `diffusion_policy_3d/rl/dppo_uav_rebuild/rewards/v2/__init__.py` | `6c79233fe05917510dceaad1caf39a5e3381741332a183f34627b96d019b747c` |
| `diffusion_policy_3d/rl/dppo_uav_rebuild/rewards/v2/facade.py` | `a5c6b0c79dd5673409ecd98cec782721d81b9d9b4b422108d58b94c3c7b1b3f0` |
| `diffusion_policy_3d/rl/dppo_uav_rebuild/rewards/select.py` | `8c331608653b68444f8ea7e7f98b99494be1463cd966c00549f8928bd2793aa3` |
| `diffusion_policy_3d/rl/dppo_uav_rebuild/environments/kinematic_adapter.py` | `9deffdbb38e10749596705f15f69a3f800706a3e72f019a064336c2080118ef2` |
| `diffusion_policy_3d/rl/dppo_uav_rebuild/engine/critic_state.py` | `937eb3537739d5015555d5b8dbc024363ab77539fa5801e168d33be85d7dd5cc` |
| `diffusion_policy_3d/rl/dppo_uav_rebuild/engine/value.py` | `0d9afac70776b1fc166e380fb8fa995b81447c5eb6a303c47974cdeb9f3dda6e` |
| `diffusion_policy_3d/rl/dppo_uav_rebuild/engine/initialization_parity.py` | `4e2db842454ec7369148dffaaf846e4e2cba85fbd3cefd26bb6b766b2d93ecb0` |
| `diffusion_policy_3d/rl/dppo_uav_rebuild/engine/plan_probe.py` | `f71c65d040565d1c6a63e19ea1af41d918b51ba977d4dcd09ea617cf7ba00146` |
| `diffusion_policy_3d/rl/dppo_uav_rebuild/engine/rollout_buffer.py` | `05ca74776b2d4414931c73ff4d971bce8b2429939ba4a98f5ffb38ede64d2938` |
| `diffusion_policy_3d/rl/dppo_uav_rebuild/engine/official_loss_components.py` | `43d9068d092706521202daa1ad0bd25b7778ee80d287b18649cccd3116d386ac` |
| `diffusion_policy_3d/rl/dppo_uav_rebuild/engine/trainer.py` | `c05174d731d475f8fc6db7d25db845d60b122a013ec89c911a2795d46503f19c` |
| `diffusion_policy_3d/rl/dppo_uav_rebuild/entrypoints/build_system.py` | `753f943f0dadc02ffef0f05f16d2ae334efe1eb4c40f243782bff56627608cfa` |
| `diffusion_policy_3d/rl/dppo_uav_rebuild/entrypoints/pipeline.py` | `d088773d836705bc6871727638d4595f0b9547259701e895c83cb8513143b543` |
| `diffusion_policy_3d/rl/dppo_uav_rebuild/entrypoints/config_schema_v2.py` | `b1047b4d5d5c5eafac971c64336417b35a7735d5d9819d31b063ebd526ecf91b` |
| `diffusion_policy_3d/rl/dppo_uav_rebuild/entrypoints/identity.py` | `7c3f3a68e25428db75e4117cad9d812132bf2f33c25709704250e2218976fe4b` |
| `diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_r1_s2_seed202609201_v1.yaml` | `0bf1ae8be8e6be763c4faed974def8335093017c98ae5b3974fd0ce984e6002b` |
| `diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_r1_s2_seed202609202_v1.yaml` | `a1b47dead564410de25296d0ea88bf5b0266716fa8176d0e5c2b768aed8b1b6a` |
| `diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_r1_s2_seed202609203_v1.yaml` | `4d586b453b1ecd007d26408a934ef162999fb8fcbec48fbc12a44c75c3691b50` |
| `scripts/smoke_routea_r1.py` | `736c81bc283d00bd25719dab94520b29d83c65dab84f9fc7e52f11e77e37fa09` |
| `tests/dppo_uav_rebuild/rewards/test_reward_v2.py` | `da0cb853e46cab3357366bfd1b987cf29d8e7289c0919c898814dbc2ff9622a5` |
| `tests/dppo_uav_rebuild/rewards/test_reward_v2_integration.py` | `561aee20cc1877c6c3632650d35451879c8b1fd3a264f5f48f38f61b01649969` |
| `tests/dppo_uav_rebuild/engine/test_r1_critic.py` | `7da37b4527a51eaf4a64ea0ccdce8af80dfd0c9f3d8a72c3e7f59bbbc162f563` |
| `tests/dppo_uav_rebuild/engine/test_r1_credit.py` | `425c24dd984bf755df779d7357cb8049f81f079bdcb935af4dcfc766fb1f05c7` |
| `tests/dppo_uav_rebuild/engine/test_r1_plan_probe.py` | `1abb0951b720d63613106189c23f8ebf73655bbde4a00570b88e5f54886b7cb0` |
| `tests/dppo_uav_rebuild/entrypoints/test_r1_config.py` | `89b5d013ca58c4de72dfc4ef30ca0fbc06f90cbadbfcc668121bbef1908824d2` |

输入：配置中 E5 BC `data/outputs/2026.08.16/08.53.26_e5_lambda0p5/checkpoints/latest.ckpt`、`third_party/dppo_official_cc7234a`、`/home/cjj/AUV/program/reports/routeA_rl_env_v1/calibration/admission_manifest.json`、`/home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train`，以及现有 run06 点云校准。R1 不读 v1.7 reward density/calibration；不得把 `w4/w8/w11/w12/w18` 或 fixed32 用于本轮训练、smoke 或调 reward。

## 步骤 0：复制后只读预检

以下命令在 WSL `idp3` Python 环境、R1 独立工作目录运行。`EVIDENCE` 为本轮新目录；原有输出不覆盖。先核对源码与配置全部 30 个文件，再执行 schema、测试和 smoke。WSL 不因 hash 不符而自行改文件。

```bash
ROOT=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
PY=/home/cjj/miniconda3/envs/idp3/bin/python
EVIDENCE="$ROOT/data/outputs/diagnostics/routea_r1_s2_20260929_run01"
cd "$ROOT"
test "$PWD" = "$ROOT"
test -x "$PY"
test ! -e "$EVIDENCE"
for seed in 202609201 202609202 202609203; do
  test ! -e "$ROOT/data/outputs/dppo_uav_rebuild/routea_r1_s2_seed${seed}_v1"
done
mkdir -p "$EVIDENCE/logs"
```

把上表两列复制为 `sha256sum -c` 的 30 行清单：每行为 `SHA-256<两个空格>WSL相对路径`，完整原始输出保存到 `$EVIDENCE/logs/source_and_config_sha256.txt`，同时保存真实退出码到同名 `.rc`。只有 30/30 `OK` 才继续。执行端可用下列只读 Python 命令直接解析**交付指导原文件**，无需手填 hash；用户还需把本指导文件复制到 WSL 指定路径：`$ROOT/zhidao/2026-09-29_RouteA_R1_最小回放与三seed训练执行指导_v1_1.md`。该指导文件不是程序源码。

```bash
"$PY" -c 'import hashlib,pathlib,re,sys; guide=pathlib.Path(sys.argv[1]); root=pathlib.Path(sys.argv[2]); rows=re.findall(r"^\| `(diffusion_policy_3d/[^`]+|scripts/[^`]+|tests/[^`]+)` \| `([0-9a-f]{64})` \|$",guide.read_text(encoding="utf-8"),re.M); assert len(rows)==30, len(rows); failures=[]; [(print(("OK" if (root/p).is_file() and hashlib.sha256((root/p).read_bytes()).hexdigest()==h else "FAIL"),p),failures.append(p) if not (root/p).is_file() or hashlib.sha256((root/p).read_bytes()).hexdigest()!=h else None) for p,h in rows]; print("R1_DELIVERY_HASH",len(rows),"failures",len(failures)); sys.exit(bool(failures))' \
  "$ROOT/zhidao/2026-09-29_RouteA_R1_最小回放与三seed训练执行指导_v1_1.md" "$ROOT" \
  > "$EVIDENCE/logs/source_and_config_sha256.txt" 2>&1
printf '%s\n' "$?" > "$EVIDENCE/logs/source_and_config_sha256.rc"
test "$(cat "$EVIDENCE/logs/source_and_config_sha256.rc")" -eq 0 || exit 1
```

运行生产 strict schema 和输入文件检查。点云校准 hash 必须为 `824b756e938af55be617ed7a211bfda204f1034b7fe13811febbeebf2bef8244`；不重标定。

```bash
"$PY" -c 'import sys,pathlib,json; from omegaconf import OmegaConf; from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.config_schema_v2 import validate_v2_config; cs=[OmegaConf.load(p) for p in sys.argv[1:]]; [validate_v2_config(c) for c in cs]; assert [int(c.run.seed) for c in cs]==[202609201,202609202,202609203]; assert all(str(c.reward_version)=="v2" for c in cs); assert len({str(c.output_dir) for c in cs})==3; paths=[pathlib.Path(str(v)) for c in cs for v in (c.checkpoint,c.vendor_root,c.admission_manifest,c.worlds_train)]; assert all(p.exists() for p in paths); cal=json.load(open("data/outputs/diagnostics/routea_v17_f0_g1_20260914_run06/pointcloud_calibration/pointcloud_calibration.json",encoding="utf-8")); assert all(str(c.environment.pointcloud_calibration_sha256)==cal["content_sha256"] and float(c.environment.max_no_ray_hit_fraction)==float(cal["max_no_ray_hit_fraction"]) for c in cs); print("R1_CONFIG_AND_ASSETS_PASS",len(cs),len(paths))' \
  diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_r1_s2_seed202609201_v1.yaml \
  diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_r1_s2_seed202609202_v1.yaml \
  diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_r1_s2_seed202609203_v1.yaml \
  > "$EVIDENCE/logs/config_schema.txt" 2>&1
printf '%s\n' "$?" > "$EVIDENCE/logs/config_schema.rc"
test "$(cat "$EVIDENCE/logs/config_schema.rc")" -eq 0 || exit 1
```

相对 checkpoint/vendor 路径按 `ROOT` 解析；若使用独立 checkout，先核对这些资产在该 checkout 存在。上段检查须在 `ROOT` 执行，预期 `R1_CONFIG_AND_ASSETS_PASS 3 12`。运行完整局部测试以及原 A0 schedule/buffer 回归：

```bash
"$PY" -m pytest -q \
  tests/dppo_uav_rebuild/rewards/test_reward_v2.py \
  tests/dppo_uav_rebuild/rewards/test_reward_v2_integration.py \
  tests/dppo_uav_rebuild/engine/test_r1_critic.py \
  tests/dppo_uav_rebuild/engine/test_r1_credit.py \
  tests/dppo_uav_rebuild/engine/test_r1_plan_probe.py \
  tests/dppo_uav_rebuild/entrypoints/test_r1_config.py \
  tests/dppo_uav_rebuild/entrypoints/test_formal_training_schedule.py \
  tests/dppo_uav_rebuild/engine/test_buffer_and_gae.py \
  > "$EVIDENCE/logs/local_gates.txt" 2>&1
printf '%s\n' "$?" > "$EVIDENCE/logs/local_gates.rc"
test "$(cat "$EVIDENCE/logs/local_gates.rc")" -eq 0 || exit 1
```

## 步骤 1：真实资产、BC actor 的最小 train-domain rollout

只跑 seed1 的**一个** admitted 训练域 decision，不做 optimizer update，不选择 task，不使用 fixed32。真实 pointcloud、E5 actor、H4 物理路径、R1 reward、critic 当前/下一状态都经过入口；带 `beta` 的 PPO loss 边界已由步骤 0 的回归测试覆盖。测试通过立刻进入 seed1 正式训练，首个 optimizer update 是真实 E5 的更新路径检查；若它失败，停止后续 seed 并回传，不依据 smoke reward 高低修改系数。

```bash
A0_U0=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy/data/outputs/dppo_uav_rebuild/routea_v17_a0_s2_seed202609201_v2/checkpoint_update_000000
if test -f "$A0_U0/state.pt"; then
  set -- --a0-update0 "$A0_U0"
else
  set --
  printf '%s\n' 'A0_UPDATE0_NOT_AVAILABLE: cross-run hash comparison pending' > "$EVIDENCE/logs/a0_update0_status.txt"
fi
"$PY" scripts/smoke_routea_r1.py \
  --config diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_r1_s2_seed202609201_v1.yaml "$@" \
  > "$EVIDENCE/logs/smoke_seed202609201.jsonl" 2>&1
printf '%s\n' "$?" > "$EVIDENCE/logs/smoke_seed202609201.rc"
test "$(cat "$EVIDENCE/logs/smoke_seed202609201.rc")" -eq 0 || exit 1
"$PY" -c 'import json,sys; rows=[json.loads(line) for line in open(sys.argv[1],encoding="utf-8") if line.lstrip().startswith("{")]; x=[r for r in rows if r.get("event")=="R1_SMOKE_COMPLETE"]; assert len(x)==1; r=x[0]; assert r["action_policy_shape"]==[1,8,11] and r["action_executed_shape"]==[1,4,11]; assert r["initialization_parity"]["a0_critic_parameter_sha256"]==r["initialization_parity"]["r1_critic_base_parameter_sha256"]; assert r["pointcloud_meta_history_count"]==8 and r["probe_repeat_equal_and_rng_isolated"] is True; print("R1_SMOKE_GATE_PASS",r["done_reason"],r["reward"])' "$EVIDENCE/logs/smoke_seed202609201.jsonl" \
  > "$EVIDENCE/logs/smoke_gate.txt" 2>&1
printf '%s\n' "$?" > "$EVIDENCE/logs/smoke_gate.rc"
test "$(cat "$EVIDENCE/logs/smoke_gate.rc")" -eq 0 || exit 1
```

预期两个 `.rc=0` 且日志有 `R1_SMOKE_COMPLETE` 和 `R1_SMOKE_GATE_PASS`。若 A0 update-0 checkpoint 可用，smoke 还必须输出 `a0_update0_parity.status=A0_R1_UPDATE0_EXACT`；不一致时 Python 非零并停止。若该 checkpoint 尚不可用，记录 `A0_UPDATE0_NOT_AVAILABLE`，同 seed 新建 A0 critic／R1 critic、actor 未变和 RNG 等价的构建门禁照常执行，实际跨运行 hash 对照留待 A0 update-0 回传后补验。若真实模型/点云/manifest 导致异常，停下回传完整栈，不降采样、不换任务、不改阈值。

## 步骤 2：三 seed 各自完整预算训练

单 GPU 顺序执行；`--config` 指向三份原始文件，不续接 pilot 或 A0 checkpoint。日志重定向保证 `.rc` 是 Python 真实退出码。u64/u96 仅观察，不以普通波动停止；只有 NaN/Inf、持续数值爆炸、无有效 actor update、身份/观测/配对失效等硬故障中止。

```bash
for seed in 202609201 202609202 202609203; do
  CFG="diffusion_policy_3d/rl/dppo_uav_rebuild/config/train_e5_dppo_r1_s2_seed${seed}_v1.yaml"
  "$PY" scripts/train_dppo_uav_rebuild.py --config "$CFG" \
    > "$EVIDENCE/logs/train_${seed}.log" 2>&1
  rc=$?
  printf '%s\n' "$rc" > "$EVIDENCE/logs/train_${seed}.rc"
  if [ "$rc" -ne 0 ]; then
    break
  fi
done
```

每 seed 新输出目录为 `ROOT/data/outputs/dppo_uav_rebuild/routea_r1_s2_seed${seed}_v1/`。预期 `train_*.rc=0`、`dppo_training_start` 为 `fresh_bc/start_update=1/total_updates=256/planned_transitions=262144`、update 1..256 各一条 `dppo_outer_update_complete`、256 个 `metrics_update_*.json`、33 个 checkpoint 目录（0 与 8..256）。每 update 1024 transitions、S2 40 optimizer attempts；非零 rejected、零 actor delta、probe 异常和硬停机均原样回传 Codex 判断，不自行调 KL/reward/样本。每 seed 还须有 `r1_plan_probe.json`；既定 checkpoint 的 metrics 含 `r1_plan_probe`，optimizer update 项含 `slice_diagnostics.prefix/tail/full`，transition records 含 `risk_exec/risk_tail/argmax_segment/risk_type/clearance/weight/risk_source`。

三个 seed 全部成功后生成只读验收与回传 hash 清单。权重 `state.pt` 由各 checkpoint 的 `SHA256.json` 覆盖，本清单对其余文件逐一计算，避免重复读取大型权重；任何计数不符停止并回传。

```bash
for seed in 202609201 202609202 202609203; do
  OUT="$ROOT/data/outputs/dppo_uav_rebuild/routea_r1_s2_seed${seed}_v1"
  test "$(find "$OUT" -maxdepth 1 -type f -name 'metrics_update_*.json' | wc -l)" -eq 256 || exit 1
  test "$(find "$OUT" -maxdepth 1 -type d -name 'checkpoint_update_*' | wc -l)" -eq 33 || exit 1
  test -f "$OUT/r1_plan_probe.json" || exit 1
  find "$OUT" -type f ! -name state.pt -print0 | sort -z | xargs -0 sha256sum \
    > "$EVIDENCE/logs/output_${seed}_sha256.txt"
done
```

失败：停止后续 seed，保留现场；不删除、不覆盖、不改 seed 后重跑。若需要 checkpoint 恢复，先把失败日志、退出码、最近 checkpoint/header、运行身份和 probe 文件回传 Codex，由 Codex 核对后给明确恢复命令。

## 回传清单与解释边界

回传本指导原文件、30 文件 hash 日志和 `.rc`、三份 YAML 原件、schema/test/smoke 日志和 `.rc`、每 seed 的完整训练日志与 `.rc`、全部 `metrics_update_*.json`、`r1_plan_probe.json`、checkpoint 0/32/64/96/128/192/256 的 header 与提交/hash 元数据、输出逐文件 hash manifest。checkpoint 权重先留 WSL，后续若需部署/评估再指定回传。首个可判读趋势来自**固定配对 dev 的多 seed 安全到达**；训练日志和单 seed 中间点不是能力结论。fixed32 仅作为冻结历史回归基准，不能声称新独立最终测试集；R1 胜只能支持组合设计，不能拆解为单项根因。这里的 rollout 属 lightweight/kinematic，不是 PX4/Gazebo 或实物飞行证据。
