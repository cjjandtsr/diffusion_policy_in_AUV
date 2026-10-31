# P6@u256 #E2：dev420执行指导 v1.0

日期：2026-10-31。用户已确认：dev420轻量运动学闭环。策略编号沿用用户确认的《iDP3-UAV RL 后训练：策略编号规范》，本文不另行定义编号。历史目录和程序文件名保持原样。

## 目标与验收

只新增 P6@u256 #E2 的420条评价：原冻结21世界、84任务×5重复。原生身份加载成功后，显式覆盖评价clip=2.0、11维sampling/logprob floor=0.1，确定性DDIM，沿用冻结bank与现有动作种子实现。确定性采样仍有固定初噪；不得依据规范中的种子简写另造种子函数。

回传后主比较为 P6@u256 #E2 对 P6@u256 #E1；辅助比较为同一 #E2 下的 P0、P5@u256。报告安全到达、碰撞、超时、配对救回/丢失、世界簇区间及累计reward；这次是开发集补充，不采用上一组合实验的绝对计数门槛，不据结果宣称未见世界泛化。没有新增训练运行，不登记P8；不运行test12，不重新训练或重新挑选检查点。

执行计划：
1. 核对入口、冻结bank和权重 → 验证：hash、身份、u256、420条件。
2. 用统一worker运行 → 验证：rc=0、420份commit、实际RL分支、#E2协议。
3. 重读原迹并封包 → 验证：首次事件/连续路径/summary一致、回传SHA256全通过。

## 交付与环境

**计算源码零修改；复用已交付文件。** Windows入口：
`C:\Users\cjj\Desktop\IDP3\chengxu\Improved-3D-Diffusion-Policy\scripts\run_routea_xk_evaluation.py`

WSL相对路径：`scripts/run_routea_xk_evaluation.py`；文件SHA256：
`15bca3863d426f881236c2dcca76bcab7cc5fa6ed3d9011e27ebf8f05d3d39e7`。

工作目录：`/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy`。
环境：已有idp3、Python3.8.20、GPU；不升级依赖，不覆盖原生源码快照。WSL现有入口hash匹配时无须再复制源码。可将本指导随执行记录保存至结果目录。

输入：
- 权重：`data/outputs/dppo_uav_rebuild/routea_a2_selective_keep_run01/T/checkpoint_update_000256/`。
- 原生源码快照：`/home/cjj/AUV/program/results/routea_test12_independent_run02/native_a2/`。仅用于恢复身份，不读取该次评价结果。
- 原dev bank：下列命令中的BANK，必须已存在；不允许现场重建。若目录已迁移，只调整BANK绝对路径，内容hash必须完全相同。

预先绑定的P6@u256身份为 `d987235a0cbb5ad395bbd6e7979bdb8d7e0cff1303b0611cb462c7d554bbd61f`，权重SHA256为 `a91c9f428c5749b8ca507168ea83f17eeb88cce4b4a03c6aeaea197b214bdd43`。这些值已与本地既有加载记录核对；本地没有重新加载WSL权重。

## 完整执行命令

在WSL Bash中整体执行。新输出目录必须不存在。`--split rl_train`不生成新bank；前检已要求使用原dev bank。`--worlds`不会覆盖bank内绑定的几何。

```bash
set -euo pipefail
ROOT=/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy
PY=/home/cjj/miniconda3/envs/idp3/bin/python
NATIVE=/home/cjj/AUV/program/results/routea_test12_independent_run02/native_a2
CKPT="$ROOT/data/outputs/dppo_uav_rebuild/routea_a2_selective_keep_run01/T/checkpoint_update_000256"
BANK="$ROOT/data/outputs/diagnostics/routea_bc_a0_r1_paired_dev_20261010_py38fix_run02/bank/bank.json"
OUT=/home/cjj/AUV/program/results/routea_p6_u256_e2_dev420_run01
cd "$ROOT"
test ! -e "$OUT"
mkdir -p "$OUT/execution"
run_logged() {
  local name="$1"; shift
  printf '%q ' "$@" > "$OUT/execution/$name.command.txt"
  printf '\n' >> "$OUT/execution/$name.command.txt"
  set +e
  "$@" 2>&1 | tee "$OUT/execution/$name.log"
  local rc=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$rc" > "$OUT/execution/$name.rc"
  return "$rc"
}
run_logged 00_preflight "$PY" - "$ROOT" "$NATIVE" "$CKPT" "$BANK" "$OUT" <<'PY'
... (见实执行脚本 / 指导原文)
PY
run_logged 01_evaluate "$PY" scripts/run_routea_xk_evaluation.py worker \
  --repo "$ROOT" --source-root "$NATIVE" --current-root "$ROOT" \
  --checkpoint "$CKPT" --expected-update 256 --policy P6 \
  --bank "$BANK" --split rl_train \
  --worlds /home/cjj/AUV/PX4-Autopilot/Tools/sitl_gazebo/worlds/train \
  --output "$OUT/full/P6" --repeats 5 --clip 2.0 \
  --min-std 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
run_logged 02_verify "$PY" - "$OUT" <<'PY'
... (见实执行脚本 / 指导原文)
PY
(
  cd "$OUT"
  find . -type f ! -name return_sha256.txt ! -name return_verify.log -print0 \
    | sort -z | xargs -0 sha256sum > return_sha256.txt
  sha256sum -c return_sha256.txt > return_verify.log
)
```

## 失败处理与回传

预期三步rc均0；实际生成420份commit及对应完整原迹，最终出现 `P6_U256_E2_420_RAW_PASS`。不预设到达数，不设置为了通过而更换seed或减分母的规则。现有worker从程序入口即关闭JIT优化，覆盖原生加载和渲染；不得移除这一范围。

任何身份、初始观测hash、物理事件、协议或数量检查失败，立即停步，保留原命令/log/rc和已生成原迹回传。不要原目录盲目重跑，不改hash或旧快照，不换权重，不调用旧的硬编码双策略分析子命令处理单策略目录。简单路径迁移可按既有授权处理，但必须保持同一bank/权重hash并记录路径调整。

回传整个 `/home/cjj/AUV/program/results/routea_p6_u256_e2_dev420_run01/`，Windows目标 `C:\Users\cjj\Desktop\IDP3\wsl_snapshot\routea_p6_u256_e2_dev420_run01\`。包括plan、bank JSON、三步command/log/rc、policy_provenance、全部trace/commit、summary、verification、SHA256清单与校验日志。`bank.json`为冻结JSON副本；其引用的初始NPZ仍保留在原bank目录，worker按冻结观测hash检查新渲染，不从该副本旁加载NPZ。模型权重和原生源码快照留在既有位置，不复制进本次结果包。

## 本地交付状态

本指导复用既有统一worker；本地核对入口SHA256、P6@u256历史加载身份及CLI兼容性，验证Bash与内嵌Python语法。真实权重加载、GPU评价及420回合结果待用户/WSL执行，不能写成实验已启动或已完成。

---

## 执行记录（WSL 侧，2026-10-31）

- 执行脚本：`/home/cjj/AUV/temp/p6_u256_e2_dev420/run_guide_v1_0.sh`（逐字复制上述命令块）
- 启动：16:33 CST，`setsid --wait` 分离会话；结束：17:25 CST，总时长约 52 分钟
- 三步 rc：00_preflight=0，01_evaluate=0，02_verify=0
- 结果：420 回合；outcomes = reached 203 / collision 143 / time_limit 74；`P6_U256_E2_420_RAW_PASS`
- 本副本在本文件加入后重新生成 `return_sha256.txt` / `return_verify.log`，以保证清单覆盖整个包
