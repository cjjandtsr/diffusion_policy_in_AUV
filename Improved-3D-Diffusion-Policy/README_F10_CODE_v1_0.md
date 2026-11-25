# F10 评价入口代码交付 v1_0

日期：2026-11-25。分支：`f10-code-20261125`，基于 `f9-code-a2-20261124`（`9801780`）。
只新增评价相关文件，未改动任何既有源码、权重、银行或配置。

## 1. 改动文件

| 文件 | 作用 | SHA256 |
|---|---|---|
| `scripts/run_routea_f10_eval.py` | F10 评价入口：`--policy {P6,BC}` × `--guidance {none,g1,G0}` | `f54adbbe91da771b5ea0e236016b2aa7b75af2fa5d968f710f9420fa683f908b` |
| `scripts/f10_chain_probe.py` | T2：同一条件下 guidance off/on 的两条 DDIM 链逐步比较 | `b525f33a17e74cf4ca833c1900608376cbfb19d9b08df652ce89f41244fe6d39` |
| `scripts/f10_compare.py` | T1/T4/T5：两个 job 的逐决策输出逐位比较 | `fb3e7e66f85e533aced9ce0925d54745f0baeffe921c46d6d0927dabbef28a19` |
| `README_F10_CODE_v1_0.md` | 本文件 | — |

## 2. 入口

```
python scripts/run_routea_f10_eval.py prepare \
  --output RUN --weights-verified <weights_verified.json> \
  --dev-bank <dev420 bank> --val-bank <val20 bank> \
  --source-root <native_a2> --checkpoint <P6 ckpt> --repo <MAIN>

python scripts/run_routea_f10_eval.py worker \
  --output RUN --policy P6|BC --guidance none|g1|G0 --split dev|val \
  [--eta F] [--consistency INT] [--tag NAME] [--limit N] [--full-run-approved]

python scripts/run_routea_f10_eval.py collect --output RUN --job RUN/probes/X --split dev
python scripts/run_routea_f10_eval.py rejudge --job JOB --bank BANK [--write OUT.json]
```

- `prepare` 把 `evaluation_sources`（16 个文件的 SHA256）、两个银行的 SHA256/content_hash、
  P6 回执（`state.pt` `a91c9f42…`、update 256、runtime identity `d987235a…`）写进 `request.json`；
  之后任何受控文件变动都会让 `load_plan()` 直接失败。
- `worker` 的 `--limit` 走 `RUN/probes/`，全量走 `RUN/runs/` 且必须 `--full-run-approved`。
- g1 = `d_safe 0.6, eta 0.4, M 4, consistency 0`；G0 = `d_safe 0.6, eta 0.2, M 2, consistency 0`；
  none = #E2。`--eta/--consistency` 只用于 T1（eta=0）这类探针。

## 3. 与既有实现的衔接

| 项 | 做法 |
|---|---|
| 评价入口基线 | `scripts/run_routea_e3g_probe.py`（G0 探针入口）的 runtime 恢复、overlay、bank 映射、`record_episode` 调用方式 |
| 引导 | 直接 `from scripts.routea_denoising_guidance import Guidance, install`，不复制、不改写；`wrapper_sha256 = 8fa3b263…` 写进 `identity.json` |
| 注入点 | `p_mean_var` 中 x0 clip 之前的 `_probe_guidance.update(x_recon, index)`；`index >= 7` 才可能改动 |
| 分支 | `P6: use_base_policy=False`（LoRA），`BC: use_base_policy=True`（基础）；两者共用同一 `official_runtime`，注入点相同 |
| 1 m 判定 | `scripts.f7_analysis.outcome_1m`（论文 arrival_1m：raw reached 或连续执行路径最近点 < 1 m），未新写判定函数 |
| 执行段修正量 | `execution_delta` 与 `f9_online.py::execution_delta` 逐字相同；`delta_matches_f9()` 在每次 `load_plan()` 时用 AST 抽取上游函数源文本比对，不一致直接失败 |
| 原迹 | `scripts.routea_paired_support.episodes.record_episode`，格式与 F5 评价一致 |

## 4. 每个决策新记的字段（`probe_metrics`）

| 字段 | 含义 |
|---|---|
| `guided` | `delta_exec > 0`（有实际修正） |
| `delta_exec` | 基线链与引导链前 4 个执行航点的世界系最大偏差（F9 定义） |
| `triggered_denoise_steps` | 代价非零、发生更新的去噪步（7–9） |
| `update_count` | 更新次数（M 倍触发步数） |
| `denoise_seconds` | `forward` 调用前后 `torch.cuda.synchronize()` 的 DDIM 环耗时 |
| `inference_seconds` | `predict()` 全程耗时（encode + 去噪 + unnormalize） |
| `cloud_points` | 引导代价实际使用的点数（`observation.point_cloud[-1]` 转到世界系后 z ≥ 0.1） |

基线链只在 `triggered_denoise_steps` 非空时补跑一次（同一个 `action_seed` 重新 seed，
RNG 流一致，不扰动引导链）；未触发的决策 `delta_exec` 严格为 0.0。

## 5. 复现命令（探针）

```
python scripts/run_routea_f10_eval.py worker --output RUN --policy P6 --guidance none --split dev --limit 10 --tag T1_none
python scripts/run_routea_f10_eval.py worker --output RUN --policy P6 --guidance g1 --eta 0 --split dev --limit 10 --tag T1_g1_eta0
python scripts/run_routea_f10_eval.py worker --output RUN --policy BC --guidance none --split dev --limit 10 --tag T4_BC_none
python scripts/run_routea_f10_eval.py worker --output RUN --policy P6 --guidance G0 --split dev --limit 10 --tag T5_P6_G0
python scripts/run_routea_f10_eval.py worker --output RUN --policy P6 --guidance g1 --split dev --limit 10 --tag T6_P6_g1
python scripts/f10_chain_probe.py --output RUN --split dev --condition 3 --guidance g1 --decisions 60 --write RUN/t2_chain.json
python scripts/f10_compare.py --new RUN/probes/T1_g1_eta0 --ref RUN/probes/T1_none --label T1 --write RUN/comparisons/T1.json
python scripts/run_routea_f10_eval.py rejudge --job <G0_dev> --bank <dev420 bank> --write RUN/t7_rejudge.json
```

## 6. 已知实现细节（不是偏离）

1. 触发标记用 `delta_exec > 0` 而不是“梯度非零”。零梯度时 `eta·g` 为 0，输出不变，
   此时 `delta_exec = 0`，与“未触发”对结局完全等价；`triggered_denoise_steps` 与 `update_count`
   同时保留，供复核。
2. `denoise_seconds` 是 `forward` 的墙钟时间，`inference_seconds` 额外含 encode/verify/unnormalize；
   两者都逐决策记录。
3. 基线链补跑会使触发决策多一次采样，因此**同样条件下引导臂的耗时不能直接与 none 臂比**；
   T6 只比较同一臂内 `guided=True/False` 决策的去噪耗时。
4. `prepare` 用 `routea_f8_diag_run03/weights_verified.json` 的 P6 回执；重跑前必须换新
   `--output`（`write_json` 拒绝覆盖）。
