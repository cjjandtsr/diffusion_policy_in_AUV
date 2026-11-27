# F12 补充 1b · 执行结果与 O2b 判据（WSL 侧，2026-11-27）

代码：`f12_phase1b_code_5512b2c3.tar.gz`（`sha256 aa536d6b08d0ac12…`）→ 应用后 `check` 0 待替换；
两个自检 `O2B_SELFTEST ok`、`O9_SELFTEST ok`。运行环境 Python 3.8.20（`idp3`），GPU RTX 4060 Laptop。

## 命令与耗时

| 步骤 | 命令 | 结果 | 耗时 |
|---|---|---|---|
| 自检 | `python scripts/f12_phase1/o2b_sustained.py --selftest` | `O2B_SELFTEST ok` | 1 s |
| 自检 | `python scripts/f12_phase1/o9_obs_distribution.py --selftest` | `O9_SELFTEST ok` | 1 s |
| 冒烟 | `o2b_sustained.py --out …/smoke --limit-states 3 --probe-repeat` | `O2B_DONE 72 probe1 True probe2 True` | 1 min 22 s |
| O2b 全量 | `o2b_sustained.py --out …/` | `O2B_DONE 4800 probe1 True probe2 None` | 1 h 06 min |
| O9 全量 | `o9_obs_distribution.py --out …/` | `O9A_DONE 5000 F<3: 1000`；`O9B_DONE 1200 {'dd_inputs_equal_o4_construction': True, 'dd_repeat_identical': True}`；`O9C_DONE 600 probe4 True` | 8 min |

说明：全量 O2b 未加 `--probe-repeat`，故 `probe2 None`（脚本 `o2b_sustained.py:209` 只在
`args.probe_repeat` 且 `n < 2` 时跑探针 2）；探针 2 已由冒烟覆盖并通过。无 `O2B_SKIP`，
200 个状态全部可行侧非 0，行数 24 × 200 = 4800。

## O2b 判据（预先写定，两条同时满足才让 RX 进第一组）

| 条件 | n | reached_1m | 到达率 |
|---|---|---|---|
| orig | 800 | 331 | 0.41375 |
| feas30_k1 | 800 | 328 | 0.41000 |
| feas30_k3 | 800 | 358 | 0.44750 |
| **feas30_k5** | 800 | 392 | **0.49000** |
| away30_k5 | 800 | 356 | 0.44500 |
| feas60_k3 | 800 | 406 | 0.50750 |

- 条件 1：`feas30_k5 − orig = +0.07625`（需 ≥ +0.10）→ **不达标**
- 条件 2：`feas30_k5 (0.49000) > away30_k5 (0.44500)` → 达标
- **结论：O2b 不达标 → P47、P48 不跑，记「RX 未进入第一组」。**

复算口径：从 `o2b_rollouts.csv` 按 `condition` 分组统计 `reached_1m`（布尔列），
4800 行全部计入，与 `o2b_summary.json` 逐项一致（另：`feas60_k3` 0.5075 最高，但不在判据内）。

## 偏离与修复（影响 1b 可运行性）

1. `scripts/f12_phase1/` 在 WSL 工作区只有手工补的 `f12_common.py`，O9 一启动即
   `ModuleNotFoundError: scripts.f12_phase1.o7_demo_reaction`。已从仓库 `main`（`git archive`）
   同步全部 10 个第一阶段脚本；`f12_common.py` 与 `main` 逐字节一致（`bd79b7b3…`），未被改写。
2. `f12_common.TRAIN_BANK` 指向 `program/results/routea_f8_diag_run03/bank.json`，该目录早已清理。
   从 Windows 快照 `Desktop/IDP3/wsl_snapshot/routea_f8_diag_run03/bank.json` 取回，
   `sha256 b4feb9e9…` 与该快照 `MANIFEST.sha256` 记录一致，只补这一个文件（未搬 178 MB 快照）。

## 回传说明

- 指引要求推到分支 `f12-pack/phase1b-run01`；按用户 2026-11-27 的仓库规则（**只保留 main**），
  本次改为推到 `main` 的 `jindu/f12_phase1b_run01/`，分支名不复用。
- 回传内容：`o2b_rollouts.csv`、`o2b_summary.json`、`o2b_probes.json`、`smoke/`（3 文件）、
  `o9a_openloop_direction.csv`、`o9b_swap.csv`、`o9c_stats.csv`、`o9c_clouds.npz`、
  `o9_probes.json`、`o2b_log.txt`、`o9_log.txt`、`MANIFEST.sha256`、本 README。
