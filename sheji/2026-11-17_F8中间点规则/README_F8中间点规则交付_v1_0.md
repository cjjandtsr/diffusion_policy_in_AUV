# F8 中间点规则与文件交付 v1_0

| 版本 | 日期 | 内容 |
|---|---|---|
| v1_0 | 2026-11-17 | 首版。F8 方案 v1_3 引用的中间点规则与文件原先只在论文仓库 `paper-v2-restructure`（`研究报告/analysis_20261117_f8probe/`、`analysis_20261116_f8pre/`、`analysis_20261116_f7/`）；Codex 本地论文克隆未同步、SSH fetch 失败，看不到。本目录把它们原样放进项目仓库，供 Codex 直接使用 |

| 项 | 值 |
|---|---|
| 项目仓库 | 分支 `claude/2026-11-17-f8-subgoal-rule`，基于 main `6efc8c42` |
| 论文仓库来源 | `paper-v2-restructure` `61a2adc3f2a85b5ddd29a7391c2e1d8c19671506` |
| 依据 | 论文仓库 `研究报告/2026-11-17_F8执行方案_引导绕行重标注注入保持库_v1_4.md` |

| 内容 | GitHub | Windows | WSL |
|---|---|---|---|
| 本目录 | 项目仓库 `sheji/2026-11-17_F8中间点规则/` | `C:\Users\cjj\Desktop\IDP3\sheji\2026-11-17_F8中间点规则\` | 按需复制 |
| dev420 P6 原迹（脚本输入） | 项目仓库 `jindu/routea_f5_run01/results/eval_dev/full/P6/traces/`（main 为 `*.jsonl`；整理分支为精简 `*.jsonl.gz`） | `C:\Users\cjj\Desktop\IDP3\jindu\routea_f5_run01\results\eval_dev\full\P6\traces\` | — |

## 文件（SHA256，与论文仓库同名文件逐字节相同）
```
32cbc908b61c899a9ad2d55978857de3eb6dfa752182411be7dcf7b12bd2ab3d  blocked_tasks_dev420.json
3bfbbd3305d430e8a227e1315daa71d77244fd376e0884b442e30ec714e19760  geo_field.py
562356385db6eab508169fd7355f179dbf57d08053ef51869c81079aa3bfafab  pre2_rows.json
e0cd667265ad9e6dab8715ce5416ad872c471c6a7a3f39bd4e8c927b4f7ce101  subgoal_feasibility.py
3c635f21f5346855ae5b5a2eb2adfb71617180c1d6f72caa6ea241695d70d226  subgoal_feasibility_output.txt
e8ac5801de37dd37b971cb109cd636f7f9b2600a00f5f0ae20123eff53fef8d2  subgoals_dev420_blocked.json
```

| 文件 | 用途 |
|---|---|
| `geo_field.py` | 测地距离场：0.1 m 实体栅格 → 欧氏距离变换得净空 → 每 3 格取 1（0.3 m 粗栅格），净空 ≥0.3 m 为可通行，26 邻接 Dijkstra；`line_clear(W, a, b)` 返回线段最小净空。依赖 numpy、scipy |
| `subgoal_feasibility.py` | 为 dev420 的 33 个直线被挡任务生成中间点（探针用） |
| `subgoals_dev420_blocked.json` | 上一行的输出：21 个任务的中间点（世界坐标，按经过顺序） |
| `subgoal_feasibility_output.txt` | 上一行的逐任务输出（14 个任务 1 个中间点、7 个 2 个、12 个无法拆分） |
| `blocked_tasks_dev420.json` | dev420 直线被挡的 33 个任务 |
| `pre2_rows.json` | 33 个任务的越顶/侧绕标签（键 `A` 下每项的 `over`），F8 分层判据用 |

## 中间点规则（F8 第 1 步对训练任务按同一规则实现）
1. 区域：`lo = min(起点, 目标) − 6 m`，`hi = max(起点, 目标) + 6 m`，竖直范围取任务的 `z_range`；`W = World(geometry, lo, hi)`。
2. 路线：`path = W.field(目标).path(起点, 1e9)`，即净空 ≥0.3 m 的测地最短路线上的点序列。
3. "可通过" = `line_clear(W, a, b) ≥ 0.3`。
4. 1 个中间点：路线上所有满足"起点→w 可通过且 w→目标可通过"的点中，取 `|w−起点| + |目标−w|` 最小者。
5. 没有 1 个中间点的解时，2 个中间点：在路线上每隔 1 个点取点（下标步长 2），取满足三段都可通过、总路程最短的一对 (a, b)，a 在 b 之前。
6. 2 个也不行：该任务不入候选，计数报告。

## 复现命令（只读；脚本会把 JSON 写到自己所在目录，请先复制到临时目录再运行）
```bash
mkdir -p /tmp/sg && cp subgoal_feasibility.py /tmp/sg/
python3 -I /tmp/sg/subgoal_feasibility.py <项目仓库根目录> blocked_tasks_dev420.json <本目录绝对路径>
sha256sum /tmp/sg/subgoals_dev420_blocked.json   # 应为 e8ac5801de37dd37b971cb109cd636f7f9b2600a00f5f0ae20123eff53fef8d2
```
Claude 已在整理分支的精简 `*.jsonl.gz` 原迹上重跑，JSON 与逐任务输出都与上表逐字节一致（约 47 s，CPU）。
