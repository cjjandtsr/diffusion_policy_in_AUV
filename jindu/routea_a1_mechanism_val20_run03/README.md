# A1 交叉续飞与改目标响应（routea_a1_mechanism_val20_run03）

2026-10-21 ~ 10-23 的 A1 机制实验：val20 run03 同状态采样、交叉续飞（对角复现 / 交叉干预）与改目标（目标响应）探针，以及后续的独立分析与评议文档。**原始证据来自 Windows `C:\Users\cjj\Desktop\IDP3\wsl_snapshot\routea_a1_mechanism_val20_run03\`（WSL 侧原件已于 2026-11-03 按数据集中政策迁删）。**

## 结构

| 路径 | 内容 |
|---|---|
| `execution/` | 执行证据 25 文件：`commands.log`、`master_run.{sh,log,pid}`、`task_plan.md`、`preflight_identity_v1_2.{py,log}`、`pytest.{log,rc}`、`same_state_{main,early}.{log,rc}`、`val_pair_check.log`、`delivery_sha256.txt`、`environment_versions.txt` 等 |
| `same_state_main/` | 固定状态采样（主）：`preflight.json`、`models.json`、`recovery/`（每状态每策略的续飞记录，`*.json`）、`pack_sha256_001.txt`、`logs/` |
| `same_state_early/` | 固定状态采样（早期检查点组），结构同上 |
| `val_u128/`、`val_u256/` | val20 两档检查点的配对回合：`paired_episodes.jsonl`、`endpoint_report.json`、`arrival_boundary_audit.json`、`evaluation_provenance.json` |
| `analysis/` | 独立分析与复算产物（见下） |

压缩约定：**仅 `*.json` / `*.jsonl` 逐文件 gzip**，其余（`*.npz`、日志、`*.rc`、`*.txt`、`*.sh`、`*.py`）原样。

```bash
find . -name '*.json.gz' -o -name '*.jsonl.gz' | xargs -n1 gunzip -k
```

## analysis/ 文件

| 文件 | 内容 |
|---|---|
| `2026-10-21_A1_val20_run03独立验收与同状态机制分析_v1_0.md` | run03 独立验收 + 同状态机制分析（证据根即本目录） |
| `2026-10-21_A1_val20到达边界阻塞复盘与修复_v1_0.md` | 到达边界阻塞的复盘与修复 |
| `2026-10-22_A1扩散机制与循环规划重新分析_v1_0.md` | 扩散机制与循环规划的重新分析 |
| `2026-10-22_A1独立技术审查复核与共同u1优势分析_v1_0.md` | 独立技术审查复核 + 共同 u1 优势 |
| `2026-10-22_A1共同u1优势分组复算_v1_0.json` | 上述分析的分组复算数据 |
| `2026-10-22_A1扩散与循环规划复查_重算数据_v1_0.json` | 扩散/循环规划复查的重算数据 |
| `2026-10-23_A1交叉续飞与目标响应独立复算_v1_0.json` | **交叉续飞与目标响应**的独立复算（1.78 MB） |
| `2026-10-23_A1交叉续飞目标响应_v1_0.png` | 交叉续飞/目标响应结果图 |
| `2026-10-23_A1交叉续飞目标响应独立分析与研究备忘录评议_v1_0.md` | 独立分析与研究备忘录评议 |
| `2026-10-23_A1实验判读与讨论材料分离_纠偏说明_v1_1.md` | 判读与讨论材料分离的纠偏说明 |

## 关键结论（摘自 10-26《iDP3-UAV：A2结果与后继方向完整交接 v1.0》§3.3，只转录不重算）

- val20 run03：BC / u128 / u256 到达 **24 / 29 / 28**，碰撞 **22 / 20 / 26**。
- 交叉续飞：**162 对角完整复现**、**108 交叉含 5 吸收**、**1386 目标响应样本**。
- 第一 H4 来源 × 后续策略四格：A = u128→u128、B = u128→u256、C = u256→u128、D = u256→u256。

判读与能力结论以 `analysis/` 内三份独立分析/评议文档为准；本 README 只做索引，不新增判断。
