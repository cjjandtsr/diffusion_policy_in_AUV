# C侧 2026-11-20：论文 186 任务各方法原迹（只读导出）

执行依据：`执行指引/C侧_2026-11-20_论文186任务原迹导出_v1_0.md`
用途：供 `研究报告/2026-11-20_被挡任务偏离时机与前方自由距离_EGO对BC与P6_v1_0.md` §五 的同任务配对。
性质：只读取件。未运行训练/评价，未改任何源码或配置。

## 源
`/home/cjj/AUV/program/results/paper_186_v1_0_run01`（3.3 GB 整目录；本次只取各 `traces/`）

## 产物
| 文件 | 内容 |
|---|---|
| `tree.txt` | `maxdepth 4` 目录结构（286 行） |
| `trace_dirs.txt` | 全部 29 个 `traces/` 目录 |
| `trace_counts.txt` | 每个 `traces/` 目录的文件数 |
| `trace_heads.txt` | 每个 `traces/` 目录首文件的前 3000 字节（用于确认方法身份） |
| `tar_members.txt` | 实际打包的成员清单（26 个目录） |
| `parts/paper186_traces_non_ego.tar.gz0{0,1,2}.part` | 4796 个 `.jsonl` 的 tar.gz 拆包，3 段（54/54/52 MB），合计 167,947,912 B |
| `SHA256SUMS` | 上述文件的 SHA256 |
| `commands_rc.log` | 各步骤 rc、包大小、包内文件数 |

## 入包（26 个 traces 目录，4796 文件）
| 目录 | 文件数 |
|---|---|
| `fast/full/fast/traces` | 933 |
| `px4/full/px4/traces` | 932 |
| `px4_official/full/px4/traces` | 930 |
| `fast_official/full/fast/traces` | 930 |
| `px4_official_fc/full/px4/traces` | 341 |
| `invalid_*`（4 个：identity_break / source_change / nonfinite_reference v1,v2 / waypoint_dt） | 253 / 31 / 36 / 107 / 243 |
| `diagnostics/*`（9 个 probe/smoke） | 2–4 |
| `smoke_dev_*`（6 个） | 4 |

## 未入包
| 目录 | 文件数 | 原因 |
|---|---|---|
| `ego/full/ego/traces` | 932 | 已单独交付（`ego_full_traces_932.tar.gz`） |
| `aborted_ego_nonpersistent_transport_v1/.../ego/traces` | 20 | EGO 家族，非论文对比臂 |
| `smoke_dev_ego/full/ego/traces` | 4 | EGO 冒烟 |

`invalid_*` 与 `diagnostics/*` 按指引「除 EGO 外全部」保留在包内；按目录名即可剔除，配对时只取 `fast`、`fast_official`、`px4`、`px4_official`（`px4_official_fc` 为 341 文件的部分配置）。

## 核对
- 包内文件数 4796 = 26 个目录 `ls` 文件数之和（逐目录比对一致）。
- 归档解压命令：`tar -xzf paper186_traces_non_ego.tar.gz`（成员为相对路径 `./<run>/full/<method>/traces`）。

## 归档拆包说明

GitHub 单文件上限 100 MB，原始压缩包 160 MB 无法直接入库，故按 54 MB 切成 `parts/` 三段。
解压前先拼回并核对 SHA256（应为 `630f706747baf93b5cef271a9a4ab43452ce9ee411b6e3dfa8d1fab20f11ff28`）：

```bash
cat parts/paper186_traces_non_ego.tar.gz0*.part > paper186_traces_non_ego.tar.gz
sha256sum -c <(echo "630f706747baf93b5cef271a9a4ab43452ce9ee411b6e3dfa8d1fab20f11ff28  paper186_traces_non_ego.tar.gz")
tar -xzf paper186_traces_non_ego.tar.gz
```

本地原件保留在 `program/jindu/C侧_20261120_paper186_export_full.tar.gz`（不进仓库）。
