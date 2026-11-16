# F7 run01 —— Codex 判读材料包

F7 训练轮（`approach_cbf` 提前接近速率项 w=2 + 自适应学习率）四臂 u320 训练 + 共同 #E2（dev420 + val80）评价的原始材料。
**所有 `.json` / `.jsonl` 均逐文件 gzip 压缩**（`.json.gz` / `.jsonl.gz`），内容与原始文件逐字节等价；末尾附解压命令。

---

## 1. 目录

```
jindu/routea_f7_run01/
├── README.md                          ← 本文件
├── MANIFEST.sha256                    ← 包内全部文件的 sha256（相对本目录）
├── MANIFEST_transfer_package.sha256   ← Windows 回传包原始清单（4813 行）
├── configs/                           P16/P17/P18/P19.yaml + prepared.json + calibration.json
├── banks/                             dev_bank_863bf1fb.json.gz / val_bank_a2b28e0c.json.gz
├── acceptance/                        Codex 验收输出（audit / timeout_geometry / inputs_and_status / files.sha256 / resume_fix_probe / verification）
├── eval_dev/<P>/                      dev420 原迹
│   ├── traces/*.jsonl.gz              420 个逐回合原迹（一文件一回合）
│   ├── commits/*.json.gz              420 个逐回合 commit（sha256 链 + summary）
│   ├── summary.json.gz
│   └── policy_provenance.json.gz
├── eval_val/<P>/                      val80 原迹（80 + 80 + summary + provenance）
└── train/<P>/                         u257–u320 训练逐轮证据
    ├── metrics_update_000257..000320.json.gz   64 份/臂（含 1024 条逐转移记录）
    └── f7_monitor_000257..000320.json.gz       64 份/臂
```

规模：2000 条评价原迹 + 2000 份 commit + 4 臂 × 64 轮 × 2 类训练文件 + 配置/bank/验收，gzip 后合计约 **578 MB**。

---

## 2. 评价原迹字段（`eval_dev|eval_val/<P>/traces/*.jsonl`）

每文件三部分，按行序：

1. **首行 = 回合头**（`type: "episode_start"`）
   `task_id`, `world_id`, `role`, `repeat`, `reset_seed`, `spawn_offset`, `bank_hash`, `initial_hash`,
   **`geometry`**（该回合全部障碍：`name / shape / x,y,z / radius / size_x,y,z / height / yaw`）、
   `task`（`start_pos / start_yaw / goal_pos / z_range / disturb_enable / world_name / episode_id`）、
   `obstacle_sha256`, `policy`, `policy_identity_hash`, `reward_version`, `use_base_policy`, `action_dtype` 等。
2. **中间行 = 逐决策 transition**
   `decision`, **`pre_state`**, `post_state`, **`executed_path`**（本决策执行的四段路径点 ×3D）,
   **`planned_policy_path`**（策略原始规划路径）, `executed_action_4x11`, `full_action_16x11`, `policy_action_8x11`,
   `candidate_h4_path`, `executed_time_s`, `event_segment`, `event_fraction`, `done_reason`,
   `reward`, **`reward_terms`**, `action_seed`, `observation_provenance`, `pointcloud_metadata_before/after`。
3. **末行 = `episode_end`**（终止事实与 summary 对齐）。

`commits/*.json` 为同一回合的 commit 记录：`sha256`、`summary`（`bank_hash / decisions / done_reason / initial_hash / physical_time_s / policy / repeat / role / safe_arrival / status / task_id / ...`），用于与 `summary.json` 交叉校验。

## 3. 训练逐转移字段（`train/<P>/metrics_update_<u>.json`）

每份含 20 个 outer update × 每 update 1024 条 `rollout_audit.transition_records`（合计 20,480 条/文件），
每条含：

- 决策路径：`executed_path`、`f7_audit.planned_path`、`f7_audit.segment_durations`
- **`f7_audit.approach_cbf_diagnostic`**（本转移的 approach_cbf 值，w=1 计量，乘 w=2 即奖励项）
- 位置/目标：`position_before_xyz / position_after_xyz`、`goal_distance_before_3d / after`、`net_goal_approach_3d`
- 奖励：`reward`、`reward_terms`、`reward_metrics`
- 学习信号：`advantage_raw / advantage_normalized_outer_update`、`value`、`next_value`、`return_target`、`dppo_credit`、`bootstrap_allowed`、`gae_continuation`
- 身份：`update_index`、`episode_id`、`transition_id`、`task_id`、`world_id`、`role`、`config_hash`、`runtime_identity_hash`、`checkpoint_identity_hash`

### 「提前绕行 vs 减速」怎么算

`approach_cbf` 的定义（方案 v1.3，`f7_analysis.py:43-46`）：

```
approach_cbf = -w · Σ_seg Δt · min(2, max(0, -( (c_{i+1}-c_i)/Δt + (c_{i+1}+c_i-0.6)/6 )))
```

其中 `c_i` 为决策点对障碍的点净距序列（由 `executed_path` 与回合头 `geometry` 现算）。
判别思路（用本包即可复算，无需再跑评价）：

- 用 `f7_audit.planned_path` 与 `executed_path` 的横向偏离 → 是否**提前绕行**；
- 用相邻 transition 的 `executed_time_s` / `position_*` → 是否**减速**；
- `approach_cbf_diagnostic≠0` 的转移占比、按 `role` / `world_id` 分层统计即可。

## 4. 身份与 provenance

| 项 | 值 |
|---|---|
| 父 | P6@u256，state.pt `a91c9f428c5749b8ca507168ea83f17eeb88cce4b4a03c6aeaea197b214bdd43`，runtime `d987235a…bbd61f` |
| u320 state.pt | P16 `93735e1f…c80466` / P17 `d0b8bf6d…120bd2` / P18 `aa5152cd…2eb6c7` / P19 `53646f62…b4543fd` |
| dev bank | `863bf1fb2a41c991b73cc92493aa7d8732c711b4e48c6a8d063bd2dd8c59996a`（21 世界 / 84 任务 / 420 回合） |
| val bank | `a2b28e0cf2ca32ee8a987d0f1a03f94d2783dea57fee1bd096fdb8e5ca542017`（20 世界 / 80 任务） |
| strata | dev `ecd485be4d3c3356a66865a11c487d90e0b49613cd43eee041625a1aee549b06` |
| 评价协议 | clip 2.0、确定性、LoRA 分支（`actual_use_base_policy=false`）、min-std 11×0.1 |
| bootstrap | 20000 次，seed `2026102601` |
| keep | `0.01061006086489362`，`keep_seed=2026102301` |

## 5. 结果（只报数字）

dev420（1 m 主判据，840 配对）：reached P16 197 / P17 205 / P18 266 / P19 267（P6 281）；
`arrival=−0.15595`，95% CI [−0.22976, −0.08452]，两 seed delta −0.16429 / −0.14762；`collision=−0.00357`；
**`verdict=本预算下不符合采纳标准`**。

val80（仅描述，160 配对）：reached 36 / 42 / 58 / 52（P6 61）；`arrival=−0.20`，CI [−0.2875, −0.1125]；`collision=+0.03125`。

判读归 Codex；本包不含权重（权重留 WSL，仅回传 header/SHA256/COMMITTED，见 Windows 回传包 `weights_headers/`）。

## 6. 解压与校验

```bash
# 解压（保留 .gz）
find jindu/routea_f7_run01 -name '*.gz' -print0 | xargs -0 -P 8 -n 1 gunzip -k

# 校验（在仓库根目录执行；MANIFEST 内路径为仓库根相对）
sha256sum -c jindu/routea_f7_run01/MANIFEST.sha256 --quiet && echo ALL_OK
```

## 7. 重要背景

- P17 在 2026-11-11 会话中断时停在 u283（未提交）；2026-11-16 从 COMMITTED u280 恢复。
  第一次恢复因 `f7_monitor_000281.json` 互斥写冲突失败（`pipeline._UPDATE_ARTIFACT_PATTERNS` 未含 `f7_monitor_*`），
  把三个遗留 monitor 移入手工 quarantine（不改源码、不覆盖原件）后第二次恢复成功，4h20m22s 到 u320。
  细节见交接文档与 Windows 回传包 `execution/P17_resume_failure_and_fix_20261116.md`。
- 四臂串行链日志缺 `ARM_DONE P17`（该链被杀，P17 手工补齐）；验收依据为 `logs/P17_u320_20261116_105753.status`（EXIT_CODE=0）。
