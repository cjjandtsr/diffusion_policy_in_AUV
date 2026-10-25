# runtime repair 2: 恢复诊断启动器 v1 立即退出（rc=1，无 python 运行）

- 症状: nohup bash -c 启动 5s 内 Exit 1；recovery_baseline.log 未创建。
- 根因: 启动器 trap 行写成 `trap '...' "$1" EXIT`，$1(=ROOT 路径) 被当作 signal spec →
  "invalid signal specification"；trap 注册失败 → set -e 退出。
- 修复: 改用指导 §3 同款 trap 形式（trap 'rc=$?; printf ... > "$LOG/...rc"' EXIT，路径硬编码），
  变量经 export 传入。生产代码零改动；无 orphan python 进程（pgrep 已确认）。
