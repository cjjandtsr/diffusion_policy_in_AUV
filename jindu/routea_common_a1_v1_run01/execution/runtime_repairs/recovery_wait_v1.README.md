# runtime repair 3: 等待器被 v1 残留 rc 文件欺骗（非任务故障）

- 现象: 等待器在诊断启动后数秒报完成；恢复基线 rc=1；log 停在 STAGE_START 03_preflight。
- 真相: rc 文件 mtime=16:08:19 属 v1 失败启动残留（repair 2 的 trap 失败写 rc=1）；
  v2 启动器(54590)+控制器(54592)持续运行，07_recovery.log 正常产出（首条 RECOVERY bc … seconds 34.6）。
- 判断: 任务正常，判完成条件改为 rc 文件 mtime > v2 启动时间 16:09:01，或 launcher+python 均死。
