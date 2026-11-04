# status-card 分支

本分支只存放项目当前状态卡 `jindu/当前状态卡.md`，与 main 无共同历史，**不合并到 main**。

- 写入方：仅 Claude（依据 Codex 交付物末尾的“状态变更”块及 Claude 的分析结论更新）。
- 读取：Codex 开工前运行
  `git fetch origin status-card && git show origin/status-card:jindu/当前状态卡.md`
- 历史版本：`git log origin/status-card -- jindu/当前状态卡.md`
- 规则见本地 `C:\Users\cjj\Desktop\IDP3\` 下《Claude与Codex协作分工规范》§5。
