# runtime repair 1: environments/factory.py 旧版

- 症状: §2 交付核验 36/37 OK，factory.py FAILED（WSL 932236e2… ≠ 清单 2216ab7b…）。
- 根因: WSL 侧该文件为旧版，缺 6 行（terminate_at_space_limits/space_cost_budget 参数与传递）；其余 36 文件均已由用户复制到位。
- 修复: 从 Windows 交付源码复制（其 hash=2216ab7b… 与交付清单逐位一致），备份旧版为 factory.py.wsl_before.bak，diff 见 factory.py.diff。
- 授权依据: 指导 §5「可修正已确认的文件位置…不改变计算含义」；不修改文件内容、不改变交付清单。
