# 上传说明（GitHub 副本）

- 本目录为 Windows 侧 `C:\Users\cjj\Desktop\IDP3\program\f8_probe_run02_20261117` 的镜像。
- **仅 `*.jsonl` 做了 gzip 压缩**（`samples.jsonl.gz` / `flights.jsonl.gz`），其余文件（report、README、gates/identity/summary JSON、logs、csv、py）保持原样未压缩。
- 解压：`find . -name '*.jsonl.gz' -print0 | xargs -0 -n1 gunzip -k`
- 原始文件的 sha256 见 `logs/checksums.txt` 与各阶段 `identity.json`。
