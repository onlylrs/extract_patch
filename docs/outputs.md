# 输出与恢复

## Patch 输出

`jpeg` 和 `png` 模式下，每张 WSI 一个目录，目录内只包含 patch 图片。文件名保存
level-0 坐标、读取 level 和输出大小，可被 `extract_feat` 继续消费。

`tar` 模式在每个 `slide_id/` 中将图片顺序写入 `shard-000000.tar` 等分片，减少 NAS上的小文件随机写入。启用 `output.tar_preview` 后，还会在 `slide_id/sample/` 中确定性随机抽取 `output.tar_preview_n` 张 JPEG（默认 20）供直接浏览。
`none` 模式只规划坐标，适合评估切割范围和 patch 数量。

`jpeg`、`png` 和 `tar` 模式在每张 WSI 成功完成后写入 `slide_id/index.json`。
索引中的 patch 按 plan 顺序连续编号；每项包含 `index` 和 `name`，仅 TAR 模式
额外包含 `shard`。

## 日志和恢复状态

正常完成后只保留 `logs/<run_id>.log`，其中包含配置签名、每张 WSI 的 reader、heuristic、patch 数量、耗时、错误和最终汇总。

仅失败或未完成时，隐藏目录 `logs/.state/<run_id>/` 才会保留
`patch_manifest.csv`、`config.sha256` 和 `failures.txt`。

只有当所有计划 patch 均确认写入成功，任务才标记成功并清理临时 manifest/state。
