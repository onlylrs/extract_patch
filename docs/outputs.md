# 输出与恢复

## Patch 输出

`jpeg` 和 `png` 模式下，每张 WSI 一个目录。所有 patch 使用
`x<x>_y<y>_mpp<mpp>_px<pixel size>` 命名；JPEG 扩展名为 `.jpeg`。

`tar` 模式在每个 `slide_id/` 中将图片写入 `<WSI name>_0001.tar` 等分片，减少 NAS上的小文件随机写入。启用 `output.tar_preview` 后，还会在 `slide_id/sample/` 中确定性随机抽取 `output.tar_preview_n` 张 JPEG（默认 20）供直接浏览。
`none` 模式只规划坐标，适合评估切割范围和 patch 数量。

`jpeg`、`png` 和 `tar` 模式在每张 WSI 成功完成后写入 `slide_id/index.json`。
索引中的 patch 按 plan 顺序连续编号；每项包含 `index` 和 `name`，仅 TAR 模式
额外包含 `shard`。

正式提取成功后，在 patch 根目录的同级 `preview/` 中保存
`thumbnail/<WSI>.jpeg` 和 `mask/<WSI>.jpg`。后者是带分割轮廓与 patch 网格的
thumbnail 可视化，与 preview 模式的 `contour.jpg` 相同；配置后筛时只绘制实际
保留的 patch 网格。

所有输出文件、目录和子目录在本次写入完成后都会递归应用
`output.permissions`，默认是 `"777"`。TAR 内 JPEG 成员的 mode 也使用同一配置。
例如可通过 `--set output.permissions=755` 改为 `755`。

`--center-preview` 可直接断点续跑：同一 WSI 的非空 thumbnail 和 mask 均存在时
跳过；任一文件缺失或为空时重新生成这一对文件。`output.overwrite: true` 会强制重做。

恢复不依赖 `run_id`。已有有效 `index.json`、preview 和 sample 的 WSI 会整张跳过；
未完成的 JPEG/PNG 目录会复用已有图片，未完成的 TAR 目录会扫描已有 shard 成员并
只补齐缺少的 patch。完成后重新生成 canonical `index.json` 和完整 sample。

## 日志和恢复状态

正常完成后只保留 `logs/<run_id>.log`，其中包含配置签名、每张 WSI 的 reader、heuristic、patch 数量、耗时、错误和最终汇总。

仅失败或未完成时，隐藏目录 `logs/.state/<run_id>/` 才会保留
`patch_manifest.csv`、`config.sha256` 和 `failures.txt`。

只有当所有计划 patch 均确认写入成功，任务才标记成功并清理临时 manifest/state。
