# 配置

配置文件在`configs/`, 可以用CLI替换部分值。

主要配置段：

- `reader`：`auto|aslide|openslide`、缩略图宽度、可选本地 staging。
- `heuristic_pipe` 与 `heuristics`：方法顺序和方法参数。
- `heuristic_strategy`：`first`（默认，首个成功结果）或 `arbitrate`（运行全部候选并用
  density 前景做安全仲裁）。
- `heuristic_arbitration`：仲裁模式的 `min_specialized_confidence`、
  `min_density_recall` 和 `max_density_area_ratio` 门限。
- `post_filter_pipe` 与 `post_filters`：patch 读取后的后筛顺序和参数；如nonempty后筛。默认关闭。
- `patching`：level、输出 `mpp`、patch/output size、stride、mask 覆盖率、采样上限。
  `mpp: null` 时保留指定 level 的原生 MPP；设置数值时按该 level 读取后使用 LANCZOS
  缩放到目标 MPP。
- `output`：`jpeg|png|tar|none`、质量和 shard 大小。
- `parallel`：slide/read/encode/write workers 以及 inflight 上限。
- `logging`、`preview`：默认输出位置。

`./run_extract.sh --show-config` 会把完整 resolved config 写入启动时打印的日志文件；
`./run_extract.sh --inspect <WSI>` 只解析输入和配置，不打开 WSI 或写 patch，结果同样写入日志。

预设：

- `configs/default.yaml`：density-only、JPEG。
- `configs/high_throughput_nas.yaml`：顺序 TAR 分片和受控并发。
