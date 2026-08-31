# 配置

配置文件在`configs/`, 可以用CLI替换部分值。

主要配置段：

- `reader`：`auto|aslide|openslide`、缩略图宽度、可选本地 staging。
- `heuristic_pipe` 与 `heuristics`：fallback 顺序和方法参数。
- `post_filter_pipe` 与 `post_filters`：patch 读取后的后筛顺序和参数；如nonempty后筛。默认关闭。
- `patching`：level、patch/output size、stride、mask 覆盖率、采样上限。
- `output`：`jpeg|png|tar|none`、质量和 shard 大小。
- `parallel`：slide/read/encode/write workers 以及 inflight 上限。
- `logging`、`preview`：默认输出位置。

`./run_extract.sh --show-config` 可显示完整 resolved config；
`./run_extract.sh --inspect <WSI>` 只解析输入和配置，不打开 WSI 或写 patch。

预设：

- `configs/default.yaml`：density-only、JPEG。
- `configs/high_throughput_nas.yaml`：顺序 TAR 分片和受控并发。
