# 配置

配置文件在`configs/`, 可以用CLI替换部分值。

主要配置段：

- `reader`：`auto|aslide|openslide`、缩略图宽度、可选本地 staging。
- `heuristic_pipe` 与 `heuristics`：方法顺序和方法参数。
- `heuristic_strategy`：`first`（默认，首个成功结果）或 `arbitrate`（运行全部候选并用
  density 前景做安全仲裁）。
- `heuristic_arbitration`：仲裁模式的 `min_specialized_confidence`、
  `min_density_recall` 和 `max_density_area_ratio` 门限。
- `post_filter_pipe` 与 `post_filters`：patch 读取后的后筛顺序和参数。默认关闭（空列表）。
  可用 `nonempty`（无细胞/纹理的纯底丢掉）和 `qc`（bundled MobileNetV3 keep/reject 分类器）。
  写入 pipe 的方法按顺序执行，第一个拒绝即丢弃该 patch。`qc` 需要 `torch` 与 `torchvision`，
  未列入 pipe 时不会加载模型或占用 GPU。

```yaml
post_filter_pipe:
  - nonempty
  - qc

post_filters:
  qc:
    checkpoint: null          # 默认使用内置 20260912T072309Z 权重
    device: auto              # auto|cpu|cuda|cuda:N|N
    threshold: null           # 默认使用预览时的 ckpt 内阈值 0.146775
    batch_size: 128           # 共享 QC 进程的 GPU 合批上限
    collect_timeout_ms: 8     # 跨 WSI worker 合批等待
    min_free_bytes: 1073741824  # auto 选卡时的空闲显存下限
```

`qc` 由父进程启动**一个** spawn 服务进程，模型只加载一次。WSI worker 只做 CPU 预处理并把 batch 送进该服务；服务按 `batch_size` 把多个 worker 的请求拼成一次 GPU forward。`device: auto`（默认）通过 NVML（不可用时回退 `nvidia-smi`）查询所有可见 GPU 的空闲显存，选卡过程不会创建 CUDA context；都不够则 warning 并回退 CPU。显式 `cuda:N` 时若该卡不存在或显存不够则报错。未列入 pipe 时不会启动服务、不加载模型。父进程退出（包括 `SIGKILL`）时，QC 服务和 WSI worker 都会由 Linux parent-death signal 终止。

- `patching`：level、输出 `mpp`、patch/output size、stride、mask 覆盖率、采样上限。
  `mpp: null` 时保留指定 level 的原生 MPP；设置数值时按该 level 读取后使用 LANCZOS
  缩放到目标 MPP。
- `output`：`jpeg|png|tar|none`、质量、shard 大小，以及最终输出权限。
  `permissions` 默认为 `"777"`；也可设为 `"755"`、`"664"` 等合法八进制模式。
  正式提取、preview 和 center-preview 完成写入后，会把各自输出目录内的所有文件、
  目录和子目录统一设为该权限。建议在 YAML 中加引号，CLI 可用
  `--set output.permissions=755` 临时覆盖。
- `parallel`：slide/read/encode/write workers 以及 inflight 上限。
- `logging`、`preview`：默认输出位置。

`./run_extract.sh --show-config` 会把完整 resolved config 写入启动时打印的日志文件；
`./run_extract.sh --inspect <WSI>` 只解析输入和配置，不打开 WSI 或写 patch，结果同样写入日志。

预设：

- `configs/default.yaml`：density-only、JPEG。
- `configs/high_throughput_nas.yaml`：顺序 TAR 分片和受控并发。
