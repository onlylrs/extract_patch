# extract_patch

通用、高吞吐的 WSI patch 提取工具。

## 环境

```bash
pip install pillow pyyaml tqdm openslide-python
```

单独安装(ASlide)[https://github.com/MrPeterJin/ASlide]

```bash
pip install "opencv-python-headless>=4.10,<4.12"
```

启用 `qc` 后筛时再安装 PyTorch（未写入 `post_filter_pipe` 时不会加载）：

```bash
pip install torch torchvision
```


## Preview

单张 WSI：

```bash
./run_extract.sh --preview /path/to/slide.sdpc --config configs/default.yaml
```

多张 WSI：

```bash
./run_extract.sh --preview /path/a.svs /path/b.mrxs --n-patches 8
```

TXT（一行一个路径，引号可有可无）. 示例：
```
/jhcnas6/Private/POH/26AC003863_-340669272H
/jhcnas6/Private/POH/26AC003865_-368164448X
/jhcnas6/Private/POH/26AC003867_-162307111S
```

```bash
./run_extract.sh --preview --input-list inputs.txt --config configs/poh.yaml
```

Preview 功能会按照当前配置的策略导出WSI的分割区域图`contour.jpg`，以及随机保存8张patches `patches/`供检查.
默认保存在 `outputs/preview/`.

只检查正式 extraction 使用的 thumbnail 和 foreground mask，不读取 patch：

```bash
./run_extract.sh --center-preview --input-list inputs/0.txt \
  --config configs/default.yaml --output /jhcnas6/Private/0/patch
```

这里的 `--output` 仍表示 patch 根目录；检查结果会写到同级的
`preview/thumbnail/` 和 `preview/mask/`。其中 mask 图片是与 preview 模式
`contour.jpg` 相同的分割轮廓和 patch 网格可视化，不是黑白二值 mask。再次运行时，
已同时生成 thumbnail 和 mask 的 WSI 会跳过；只生成其中一个的 WSI 会重新处理。

### level, mpp, size

默认使用最清晰的level0, mpp默认对齐到0.5 (upsample会使用LANCZOS插值)，patch size conventionally: 1200/600取决于器官。

使用 ASlide backend 时会自动启用 `Real` color correction；该设置同时作用于
thumbnail、preview 和正式提取的 patch。OpenSlide 格式带 ICC profile 时会转换到
sRGB；没有 ICC profile 时保持原始颜色。每张 WSI 的日志会明确记录
`color_correction=enabled:...` 或 `unavailable:...`。

## 正式提取

```bash
./run_extract.sh --input-list inputs.txt \
  --config configs/default.yaml \
  --output /path/to/patches
```

`run_extract.sh` 的所有模式都会使用 `nohup` 转入后台，并打印唯一父进程 PID 和
`logs/<run_id>.log` 路径；`--output` 不存在时会自动递归创建。终止任务时只需
`kill <PID>`，父进程会停止并回收全部 WSI worker。Linux parent-death signal 也会在
父进程被 `kill -9` 时终止 WSI worker，但普通 `kill` 可以完整刷新日志与恢复状态。
输入清单逐行解析并通过有界队列提交，首条启动日志在扫描 WSI 前写入；大清单不会
等待全部路径解析完才开始处理。日志中的 `input_progress` 会持续显示解析进度。

查看完整配置（内容写入启动时打印的日志文件）：

```bash
./run_extract.sh --show-config --config configs/default.yaml
```

后筛默认关闭。需要丢掉空白 patch 或运行 QC 分类器时，在配置里打开 `post_filter_pipe`：

```yaml
post_filter_pipe:
  - nonempty
  - qc

post_filters:
  qc:
    device: auto
    batch_size: 128
```

`qc` 使用仓库内置的 MobileNetV3-Large 权重（keep/reject）。父进程只启动一个 QC 服务、只加载一份模型；各 WSI worker 把 patch 送进去合批推理，再按各自 `index.json` 的 plan 顺序写回。`device: auto` 会选一张空闲显存足够的 GPU，没有则 warning 并用 CPU。未写入 pipe 时不加载 PyTorch、不占 GPU。`nonempty` 与 `qc` 相互独立，可单独或串联使用。

## 输出

Patch 文件统一命名为
`x<x>_y<y>_mpp<mpp>_px<pixel size>.jpeg`。TAR shard 使用
`<WSI name>_0001.tar`、`<WSI name>_0002.tar` 等名称。

普通文件模式的 patch 目录包含图片和 `index.json`
：

```text
patches/
└── slide_id/
    ├── x0_y0_mpp0.5_px1200.jpeg
    └── ...
```

TAR 模式可通过 `output.tar_preview: true` 在每个 `slide_id/sample/` 中额外保存
`output.tar_preview_n` 张随机 JPEG（默认 20），无需解压分片即可浏览。

每张 WSI 成功完成后，会在对应的 `slide_id/` 中写入 `index.json`。其中 patch
按 plan 顺序连续编号；TAR 模式的每项还会记录所在的 shard。

正式提取成功后，还会在 patch 根目录旁写入
`preview/thumbnail/<WSI>.jpeg` 和 `preview/mask/<WSI>.jpg`。

每次运行只保留一个 `logs/<run_id>.log`。恢复以 patch 输出目录为准，因此更换
`run_id` 后仍会跳过带完整 `index.json` 的 WSI；未完成的文件或 TAR shard 会按已有
patch 文件名继续。全部成功后自动清理本次运行状态；失败时在隐藏的
`logs/.state/<run_id>/` 中保留 manifest 和失败信息。也可选择 PNG、只生成坐标或 TAR 分片输出。

详细说明：

- [Heuristics](docs/heuristics.md)
- [配置](docs/configuration.md)
- [输出与恢复](docs/outputs.md)
- [性能调优](docs/performance.md)
- [示例](docs/examples.md)
