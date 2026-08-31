# extract_patch

通用、高吞吐的 WSI patch 提取工具。

## 环境

```bash
pip install numpy pillow pyyaml tqdm opencv-python-headless openslide-python
```

单独安装(ASlide)[https://github.com/MrPeterJin/ASlide]

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

### level, mpp, size

默认使用最清晰的level0, mpp默认对齐到0.5 (upsample会使用LANCZOS插值)，patch size conventionally: 1200/600取决于器官。

## 正式提取

```bash
./run_extract.sh --input-list inputs.txt \
  --config configs/default.yaml \
  --output /path/to/patches
```

正式提取默认使用 `nohup` 转入后台，并打印 PID 和日志路径；`--output` 不存在时会
自动递归创建。需要前台运行时追加 `--foreground`。

查看完整配置：

```bash
./run_extract.sh --show-config --config configs/default.yaml
```

## 输出

普通文件模式的 patch 目录只包含图片，命名方式为{x}_{y}_{output_size}.jpg
：

```text
patches/
└── slide_id/
    ├── 0_0_1200.jpg
    └── ...
```

每次运行只保留一个 `logs/<run_id>.log`。全部成功后自动清理临时恢复状态；仅在失败时，才会在隐藏的 `logs/.state/<run_id>/` 中保留 manifest 和失败信息用于重试。也可选择PNG、只生成坐标或 TAR 分片输出。

详细说明：

- [Heuristics](docs/heuristics.md)
- [配置](docs/configuration.md)
- [输出与恢复](docs/outputs.md)
- [性能调优](docs/performance.md)
