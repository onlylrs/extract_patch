# 性能调优

并发分为四层，默认值偏保守：

- `slide_workers`：同时处理的 WSI 数。
- `read_workers_per_slide`：每张 WSI 的并行读取数。
- `encode_workers`：JPEG/PNG 编码数。
- `writer_workers`：落盘任务数。

`max_inflight_patches` 和 `max_inflight_bytes` 是 backpressure 上限。读取速度高于编码或NAS 写入速度时，生产者会等待，而不是继续堆积内存。

建议：

1. NAS 小文件压力高时优先使用 `output.mode: tar`。
2. 先提高 `read_workers_per_slide`，观察吞吐；不要同时盲目提高所有 worker。
3. 多张大 WSI 时再提高 `slide_workers`，同时关注 NAS 和内存。
4. OpenCV 内部线程通常设为 1，避免与外层并发过度订阅 CPU。
5. MRXS 随机读取慢时可启用 staging；需要足够的本地或共享内存空间。

日志中的 segmentation/read/encode/write 时间和队列峰值用于定位瓶颈。
