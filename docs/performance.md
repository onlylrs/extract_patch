# 性能调优

并发分为 WSI 进程和单张 WSI 内部线程两层：

- `slide_workers`：同时处理的 WSI 进程数，preview 和正式 extraction 都使用它。
- `read_workers_per_slide`：每张 WSI 的并行读取数。
- `encode_workers`：JPEG/PNG 编码数。
- `writer_workers`：落盘任务数。

默认同时处理 4 张 WSI。每个 WSI 进程内部使用独立的读取、编码和写入线程；
`max_inflight_patches` 和 `max_inflight_bytes` 是每个 WSI 进程的 backpressure 上限。
读取速度高于编码或 NAS 写入速度时，生产者会等待，而不是继续堆积内存。

父进程集中记录日志和 manifest。单张 WSI 的打开、读取或分割错误会记为 failed 并
继续下一张；worker 崩溃、内存不足、文件描述符耗尽或磁盘满会终止整次运行并回收
全部子进程。通过 `run_extract.sh` 启动后，只需终止脚本打印的父进程 PID。

建议：

1. NAS 小文件压力高时优先使用 `output.mode: tar`。
2. 先提高 `read_workers_per_slide`，观察吞吐；不要同时盲目提高所有 worker。
3. 多张大 WSI 时再提高 `slide_workers`，同时关注 NAS 和内存。
4. OpenCV 内部线程通常设为 1，避免与外层并发过度订阅 CPU。
5. MRXS 随机读取慢时可启用 staging；需要足够的本地或共享内存空间。

日志中的 segmentation/read/encode/write 时间和队列峰值用于定位瓶颈。
