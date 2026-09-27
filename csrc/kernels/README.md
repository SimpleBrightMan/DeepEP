# 设备算子

- `notify/notify.cpp`：路由元数据计算与交换。
- `dispatch/dispatch.cpp`：URMA primary、MTE secondary、BF16/FP8 payload 和 scales。
- `combine/combine.cpp`：BF16 专家输出回传与归约。
- `common.hpp`：Notify 和 combine 共用的 rank 调度映射，仅保留实际使用的函数。

每个算子目录只保留实现 `.cpp`、Host 调用声明 `launch.hpp` 和 Host/Device 共用布局
`tiling.hpp`。仅本算子使用的辅助实现在 `.cpp` 内。
三者由 `cmake/device/CMakeLists.txt` 构建进 `libdeepep_kernels.so`，经 `csrc/ops` 接入 Python。
接口、限制和测试命令见 [Elastic 文档](../../docs/elastic.md)。
