# Ascend 原生实现

按绑定、业务、持续资源、单次执行与设备实现划分维护责任。

## 规划内容与职责

- `python_api.cpp`：核心 deep_ep._C 的模块初始化与注册汇总。
- `kernel_args.hpp`：Host/Device 共用内部参数的单一声明源。
- `bindings/` → `ops/` → `execution/` → `kernels/`：主要调用链。
- `runtime/`：共享上下文、资源与事件；`device/`：按需共享设备源码。
- `CMakeLists.txt`：Host、绑定和 Device 目标组合。

[完整文件规划](../docs/repository-layout.md) · [架构说明](../docs/architecture.md)
