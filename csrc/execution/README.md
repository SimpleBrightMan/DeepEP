# 执行准备与启动

根据业务约束、配置和实际已编译能力准备并启动一次执行。

## 规划内容与职责

- `plan.hpp/.cpp`：请求约束、资源要求和执行准备结果。
- `variants.hpp/.cpp`：查询由构建产生的实际变体索引。
- `launcher.hpp/.cpp`：绑定输入输出、内部参数与流依赖到 Kernel。
- 配置只选择合法候选；运行地址和对象引用在当前进程绑定。

[完整文件规划](../../docs/repository-layout.md) · [架构说明](../../docs/architecture.md)
