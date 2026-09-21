# 运行上下文与资源

维护跨调用存续的资源集合、所有权、初始化、释放和完成对象。

## 规划内容与职责

当前 `session.hpp/.cpp` 实现独占 SHMEM 资源的建立、工作区和显式销毁；只借用框架 ACL context。
真实事件委托 torch_npu 的事件 API，不建立另一套 ACL 全局初始化。支持边界见[运行时设计](../../docs/runtime-design.md)。
下列更细模块划分随实际消费者演进，不创建空实现。

- `context.hpp/.cpp`：本库操作共享的运行上下文。
- `resources.hpp/.cpp`：实际资源的取得、引用与释放适配。
- `event.hpp/.cpp`：原生完成对象与框架事件衔接。
- 核心和 Beta 按实际需要复用明确所有权的资源；生命周期进入集成测试。

[完整文件规划](../../docs/repository-layout.md) · [架构说明](../../docs/architecture.md)
