# 原生绑定

连接 Python/native 类型、对象和调用；业务语义委托给 ops，资源操作连接 runtime。

## 规划内容与职责

- `bindings.hpp`：私有模块注册声明。
- `legacy.cpp`、`elastic.cpp`：V1/V2 对象和方法注册。
- `config.cpp`：Config 注册；实际类型定义由 ops/config.hpp 维护。

[完整文件规划](../../docs/repository-layout.md) · [架构说明](../../docs/architecture.md)
