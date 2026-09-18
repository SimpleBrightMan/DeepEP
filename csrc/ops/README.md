# EP 业务操作

解释输入模式、路由、布局、handle 条件和结果意义，向 execution 提交本次执行需求。

## 规划内容与职责

- `config.hpp`：Config 的唯一 native 定义及默认值。
- `legacy.hpp/.cpp`、`elastic.hpp/.cpp`：对应接口族的业务操作。
- `layout.hpp/.cpp`：路由计数和布局准备编排。
- 设备资源由 runtime 提供；实现选择和 Kernel 启动由 execution 承接。

[完整文件规划](../../docs/repository-layout.md) · [架构说明](../../docs/architecture.md)
