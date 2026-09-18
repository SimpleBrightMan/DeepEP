# 完整调用示例

展示公开 API 如何放入实际业务流程；当前没有可运行模块。

## 规划内容与职责

- `_distributed.py`：进程组和 NPU 环境准备、退出。
- `legacy.py`：V1 布局准备、dispatch、参考专家计算、combine 与验证。
- `elastic.py`：V2 数据布局与 EPHandle 衔接。
- `overlap.py`：通信和独立计算重叠，以及消费结果前的完成依赖。
- 示例需固定有效的环境、dtype、shape、初始化与清理步骤。

[完整文件规划](../docs/repository-layout.md) · [架构说明](../docs/architecture.md)
