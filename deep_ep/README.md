# Python API

维护公开对象、用户参数处理、原生方法委托和返回组织。

普通导入不加载 Torch/torch_npu 或创建 NPU 资源。安装扩展后支持受限资源和真实事件；未实现算法继续明确报错。详见 [API 契约](../docs/api-contracts.md)及[运行时设计](../docs/runtime-design.md)。

## 规划内容与职责

- `__init__.py`：汇总 Buffer、Config、ElasticBuffer、EPHandle 和完成对象导出。
- `_native.py`：延迟加载自身扩展并验证 SDK 身份，保留未实现错误边界；`_runtime.py` 管理共享资源生命周期。
- `buffers/legacy.py`：V1 接口；`buffers/elastic.py`：V2 与 EPHandle。
- `utils/event.py`：完成包装；`utils/envs.py`：本库环境选项。

[完整文件规划](../docs/repository-layout.md) · [架构说明](../docs/architecture.md)
