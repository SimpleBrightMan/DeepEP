# Python API

维护公开对象、用户参数处理、原生方法委托和返回组织。

当前提供 API 声明和明确失败的桩。普通导入不加载 Torch/torch_npu 或创建 NPU 资源；所有运行时构造和操作均抛出 `NotImplementedError`。详见 [API 声明设计](../docs/api-contracts.md)。

## 规划内容与职责

- `__init__.py`：汇总 Buffer、Config、ElasticBuffer、EPHandle 和完成对象导出。
- `_native.py`：当前提供未实现错误边界，后续集中加载核心扩展；`_version.py` 尚未提供。
- `buffers/legacy.py`：V1 接口；`buffers/elastic.py`：V2 与 EPHandle。
- `utils/event.py`：完成包装；`utils/envs.py`：本库环境选项。

[完整文件规划](../docs/repository-layout.md) · [架构说明](../docs/architecture.md)
