# Python API

维护公开对象、用户参数处理、原生方法委托和返回组织。

## 规划内容与职责

- `__init__.py`：汇总 Buffer、Config、ElasticBuffer、EPHandle 和完成对象导出。
- `_native.py`：集中加载核心扩展与配套原生库；`_version.py`：版本信息。
- `buffers/legacy.py`：V1 接口；`buffers/elastic.py`：V2 与 EPHandle。
- `utils/event.py`：完成包装；`utils/envs.py`：本库环境选项。

[完整文件规划](../docs/repository-layout.md) · [架构说明](../docs/architecture.md)
