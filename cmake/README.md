# 原生构建规则

根工程统一解析依赖，明确 Host/Device 编译目标和安装产物。

## 规划内容与职责

- `Dependencies.cmake`：有效依赖来源、版本与 targets。
- `HostTargets.cmake`：Python、Torch、torch_npu 与 Host ABI。
- `AscendC.cmake`：对锁定 CANN 工具链的编译适配。
- `Install.cmake`：安装规则；`build_info.json.in`：实际配套及能力元数据。

[完整文件规划](../docs/repository-layout.md) · [架构说明](../docs/architecture.md)
