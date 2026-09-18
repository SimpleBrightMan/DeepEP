# 配置分工

分别管理资源连接、实现策略与发行内容；当前仅建立分类目录，尚无可执行配置。

## 规划内容与职责

- `topologies/`：设备分组、互联和共享资源，目标文件 example.json。
- `policies/`：合法实现的分核、Tile、流水和重叠策略，目标文件 default.json。
- `release/`：功能开关、架构和配套，目标文件 core.json、with_megamoe.json。
- 配置 schema、参数值和支持范围由真实消费者与设备验证确定。

[完整文件规划](../docs/repository-layout.md) · [架构说明](../docs/architecture.md)
