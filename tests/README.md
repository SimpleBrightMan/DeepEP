# 功能与集成验证

按照 API 契约、业务结果和集成行为组织测试；当前没有可运行测试代码。

## 规划内容与职责

- `api/`：固定原生 API 基线、导出和典型完整调用。
- `legacy/`：V1 dispatch、combine、低时延和复用路径。
- `elastic/`：V2 dispatch、combine、EPHandle 与布局衔接。
- `integration/`：核心/Beta 安装、框架接入与资源生命周期。
- `utils/`：分布式环境辅助和业务参考；Beta 专属测试放在实验目录。

[完整文件规划](../docs/repository-layout.md) · [架构说明](../docs/architecture.md)
