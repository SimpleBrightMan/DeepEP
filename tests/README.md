# 功能与集成验证

按照 API 契约、业务结果和集成行为组织测试。当前可运行 `api/` 的声明与错误行为测试；通信数值、训练和性能测试尚未提供。

## 运行 API 契约测试

从仓库根目录运行（建议使用隔离的 Python 环境）：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/api -m "not framework"
```

无需安装 Torch、torch_npu、CANN 或 SHMEM。测试检查公开对象身份、固定签名/默认值、明确报错、无可选依赖导入，以及未实现事件不能假装完成。

在已配置 Torch/torch_npu 的环境额外运行：

```bash
python -m pytest tests/api --require-framework -s
```

该模式要求框架兼容检查实际执行；依赖缺失会失败。检查仅导入框架、验证接口仍明确报错且未初始化 NPU，不分配设备 Tensor 或运行通信。普通 CPU 运行会明确跳过该单项；其他错误不会被转换为 skip。

GitCode 的 `compile` 触发流水线已增加 CPU API 契约 job。框架兼容检查、通信功能资格与流水线扫描结果应分别记录。

## 规划内容与职责

- `api/`：固定原生 API 基线、导出和典型完整调用。
- `legacy/`：V1 dispatch、combine、低时延和复用路径。
- `elastic/`：V2 dispatch、combine、EPHandle 与布局衔接。
- `integration/`：核心/Beta 安装、框架接入与资源生命周期。
- `utils/`：分布式环境辅助和业务参考；Beta 专属测试放在实验目录。

[完整文件规划](../docs/repository-layout.md) · [架构说明](../docs/architecture.md)
