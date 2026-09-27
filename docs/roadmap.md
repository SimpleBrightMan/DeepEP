# 路线图与资格矩阵

本文记录项目开发计划、功能进展和设备验证状态。开发阶段按依赖关系组织，发布日期将另行公布。

## 当前状态

| 项目 | 状态 | 开发内容 |
| --- | --- | --- |
| API 基线 | 已固定、已提供声明 | 原生提交 `a56d6156febcd9976e55adc85b5155bfac9f28f8`；导出/签名/错误行为测试，详见 [API](api.md) |
| API 运行时 | 资源/事件及受限 V2 dispatch/combine 已实现 | 单 session MTE+URMA、真实事件、Notify/expanded dispatch/combine，见[运行时设计](runtime-design.md) |
| V2 expanded dispatch | 已接通，设备资格待验证 | BF16/FP8、weights、fresh/cached、forward-list MTE 转发，见[专项测试](elastic.md) |
| V2 expanded combine | 已接通，设备资格待验证 | BF16、重复调用、复用 dispatch handle，见 [Elastic 文档](elastic.md) |
| 独立参考与功能测试框架 | 已提供 | 保留 CPU Golden 和基础测试；当前 Elastic 使用 dispatch/combine 两个独立 NPU 入口，见[测试说明](../tests/README.md) |
| 构建与发行 | Host wheel 构建已实现 | 根 CMake、构建桥、SDK 身份、sdist 重建和安装验证；正式发行/算法资格仍待完成 |
| V1 功能桩与小型闭环 | CPU 模拟已提供 | 显式测试后端、6 个具名 BF16 场景、逐专家计算/归并、L2 和故障注入；不表示公开 wheel 算法或 NPU 通信通过，见[设计](functional-roundtrip.md) |
| NPU Tensor 小型闭环 | 双卡功能桩已验证；原生入口已提供 | NPU Tensor/HCCL 参考通信，6 场景逐 rank 通过；`validate_package.py --ep-roundtrip` 和免重建入口供后续原生算子验收，见[NPU 测试](npu-roundtrip.md)；不等于原生 EP 通过 |
| V1/V2/低时延/异步 | 待实现、待设备验证 | 完整业务用例与目标设备资格 |
| MegaMoE Beta | 待实现、待设备验证 | 独立扩展、融合流程与端到端基线 |
| 性能框架 | 已提供、设备未测量 | CLI/dry-run、计时/统计引擎、JSON/CSV 及 CPU 协议测试；原生适配器待接入，缺失能力在初始化前拒绝，见[详细设计](benchmark-design.md) |
| 许可证与第三方声明 | 待维护者确定 | 建立 `LICENSE`、`NOTICE` |

## 开发阶段

| 阶段 | 开发内容 | 完成条件 |
| --- | --- | --- |
| 工程基础 | 包入口、根构建、Host/Device 目标、固定依赖和安装清单 | 从干净源码构建、安装并核对真实产物 |
| 核心 EP | V1/V2 的真实实现、用例及固定 API 基线 | 公开调用、业务结果和完成行为通过目标设备测试 |
| Beta 发行 | MegaMoE 完整单元、可选安装及性能入口 | 独立功能测试、匹配基线与端到端测量可复现 |
| 共享提炼 | 有真实消费者的设备辅助或组件 | 契约、正确性、性能和维护收益可验证 |

## 设备资格记录

| 模式 | 设备/拓扑 | dtype/shape/规模 | 测试记录 | 当前资格 |
| --- | --- | --- | --- | --- |
| Host 资源与事件（非 EP 算法） | Ascend950，单机 MTE | HCCL EP2/4/8；Gloo EP2；显式工作区 | [运行时验证](runtime-design.md) | 受限资源/事件验证通过 |
| V1 高吞吐 | 待确定 | 待确定 | 无 | 未验证 |
| V1 低时延 | 待确定 | 待确定 | 无 | 未验证 |
| V2 | 待确定 | 待确定 | 无 | 未验证 |
| 异步与缓存复用 | 待确定 | 待确定 | 无 | 未验证 |
| MegaMoE Beta | 待确定 | 待确定 | 无 | 未验证 |

正式发行逐项固定源码、配置、依赖、命令和结果位置。README、元数据、构建产物与资格矩阵应描述同一支持范围。
