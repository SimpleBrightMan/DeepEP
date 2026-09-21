# API 与兼容性规划

本文定义 V1/V2 接口的兼容性目标与工作流。接口基线和未支持操作的失败测试已提供；安装原生扩展后可使用受限资源构造/销毁、通信流和真实事件。支持条件见[运行时设计](runtime-design.md)。dispatch/combine 等 EP 算法仍报 `NotImplementedError`。目标数据语义见[API 契约](api-contracts.md)，资格见[路线图](roadmap.md)。

## 公开对象与职责

| 对象 | 规划位置 | 角色 |
| --- | --- | --- |
| `Buffer` | `deep_ep/buffers/legacy.py` | V1 布局、dispatch、combine 与对应低时延流程 |
| `ElasticBuffer` | `deep_ep/buffers/elastic.py` | V2 业务操作 |
| `EPHandle` | 与 V2 同组 | V2 布局及跨调用业务元数据 |
| `Config` | 当前在私有 `_config.py` 声明，后续由原生绑定提供 | 兼容配置对象；当前不能构造，不提供 Ascend 调优默认值 |
| `EventOverlap` 等完成包装 | `deep_ep/utils/event.py` | 用户可观察的完成与等待行为 |

公开导入集中在 `deep_ep/__init__.py`。普通导入不加载 Torch、torch_npu 或扩展；运行时调用通过 `_native.py` 延迟加载自身安装的扩展并校验 SDK 身份。相同公开类型可从对应 buffers/utils 模块导入。

## 工作流

- V1：准备布局 → dispatch → 按接收布局计算专家 → combine。
- V1 低时延：按独立方法、输出和完成契约实现，逐模式验证。
- V2：通过 `ElasticBuffer` 产生数据与 `EPHandle`，按对应布局完成专家计算，再使用匹配的 handle 执行 combine。
- 异步与缓存：消费结果前建立事件或基线规定的完成依赖；handle 复用需满足路由、布局、实例和配置约束。

权重处理、量化/scale、有效区域、梯度流程与 hook 行为随具体模式定义，实施时从固定的原生接口基线取得，不能仅凭名称推导。

## 固定基线

已新增 `tests/api/ascend_api.json`，记录所选 EP 接口的固定基线和差异：

| 字段类别 | 要求 |
| --- | --- |
| 参考版本 | 原生仓地址、固定 tag/commit、核验日期 |
| 公开对象 | 导入名、类型身份、方法、签名和默认值 |
| 输入与输出 | dtype、shape、布局、可选输入、返回结构、有效区域 |
| 业务行为 | 路由、权重、handle 生命周期与复用条件 |
| 完成行为 | 异步选项、事件、hook、允许消费结果的时机 |
| 覆盖证据 | 测试路径、目标设备和本版本支持范围 |

兼容性基线固定到 [DeepEP 原生仓库](https://github.com/deepseek-ai/DeepEP/tree/a56d6156febcd9976e55adc85b5155bfac9f28f8) 的提交 `a56d6156febcd9976e55adc85b5155bfac9f28f8`。源码路径、SHA-256、方法签名和明确的差异离线保存。基线仅覆盖本批选定对象，不代表兼容上游全部功能。

## Ascend 参数语义

保持原生参数名时，需要明确它映射到哪些 Ascend 资源、支持范围与默认行为。AIV 核数、workspace、对齐、设备拓扑和策略的具体数值均待实现验证，不直接沿用其他仓库的默认值。

接口变更同时更新绑定、业务处理、示例、兼容性测试及[资格矩阵](roadmap.md)。

公共命名区分通信域 `scaleup / scaleout`、内存语义 `memory_transport` 和后端资源 QP；本次关键字迁移和后续多引擎配置约束见 [API 契约](api-contracts.md)。命名不代表新增后端支持。
