# API 声明设计

本文件记录固定 API 基线与算法目标。源码未构建扩展时，运行时调用仍明确失败；安装 wheel 后实现的资源/事件子集见[运行时设计](runtime-design.md)。所有未实现通信操作继续在读取 Tensor 前报 `NotImplementedError`。机器可读基线中的 `declarations_only` 是初始基线状态，不是当前运行时能力清单。

这一阶段允许框架维护者核对导入、参数和类型，让后续真实实现与同一套契约对齐。可导入不代表支持通信、训练或任何目标设备；设备资格仍见[路线图](roadmap.md)。

## 固定基线与范围

参考 [DeepSeek DeepEP](https://github.com/deepseek-ai/DeepEP/tree/a56d6156febcd9976e55adc85b5155bfac9f28f8)，提交 `a56d6156febcd9976e55adc85b5155bfac9f28f8`。

[机器可读基线](../tests/api/ascend_api.json)记录本项目选定方法的参数顺序/种类、默认值，以及参考源文件/行号和 SHA-256。它是经过平台适配的 Ascend 契约，不是原样复制的上游签名快照或上游全部 API 的兼容声明。契约测试离线运行，不依赖上游 main 或网络。

平台相关命名统一如下：`scaleup / scaleout` 表示域内/域间传输，`compute_units` 表示后端定义的调度计算单元，`device` 表示设备侧；不暗示具体链路、分核策略或算子能力。不保留旧平台关键字别名，旧调用需迁移；位置顺序和参考默认值不变。RDMA 等技术名仅在描述具体后端时使用，不与通信域名称混用。

### 本次检视修订的调用迁移

| 原名称 | 当前名称 |
| --- | --- |
| `Config.num_max_rdma_chunked_send_tokens` / `num_max_rdma_chunked_recv_tokens` | `num_max_scaleout_chunked_send_tokens` / `num_max_scaleout_chunked_recv_tokens` |
| `Config.get_rdma_buffer_size_hint` | `Config.get_scaleout_buffer_size_hint` |
| `Buffer.num_rdma_bytes` | `num_scaleout_bytes` |
| `Buffer.get_low_latency_rdma_size_hint` | `Buffer.get_low_latency_scaleout_size_hint` |
| `Buffer.dispatch(num_tokens_per_rdma_rank=...)` | `num_tokens_per_scaleout_rank=...` |
| `ElasticBuffer.get_theoretical_num_compute_units(rdma_gbs=...)` | `scaleout_gbs=...` |
| `Buffer.allow_scaleup_for_low_latency_mode` | `allow_memory_transport_for_low_latency_mode` |
| `Buffer.allow_multinode_scaleup` | `allow_multinode_memory_transport` |

构造参数行表示关键字名，不承诺对象存在同名属性。这里只修改名称，不新增原生执行能力。

### 通信域、传输语义与资源配置分层

`scaleup` 不等于某一种内存传输，也不必等于单节点。内存语义和消息语义后端必须分别进行能力确认，不能用一个域名把两者混为一谈。因此两个 `memory_transport` 开关只描述内存语义路径的许可，不包含其他 scale-up 后端，也不表示低时延或多节点能力已经实现。`num_scaleup_bytes / num_scaleout_bytes` 是对应域的容量参数，不选择传输引擎；当前仅前者接入单域 MTE 资源路径，后者的非零值仍拒绝。

`num_qps_per_rank`、`num_allocated_qps`、`num_qps` 和 `get_theoretical_num_qps` 是保留的 QP 接口，不作为 UDMA、RDMA、SDMA 共同的资源计数。当前 MTE 运行时不读取 QP 预算：V1 构造仅接受参考默认值 24，V2 构造仅接受默认值 0，其他值在创建 Runtime 前拒绝；这两个默认值不代表实际创建了相应数量的 QP。算法调用及 QP 容量查询仍未实现。

后续扩展采用以下设计约束，本次不增加空配置对象或未验证的队列实现：

1. 通信域和后端类型独立选择；多后端配置必须按后端具名，不能将单个 `num_qps` 广播到所有引擎。
2. 各后端分别定义资源类型、单位、作用域（每 rank/peer/channel）和容量上限；QP、通道和队列不做隐式等价转换。
3. 若保留旧标量入口，只能映射到一个已明确选择且经过验证的 QP 后端；混合后端歧义、冲突配置和未支持组合必须在资源分配前拒绝。
4. 具体字段和默认策略随对应后端实现单独评审，同时增加资源独立性、容量边界、冲突拒绝、创建/释放和真实设备验证。

| 对象 | 导入路径 | 本批范围 |
| --- | --- | --- |
| Buffer | `deep_ep`、`deep_ep.buffers`、`deep_ep.buffers.legacy` | V1 构造、layout、dispatch/combine、配置/流查询及主要低时延接口声明 |
| ElasticBuffer、EPHandle | `deep_ep`、`deep_ep.buffers`、`deep_ep.buffers.elastic` | V2 核心 EP、路由对象构造签名、流/拓扑/配置查询声明 |
| Config | `deep_ep` | V1 配置构造及容量提示接口声明 |
| EventHandle、EventOverlap | `deep_ep`、`deep_ep.utils`、`deep_ep.utils.event` | 捕获对象、等待、hook 和 context-manager 声明 |

各导入路径指向同一个类。`topk_idx_t` 依赖原生构建的索引宽度选择，本批不伪造 dtype 常量；Engram、PP、AGRS、设备架构探测、原生扩展内部方法及 V1 内部传输专用方法不在本批声明范围。完整选定方法清单以基线为准。

`Config` 暂置于私有 `_config.py`，原生绑定落地时继续使用同一个公开入口，不同时维护两套实际配置策略。上游默认值 20/6/256/6/256 仅为签名记录，**不是 Ascend 分核和工作区推荐配置**；当前 Config 构造也会报未实现。

## 一个明确的默认值差异

上游 `ElasticBuffer.dispatch(..., do_expand=False)` 在本项目声明为 `do_expand: Optional[bool] = None`，用于区分省略和显式传值。预定执行语义如下，当前桩不执行这些分支：

| 调用 | 目标行为 |
| --- | --- |
| 首次 dispatch，省略/传 None | 采用 False，保持不展开布局 |
| 首次 dispatch，显式 False/True | 使用指定布局 |
| cached dispatch，省略/传 None | 继承 handle 的原布局 |
| cached dispatch，显式模式与 handle 不同 | 明确报错，不能静默改变行布局 |

除上述平台命名和布局默认值差异外，其余选定 Python 方法保留参考接口的顺序和默认值。Config 的两个 size-hint 方法保留位置参数调用；文档参数名不扩展其关键字能力。事件和 Config 的参考签名来自上游 C++ 注册点。

## 目标数据与返回契约

以下是后续实现需要满足的目标，**当前调用均报未实现**。T 为源 rank 实际 token 数，N 为接收 rank 的去重行数，R 为有效专家路由行数，H 为 hidden，K 为 top-k；padding 能力另行资格验证。

| 操作 | 目标返回 |
| --- | --- |
| V1 get_dispatch_layout | rank counts、scale-out rank counts 或 None、expert counts、rank mask、event |
| V1 dispatch | recv_x、recv_topk_idx、recv_topk_weights、expert counts、handle、event，共六项 |
| V1 combine | combined_x、combined_topk_weights、event，共三项 |
| V2 dispatch | recv_x、recv_topk_idx、recv_topk_weights、EPHandle、EventOverlap，共五项 |
| V2 combine | combined_x `[T,H]`、combined_topk_weights `[T,K]` 或 None、EventOverlap，共三项 |

V2 False 使用 rank 去重布局：x `[N,H]`、idx/weights `[N,K]`；True 使用 expert-major 布局：x `[R,H]`、idx 为 None、weights `[R]`。权重输入可选，未传时权重输出为 None。重复路由按 top-k 槽位保留，不能简单去掉重复 expert。

BF16 激活、FP32 路由权重/梯度作为首个实现目标。签名中保留的 FP8、alignment、列主序 scale 布局、bias 和调优参数均未提供执行能力，不能根据参数存在推导支持域。

`combine(topk_weights=...)` 表示独立回传的路由值/梯度，不能自动把该值再乘到 x。前向 gate 权重应只在约定的一处施加。训练包装可使用 combine backward→cached dispatch、dispatch backward→combine 来回传 dX/dProb；本批没有 autograd 或通信反向实现。

## handle、事件和生命周期目标

EPHandle 的调用方字段包括 `do_expand`、`num_experts`、`num_max_tokens_per_rank`、`num_recv_tokens_per_expert_list`。原生路由计划应绑定 buffer，保留到对应 combine/backward/recompute 完成；后续非 cached dispatch 不得偷偷替换旧 handle 的计划。

capacity 是上界，实际 T 可在容量内变化，rank 间也可不同；cached 复用须匹配其具体路由输入。以上均为后续验收目标，当前没有内存池、路由检查或变长实现。

V1 `Buffer.capture()` 目标返回 EventOverlap，V2 `ElasticBuffer.capture()` 目标返回 EventHandle。入口仅解包 EventOverlap，不能对原始 EventHandle 再通用解包一次。异步返回后，调用方建立完成依赖才能消费结果；dtype/contiguous 转换应发生在 capture 前。接口实现需明确输出所有权，避免旧结果被后续通信覆盖。

本批连 `EventOverlap(None)` 也明确失败，不把空包装当作可用的“已完成事件”。同理，无法通过 stub 构造可复用 EPHandle。

## 模块边界与后续接入

```text
公开导入 deep_ep
    → buffers / utils / 私有 Config 声明
        → _native._unavailable(operation)
            → NotImplementedError，附操作名和实现状态
```

普通导入及签名检查不需要框架。已实现运行时经 `_native.require_native` 延迟加载同一安装目录的扩展，校验实际 SHMEM 库；不导入其他安装包中的 `deep_ep._C`。上图仅描述尚未实现操作的失败路径。运行时解析 Torch 注解不属于无依赖导入保证。

后续真实后端应在相同模块边界接入参数处理和绑定，并与功能测试一起启用；不能通过移除异常、返回占位 Tensor 或空 kernel 宣称实现完成。未支持模式仍需具体错误。对外异常含操作名和文档位置，不含设备凭据或内部环境信息。

## 验证

运行方式见 [tests/README.md](../tests/README.md)。CPU 测试核对导出身份、固定签名、默认值差异、调用绑定和所有声明的明确失败；隔离子进程阻止 Torch/torch_npu/原生扩展导入，检查没有隐式初始化。

安装框架的环境可使用 `--require-framework` 强制执行额外兼容检查：普通导入不初始化 NPU，未实现的通信/配置操作仍失败。已实现事件和资源由独立运行时测试覆盖。这项通过不代表通信、数值、训练或性能通过。
