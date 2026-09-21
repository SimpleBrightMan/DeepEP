# V1 dispatch/combine 功能桩与小型集成测试

## 目标和边界

先在不依赖设备的环境验证完整的数据流程：

```text
确定性输入 → dispatch → 检查接收数据/索引/计数
          → 逐专家变换和本地加权归约 → combine → 全 rank 精度检查
```

`tests/integration/functional_stub.py` 是显式选择的、单进程模拟两个 rank 的测试适配器。
它产生真实计算结果，而不是空返回值或“未实现”异常；但它**不是** `deep_ep.Buffer` 的替代实现，
不模拟其二进制 ABI，不注册到原生后端，不装入 wheel，不发起 NPU/SHMEM/HCCL 通信。
此 CPU 适配器不更改公开接口的执行能力、设备能力门禁和 benchmark 原生能力检查。
本批公开接口的平台命名适配另见 [API 契约](api-contracts.md)。

通过本测试只表示功能桩和测试链路通过，不表示安装后的算法 API、高吞吐性能或低时延性能通过。
原生算法尚未实现的调用继续明确拒绝，不能自动退回功能桩。

## 可执行入口

本页命令是 CPU 快速回归。真实 NPU Tensor、最外层脚本选项及后续算子接入见
[NPU 集成测试](npu-roundtrip.md)，其中 `npu-stub` 与 `native` 共用本页场景/数值定义，
不通过本页 CPU 模拟器执行设备运算。

在源码或解包的 sdist 根目录运行，Python 3.11+，无需 Torch、NumPy 或设备依赖：

```bash
python -m tests.integration.run_roundtrip --backend functional-stub
python -m pytest tests/integration -q --durations=5
```

第一条命令输出 JSON；省略 `--backend`、指定 `native`、`auto` 或低时延模式均失败。
异常直接返回非零退出，不转成 skip。JSON 明确包含：

- `backend: functional-stub`、`execution: single-process-cpu-simulation`。
- `device_communication_verified: false`、`wheel_api_verified: false`。
- `device_reference_captured: false`、`second_reference: cpu-bf16-fp32-path-simulation`。
- 全部场景、逐 rank L2 指标、第二轮结果和整套运行耗时。

`suite_wall_seconds` 只是测试耗时，不是通信延迟；不输出带宽或伪造设备性能。
pytest 默认 CPU 回归会收集这些测试。现有打包验证脚本会把测试复制到安装环境执行，
但这仍然只是测试适配器验证，不能因此把 `wheel_api_verified` 改成 true。

## 模拟器支持域

仅 V1 普通布局、BF16 激活、可选 FP32 权重，alignment=1，两个模拟 rank，K=1/2。
测试规模上限为每 rank 64 个 token、H<=512、E<=64；expert 在 rank 间均分。
输入拒绝越界索引、错误 shape、非数值、NaN/Inf 及超出有限 dtype 范围的数据。
输入浮点数按明确的 BF16/FP32 位置转换；combine 输入须已可由 BF16 表示。
这些限制用于约束测试计算量，并非 Ascend 算子支持规格。

dispatch 按目标 rank 去重：同一 token 命中该 rank 的两个专家时，仅接收一行，
但每个有效专家槽位分别计数、分别计算。索引转换为本地 expert ID，非本地槽位为 -1。
空目标 rank、全 mask、零 token rank、重复 expert 槽位另有微型语义测试；不声明对应设备支持。

功能桩使用独立、不透明的 handle，不能传入公开 Buffer。每个实例只保留一轮待归并结果：
成功 fresh dispatch 使旧 handle 失效，成功 combine 消费 handle；跨实例、旧 handle、重复消费和销毁后调用均拒绝。
失败的参数校验不消费已有 handle。接收快照为不可变数据，新一轮不会覆写前一轮输出。
上述限制是**测试适配器**约定，不将其套到原生 cached/multi-handle API。
CPU 调用同步完成，不伪造 NPU Event，也不声称验证异步、通信流或 hook 完成语义。

## 专家计算、权重与精度

测试专家函数为 `f_e(x) = (e+1)/8 * x + (e-3)/64`。端到端数学答案为
每个有效槽位的 `gate * f_expert(x)` 之和，无权重时 gate=1。

实际测试路径只消费 dispatch 返回的数据、本地专家索引和权重：

1. 输入转 BF16，权重转 FP32。
2. 专家乘法/加法按 FP32 计算，专家结果转 BF16。
3. gate 乘法与本地累加按 FP32 计算，每个目标 rank 的 payload 转 BF16。
4. combine 对来自不同 rank 的 payload 做 FP32 累加，最终转 BF16。

gate 只在第 3 步应用一次。普通 combine 的 `topk_weights` 是独立值回传通道，
测试发送与 gate 不同的 `0.25 + 0.5 * gate`；非本地槽位置 777，检查归并时确实忽略它。

dispatch 有效载荷、路由、权重和计数精确比较。最终计算使用现有 [L2 标准及小值域规则](precision.md)，
**每个 rank 独立通过**。FP64 Golden 直接从原始输入生成，不读取 handle/接收数据；
第二标杆从原始输入独立模拟上述 dtype 路径，并明确标注不是独立设备 capture。
桩不调用路由 oracle；只共享基础浮点舍入工具，不共享分发/归并实现。
手算锚点以及故障注入分别验证专家公式、漏贡献、重复乘 gate、错回传、错源顺序、错 shape 和非有限值。

## 小型矩阵与耗时控制

默认 E=8、H=256/512、每 rank T=16/17，6 个具名场景：

| 场景 | 核验重点 |
| --- | --- |
| mixed_k1 | K=1，本地/远端路由 |
| cross_rank | K=2，跨目标 rank、非均匀权重 |
| same_rank | 同 rank 两个专家，rank 去重不丢专家贡献 |
| hotspot | 热点专家、空专家及空接收 rank |
| masked_zero | 部分/全部 -1、合法零权重 |
| fresh_iterations | 连续两轮改变输入和路由，不使用旧 handle |

不做全维度笛卡尔积、性能 warmup、大量重复或自动安装。
全矩阵在 pytest 中用模块级 fixture 运行一次，报告断言复用结果；CLI 单独做端到端检查。
故障和边界使用 H=1/4/16 的微型输入，不重复大形状。
源码测试、wheel 构建安装、NPU smoke 和 benchmark 继续分开，避免每个 case 重建/重装。

## 高吞吐与低时延的后续接入

固定上游提交 `a56d6156febcd9976e55adc85b5155bfac9f28f8`：

- [V1](https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/deep_ep/buffers/legacy.py)
  使用不同方法族；低时延使用按 expert 组织的容量缓冲区/有效计数，并有 recv hook 和缓冲复用约定。
  `low_latency_combine` 执行带权归并，不能直接把普通 combine 换个函数名。
- [V2](https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/README.md)
  已统一为 ElasticBuffer 接口；`do_expand` 是布局选择，不是低时延开关。
  Ascend 可在统一前端下使用不同执行实现，不要求共用同一个 kernel。

本批不提供低时延桩、自动算法选择、FP8、V2/cached、异步、GEMM 或训练反向。
后续顺序为：接真实 BF16 普通路径并验证 wheel API → 补低时延独立适配/契约测试 → 设备实测后设计模式选择。
真实后端须使用设备 Tensor、真实 handle/事件和实际通信；不得通过调用功能桩满足资格门槛。
