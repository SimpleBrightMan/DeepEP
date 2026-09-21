# NPU Tensor 集成测试与算子接入

本入口使用真实 NPU Tensor，运行 dispatch → 专家计算/加权 → combine → 逐 rank L2 精度检查。
输入、接收数据、专家计算、combine 输入和最终输出都检查 NPU device、dtype 和 shape。
CPU 仅用于确定性输入生成、独立 Golden 和结果核对；不使用 CPU 功能桩代替设备执行。

## 两种明确分开的后端

| 选项 | 被测执行路径 | 成功表示 |
| --- | --- | --- |
| `npu-stub` | NPU 基础算子 + HCCL AllGather/AllReduce，模拟普通布局分发/归并 | NPU Tensor 功能链路通过；不是 DeepEP 原生算子资格 |
| `native` | 安装后的 `deep_ep.Buffer` 公开接口；Gloo 只做运行时 CPU 元信息协调 | 所选原生 dispatch/combine 用例通过；不证明性能或未选场景 |

HCCL 在这里是 **npu-stub 的显式通信实现**，不加入 DeepEP 原生算法的强制依赖。
两种模式都不回退到 CPU 或其他通信实现。native 遇到未实现、缺能力、未安装、未返回真实完成事件或精度错误会失败。
目前 Host-only wheel 尚未声明三个原生算法，native 会在设置 NPU/分配 SHMEM 前拒绝；不能修改元数据制造“通过”。

## 最外层：构建、安装、运行一次完成

先由调度方预留两个健康设备，准备匹配的 torch/torch_npu、CANN 和 SHMEM SDK。
设备编号和路径为示例，须替换为本作业资源；输出目录必须全新且位于源码目录外。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1

# 构建/隔离安装后执行 NPU Tensor 功能桩，结果不记为原生 EP 通过。
python scripts/validate_package.py --output /path/to/new-validation \
  --system-site-packages --ep-roundtrip npu-stub

# 后续原生算子接通后：同一组数据、同一专家处理和精度检查。
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:19091
python scripts/validate_package.py --output /path/to/new-native-validation \
  --system-site-packages --ep-roundtrip native
```

可继续加 `--wheelhouse /path/to/wheels` 离线安装。
`--runtime-ranks 2` 是独立的资源/事件探针，不等于算法验证；可以与 `--ep-roundtrip` 同时启用。
本批 EP 集成固定双卡，不能与 `--runtime-ranks 4/8` 混用同一个设备列表。
不加 `--ep-roundtrip` 时保留原来的无设备默认行为。

## 开发迭代：不必每次重新构建和安装

在已经安装候选 wheel 的 Python 环境中，直接运行外层检查脚本：

```bash
python scripts/validate_ep.py --backend native --devices 0,1 \
  --case mixed_k1 --output /path/to/new-single-case

python scripts/validate_ep.py --backend native --devices 0,1 \
  --output /path/to/new-full-suite

# 算子尚未接入时验证测试链路：无需安装 DeepEP wheel，但需要真实 NPU/框架/HCCL。
python scripts/validate_ep.py --backend npu-stub --devices 0,1 \
  --output /path/to/new-stub-suite
```

脚本自动复制测试到源码目录外，清理 PYTHONPATH，使用当前 Python 启动 `torchrun`，不安装依赖、不重建、不自动选卡、不重置设备。
native 检查 package/扩展属于当前 Python 的安装目录，并核对两 rank 的构建身份相同。
省略 `--case` 运行 6 个小场景（含 fresh 连续两轮），不是大矩阵或 benchmark；详见[场景与数值约定](functional-roundtrip.md)。
`--case` 支持 `mixed_k1/cross_rank/same_rank/hotspot/masked_zero/fresh_iterations`。

npu-stub 一次启动运行全部场景；native 暂按每个 case 独立启动，避免未经资格验证就反复初始化 SHMEM。
同一个 case 内只创建一个 Buffer，fresh_iterations 在该 Buffer 内运行两轮；这不声明 cached 或多 handle 支持。
安装包和测试副本只准备一次。后续验证原生工作区复用安全后再优化跨 case 批处理，不删除安全门禁换取速度。
`--timeout` 默认每次 torchrun 600 秒，进程组通信超时 120 秒；禁止 Python `-O` 和 `PYTHONOPTIMIZE` 去掉精度断言。

## 算子开发者需要对齐的契约

不需要重写集成用例或导入自己的私有 kernel。补齐并保持以下公开接口及返回语义：

1. `Buffer.get_dispatch_config/get_combine_config` 与对应工作区容量计算。
2. `Buffer.get_dispatch_layout`、`Buffer.dispatch`、`Buffer.combine`；普通 rank 去重布局、BF16 激活、int64 路由及 FP32 值通道。
3. 上述调用的 `async_finish=True` 返回真实完成事件，`current_stream_wait()` 建立读取依赖。
4. buffer 创建、handle 正确关联、两轮 fresh 调用以及销毁；不能依赖测试端重造正确结果。
5. 构建信息 `communication_operations` 仅在实际接通后声明
   `Buffer.get_dispatch_layout`、`Buffer.dispatch`、`Buffer.combine`；同步更新相关运行时/未实现契约回归。

测试专家函数在真实 recv_x 上执行，普通 combine 不再给 token 乘 gate；独立 FP32 值回传另行核对。
当前第二标杆仍为明确标注的 CPU BF16/FP32 路径模拟，不是独立设备实采结果。
新接入 kernel 的每 rank 精度必须通过现有 L2 和小值域检查；不能因为换成 native 就调宽阈值。

## 结果、失败与 CI

每个 case/job 保留 torchrun 日志和 `rank-0.json`、`rank-1.json`。
只在所有被选场景、两个 rank、两轮（如适用）、真实事件等待及资源销毁均成功后生成 `summary.json`。
外层验证身份、完整场景列表、NPU Tensor 检查和逐 rank 精度，缺失、失败或 skip 不算通过。

- npu-stub：`npu_tensor_roundtrip_tested=true`，`native_ep_verified=false`，`wheel_api_verified=false`。
- native：本次成功后 `native_ep_verified=true`、`wheel_api_verified=true`，保留构建身份与选择的场景范围。
- 不输出桩带宽；总耗时只是测试耗时，不是算子性能。

原生/NPU CI runner 仍需维护者配置真实 SDK、框架和已预留设备。
配置后可直接执行上述命令并归档结果，不需要另写 CI 专用测试逻辑。
现有 CPU CI 只测试启动器、适配契约、报告门禁和故障传播，不把这些自检记为 NPU 执行。

## 本批已执行的验证

2026-09-21：Ascend950 单机双卡，torch 2.7.1+cpu + torch_npu 2.7.1.post5.dev20260724，
先使用独立入口，再通过最外层 `--runtime-ranks 2 --ep-roundtrip npu-stub` 执行全部 6 个场景。两个 rank 均通过，
fresh_iterations 的第二轮也均通过，测试进程正常退出，完成后无本次占卡进程。
独立入口首次全套耗时约 55.7 秒，包含框架/通信初始化、输入准备和精度检查，不是通信算子延迟。
`torch` 版本中的 `+cpu` 是基础框架构建标识；本次 Tensor 的设备为 NPU，由 torch_npu 执行。

该结果只确认真实 NPU Tensor/HCCL 功能桩链路，不确认 DeepEP 原生算法、性能或其他平台资格。
最外层已重新完成 wheel/sdist 构建、sdist 重建、离线依赖安装、全新 venv 安装、
源码目录外导入检查、安装后回归，以及双卡资源/事件探针和 NPU Tensor 闭环测试。
双卡探针中 V1/V2 资源共经历四次生命周期，真实事件依赖与销毁后状态检查通过。
一次闭环运行因 worker 收到 SIGTERM 中断，未计为成功且未生成成功摘要；重新执行完整流程后通过。
最终安装的 Host-only wheel 另执行 native 单场景准入检查：两个 rank 均因缺少上述三个 EP 操作
在设置设备前报未实现，外层返回非零且未生成成功摘要；这是失败门禁验证，不是原生算法通过。
