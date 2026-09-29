# 原生构建与运行时设计

## 范围与非目标

本批交付可构建、可安装、可检查来源的 `ascend-deepep` wheel，导入名保持 `deep_ep`。
运行时将框架事件与 SHMEM 资源分层管理。

当前实现 Host 资源、完整 Notify 和受限的 [V2 expanded dispatch / combine](elastic.md)。
ElasticBuffer 在同一 SHMEM session 中启用 MTE+URMA，legacy Buffer 保留 MTE 配置。
BF16/FP8、可选权重、cached dispatch 和 BF16 combine 已有执行路径；V1 算法、训练和
benchmark 实测适配器尚未接通。wheel 构建成功不代表设备精度或性能获得资格。

## 构建、安装与依赖

`PEP 517 → setuptools build_ext → 一次根 CMake → install 暂存树 → wheel`。
元数据和 sdist hook 不加载框架、不初始化设备、不配置 CMake。真正构建时才检查 SDK。
Host C++17 扩展使用独立 pybind11，不链接 PyTorch C++ ABI；框架流/事件经 Python API 访问。
Notify/dispatch/combine 使用独立的 bisheng `-xasc` target，Host 扩展保持普通 C++17。

- Python ≥3.11、Linux；当前编译基线 CANN 9.2.0，目标 Ascend950。
- Python 构建依赖和 SHMEM 来源写入 `dependencies.lock.json`。
- 普通（HT）dispatch/combine 的 shared-jetty 支持依赖 [SHMEM PR #851](https://gitcode.com/cann/shmem/pull/851)，当前锁定该 PR 分支的确定提交。
- CANN、SHMEM 由部署环境提供，不联网拉取、不重新分发驱动/SDK，不自动安装其他 `deep_ep` 包。
- SHMEM 提供方须确认固定提交的构建来源；`record_shmem_sdk.py` 检查固定公共头文件并记录 SDK 内容校验和。
  头文件校验不能证明二进制的源码来源，提供方声明与后续实测都必需。
- wheel 不携带 SDK；使用环境配置的动态库搜索路径。安装 RPATH 不含构建机绝对路径。
  加载后通过 `dladdr` 定位实际 SHMEM 库并比对构建时 SHA-256，防止同名旧库被加载。
- `_build_info.json` 记录版本、架构、SDK 校验和和真实能力，不写工作目录、主机名或账户。
- 不同 SDK/架构/Python 配置按指纹分开构建。Release 启用栈保护、FORTIFY、RELRO/NOW、不可执行栈。
- MegaMoE 开关只接受 OFF；其他 SoC/未验证工具链明确拒绝，不宣传兼容。

## 公开接口资格边界

| 接口 | 本批实现/限制 |
| --- | --- |
| Buffer 构造 | 显式进程组、`num_scaleup_bytes`、`explicitly_destroy=True`；只支持默认 MTE 资源配置，`num_scaleout_bytes` 非零仍拒绝 |
| ElasticBuffer 构造 | 进程组或显式 rank/world_size、`explicitly_destroy=True`；显式 num_bytes 或根据 T/H/K/dtype 给出容量预算 |
| destroy | 各 rank 同序显式调用；成功后幂等；不销毁调用方进程组/设备 |
| get_comm_stream | 返回框架创建的真实 NPU 通信流；销毁后拒绝 |
| capture | V1 返回 EventOverlap；V2 返回 EventHandle；真实记录当前流 |
| EventHandle/EventOverlap | 真实流等待、单次 hook、上下文退出等待、可选释放；空事件明确拒绝 |
| ElasticBuffer.barrier | 仅 `with_cpu_sync=True, sequential=True` 的主机同步屏障；默认异步模式拒绝 |
| ElasticBuffer.dispatch/combine/EPHandle/size hint | 见 [Elastic 文档](elastic.md)；元数据和 payload 输出有独立框架存储 |
| Config/拓扑/其他算法操作 | 保留签名及未实现错误 |

legacy Buffer 保留容量 2 MiB～1 GiB、规模 2～8 的限制；ElasticBuffer 工作区支持至 32 GiB，2 MiB 对齐，完整算子链路支持 EP2～128 的 2 的幂规模。
通信域不等同于具体传输引擎；`memory_transport` 开关仅指内存语义路径。QP 参数不用于当前 MTE 资源配置，也不统一表示各类 DMA 队列/通道；详见 [API 分层与后续扩展设计](api-contracts.md#通信域传输语义与资源配置分层)。
数值范围是资源准入边界，不是这些规模下的通信性能声明。
调用方必须先设置当前 NPU，所有使用/销毁保持创建时设备与 ACL context。
Gloo/HCCL 组仅用于初始化配置一致性检查；库不创建或销毁该组。

`DEEPEP_SHMEM_ENDPOINT=tcp://<IPv4>:<port>` 必須显式指定独立且未占用的端口，所有 rank 相同。
端口由部署/作业调度方分配；库不关闭其他服务，不使用固定默认端口，也不冒充安全网络配置。
SHMEM 初始化/创建/控制超时 1～300 秒；分布式进程组超时由调用方配置。
测试启动器另有进程级超时。底层 void 屏障接口不能提供额外错误码，异常作业应整体退出重启。

## 生命周期与失败策略

1. 未支持选项与尺寸预检；确认当前包自身扩展和外部 SDK 身份。
2. 各 rank 检查容量、超时、endpoint 摘要及已有资源占用，全部一致才进入 SHMEM。
3. 建立框架通信流；初始化独占 SHMEM 实例并分配、清零对称工作区。
4. 运行时只拥有自己的 SHMEM/工作区；不调用 `aclInit/aclFinalize/aclrtResetDevice`。
5. 显式销毁先等通信流完成，再作跨 rank 屏障、释放工作区并 finalize 自己的 SHMEM。

本批不支持同进程同时创建多个 Buffer，也不接管外部已初始化 SHMEM。
活跃 Runtime 由进程级 owner 持有；忘记 destroy 只警告并保留，不在 Python GC/解释器退出时偷偷做 collective。
这是一项显式限制，不是自动清理成功。创建失败/清理失败不伪装成功；不确定 SDK 状态下保留占用并要求作业退出。
各 rank 必须同序调用构造、屏障、销毁；局部参数错误/进程退出不可能由该层自动恢复整个分布式作业。

事件等待只向当前流添加依赖，不等于 CPU 同步。释放 extra_tensors 前在消费者流上调用 `record_stream`，
防止 allocator 在消费者完成前复用内存。hook 仅在等待成功后执行一次；等待失败保留资源与 hook。

## 测试与合入门槛

| 层次 | 验证内容 | 不能替代 |
| --- | --- | --- |
| source CPU | 全部签名、所有未实现失败路径、精度/reference/benchmark 回归、运行时故障注入 | 原生编译/设备资格 |
| 打包 | metadata 不触发构建、wheel 内容、sdist 重建、构建信息一致性 | NPU 功能 |
| 隔离安装 | 新 venv、源码目录外运行、Python/扩展实际路径、SDK 身份、导入不初始化设备 | 通信正确性 |
| 真实 NPU | 跨流数据依赖、hook/释放、V1/V2 创建销毁、重复创建、重复 destroy、并发 owner 拒绝、rank 配置冲突、保留调用方 PG | dispatch/combine 数值或带宽 |

`scripts/validate_package.py` 串联构建、sdist 重建、安装和测试；设备部分必须显式选择设备和 rank 数。
默认 pytest 不安装依赖、不占设备、不把跳过算通过。原有通信矩阵仍保留，等后续真实算法逐项启用。
缺少 SDK/框架/设备的必选验证必须失败，不能降级成纯 Python wheel 或空测试通过。

### 验证记录

开发包 `0.1.0.dev4` 的验证配套为 Linux aarch64、Python 3.11、GNU C++ 12.3.1、CANN 9.2.0、
PyTorch/torch_npu 2.7.1 配套、Ascend950 和锁定的 SHMEM SDK。
仅以下明确列出的资源/事件能力获得本批验证，不能扩展为 EP 算法资格。

| 验证项 | 结果 |
| --- | --- |
| 固定 API | 6 个类型、39 个选定接口的签名基线；未支持操作保持失败边界 |
| Linux 源码测试，要求框架导入 | 402 passed、251 skipped；skip 为尚未实现的设备通信用例 |
| wheel 安装后的测试 | 358 passed、283 skipped；含不再适用的源码/桩断言及未实现通信矩阵，不将其计为设备通过 |
| Windows 无 SDK 回归 | 393 passed、259 skipped、1 deselected；包含 8 个仅 Linux 的原生故障注入编译测试跳过 |
| 构建、sdist 重建、安装 | wheel/sdist 成功，重建能力/源码身份一致，安装包与扩展路径独立于源码目录，离线依赖安装通过 |
| HCCL 初始化组 | EP2、EP4、EP8 的全部 rank 正常退出；每 rank 4 次 V1/V2 资源生命周期 |
| Gloo 初始化组 | EP2 的全部 rank 正常退出，同样执行 4 次生命周期 |
| 真实事件/释放 | 跨流结果、hook、context-manager、待完成流工作、销毁后调用方 Tensor 存活、重复 destroy 通过 |
| 失败边界 | SDK 身份不匹配、外部占用、初始化/分配/清理失败、rank 配置冲突及不支持选项检查通过 |
| ELF | 无构建机 RPATH；RELRO、NOW、不可执行栈；仅链接所需系统/SDK 库，不链接 Torch C++ 库 |

可复现命令见[构建指南](build.md)。框架依赖通过显式选择 `--system-site-packages` 复用，
DeepEP 与固定构建/测试依赖安装在新 venv；不将此表描述为完全无系统依赖的容器验收。
CI 已接 CPU 门禁；原生/NPU CI runner 待维护者配置，不能将本表设备验证写成 CI 已自动通过。
其他 SoC、框架配套、跨节点/多实例、自动 GC 清理、算法容量提示及 EP 数据通路仍未获得资格。
