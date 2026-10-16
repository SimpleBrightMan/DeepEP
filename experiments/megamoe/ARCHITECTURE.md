# MegaMoE 工程设计与维护指南

本次整理以 `18ada75b1` 为基线。目标是减少概念与维护成本，不是缩短所有函数或改变调度算法。
硬件完成协议、tile/wave 预算、GMM2 lag、缓存策略、权重布局和数值契约必须单独评审、单独验收。

## 1. 支持范围

- 对外继续支持 `fp8xfp4`（W4A8）和 `fp8xfp8`（A8W8），输入可为已量化 FP8 或接口支持的 BF16。
- 通信固定为 MTE、非量化 BF16 Combine；保留输入量化和 SwiGLU 后的 MX 量化。
- W4A8 保留共享专家、gate/up 配对、timer 和 compute replay；replay 是诊断，不生成新的完整 MoE 输出。
- 删除 URMA/layered、量化 Combine 和 top-k weight prefetch 的不可达执行逻辑，不提供自动 fallback。
- A8W8 的 MX 计算独立放在 `stage/mega_moe_gmm{1,2}_a8w8.hpp`，不是待删除的 legacy。
- tiling 仍保持 224B ABI；保留的协议字段不代表仍支持相应模式。新增模式不能仅设置一个旧字段就启用。

## 2. 分层：每一层只回答一种问题

```text
Python API             资源归谁，输入是否合法，本次执行哪个 BS？
  → Host Plan          如何派生配置、申请容量并启动设备 kernel？
    → Device 调度      哪个角色执行哪个专家、wave、slice？
      → Stage          本任务读写哪些地址，等待/发布什么依赖？
        → CATLASS      物理缓冲怎样轮转，指令和硬件 credit 怎样推进？
```

禁止下层反过来决定上层 wave 策略。函数返回只表示该角色执行到这里，不能替代其他引擎的数据完成事件。

## 3. 文件职责与阅读顺序

以下路径相对 `experiments/megamoe`。

| 文件 | 主要职责 / 先读的函数 |
|---|---|
| `python/deep_ep_experimental/megamoe/api.py` | `SymmBuffer` 持有资源；`prepare/_prepare` 维护当前 BS 的 Plan；`_run` 调用原生层；`transform_weights_for_mega_moe` 准备物理权重布局 |
| 同目录 `validation.py` | `check_routing/collective_check` 在进入设备通信前共同报告错误 |
| `csrc/bindings.cpp` | Python/原生协议检查、Plan 绑定和常量导出 |
| `csrc/forward.hpp/.cpp` | `MakeTiling` 派生配置；`Plan::Plan` 分配资源并映射 peer；`Plan::forward` 校验和启动 |
| `csrc/kernel_launch.hpp` | 原生启动边界，不组织设备算法 |
| `csrc/kernels/mega_moe/mega_moe.cpp` | 注册的模板实例与启动选择，是“实际运行什么”的事实源 |
| 同目录 `mega_moe_tiling.hpp` | Host/Device 配置 ABI 与缓冲配置结构 |
| 同目录 `mega_moe_arch35.hpp` | W4 公共输入、共享专家和输出阶段；`ProcessWave` 是总入口 |
| 同目录 `mega_moe_wave_a8w4.hpp` | 角色分流、wave/slice 推进、AIV1 就绪驱动调度 |
| 同目录 `mega_moe_wave_a8w8.hpp` | A8W8 独立调度，不把它当作 W4 开关分支阅读 |
| `common/mega_moe_wave_schedule.hpp` | 无搬运副作用的范围推进与 wave 边界计算 |
| `common/mega_moe_shared_policy.hpp` | 输入协议、共享专家策略选择；策略不是 ready 信号 |
| `common/mega_moe_types.hpp` | Context、地址视图与任务分工类型，不实现流水 |
| `common/mega_moe_workspace.hpp` | 本地 GM 区域与同步区布局、Reset 元素数的唯一来源 |
| `common/mega_moe_peermem.hpp` | 对称窗口容量及 peer 内地址布局 |
| `common/mega_moe_peer_rows.hpp` | Dispatch 的 peer/行段选择，不发起 DMA |
| `common/mega_moe_block_scheduler.hpp` | GEMM tile 编号与分核辅助 |
| `common/mega_moe_gmm_common.hpp` | 计算配置与布局类型；不决定外层 wave 顺序 |
| `common/gmm1_async_completion.hpp` | GMM1 FIX full/empty 与消费者 credit 协议 |
| `stage/mega_moe_token_quant.hpp` | 输入量化/通信记录准备；FP8 输入仍需正确打包 |
| `stage/mega_moe_send_mask.hpp`、`mega_moe_route_compact.hpp`、`mega_moe_count_arrival.hpp` | 路由元数据、count 与发布/接收协议；文件名 mask 不等于运行 URMA |
| `stage/mega_moe_token_dispatch.hpp` | 绑定本核目标行、读取 peer token/scale、落地后发布 GMM1 ready |
| `stage/mega_moe_gmm1_activation.hpp` | W4 GMM1 计算、权重 Prologue、激活辅助与地址绑定 |
| `stage/mega_moe_gmm2_combine.hpp` | W4 GMM2、BF16 token Combine 与就绪辅助 |
| `stage/mega_moe_gmm1_a8w8.hpp`、`mega_moe_gmm2_a8w8.hpp` | A8W8 MX 计算，不再挤在 W4 主文件中 |
| `stage/mega_moe_shared_expert_input.hpp` | 共享专家输入准备 |
| `stage/mega_moe_workspace_reset.hpp` | 每次调用的同步状态清理；必须与真实分配同源 |
| `stage/mega_moe_unpermute.hpp`、`mega_moe_unpermute_fused.hpp` | 按源 token 聚合专家结果 |
| `catlass_gmm/mega_moe_grouped_matmul.hpp` | 持久物理流水，`Begin/DrainOutput/End/HandoffTo` 生命周期 |
| `catlass_gmm/block_prologue_a8w4.hpp` | AIV0：FP4→FP8/NZ→L1 权重供数 |
| `catlass_gmm/block_mmad_a8w4.hpp` | AIC：L1/L0 供数、MMAD 与 FIX |
| `catlass_gmm/mega_moe_mx_matmul.hpp`、`block_mmad_mx.hpp` | A8W8 的 MX 计算适配与物理执行 |
| `catlass_gmm/a8w4_k_plan.hpp`、`resource.hpp` | 共用资源布局与 K 计划；AIC/AIV0 必须一致 |
| `epilogue/` | SwiGLU、MX 量化及其 UB 布局；不决定 Dispatch 优先级 |
| `common/ascend_timer_v2*`、Python `timer/` | 采集/解析协议；不为计时额外排空流水 |
| `mc2_common/` | 运行上下文、基础同步和异常定位辅助 |
| Python `benchmark.py/performance.py` | 输入生成与测量；完整通信、replay、timer 必须区分 |
| Python `reference.py/golden.py` | 独立数值参考、Golden 缓存和比较 |
| `scripts/build.py/batch_launch.py/batch_node.py` | 构建及单机/多机启动，不在这里选择设备调度算法 |
| `tests/` | 实际生产函数的 CPU 行为回归，以及 API、打包、测量协议测试 |

## 4. W4 主调度与核心函数

`ProcessWave` 连接输入准备、可选共享专家 GMM1、路由计算、共享专家尾部和输出。
`ProcessMoeExpertStages` 准备 count 并分流角色：AIC/AIV0 进入 `ProcessGmmWaves`，AIV1 进入
`ProcessAiv1PriorityStages`。这不是三条串行阶段。

| 函数 | 输入 → 输出 / 责任 |
|---|---|
| `RunGmm1Wave` | 下一位置 cursor → 固定 wave 边界；按预算消费 slice |
| `RunNextGmm1Slice` | wave 当前位置、专家总行数 → 当前 slice，并推进 wave.end |
| `RunGmm1ActivationForExpert` | 完整 ExpertContext + 只读 slice → 绑定地址并提交 GMM1/Prologue |
| `RunGmm1A8W4` | shape、地址、分核状态、可选强类型流水 → 本核 tile 执行及 rolling owner 更新 |
| `RunGmm1ComputeTiles` | 已准备配置与 scheduler → AIC tile 循环或 AIV0 权重循环 |
| `RunGmm2CombineForWaveRange` | 固定 wave → 对应 GMM2，真实 FIX 完成后统一发布 ready，再交还流水 |
| `RunGmm2CombineForExpert` | 一个只读 slice → 地址绑定和 GMM2 提交；不提前发布完成 |
| `ProcessAiv1PriorityStages` | 多个任务 cursor → 按就绪情况推进；路由激活优先，其次 Dispatch、共享激活、Combine |
| `TryRunReadyActivation` | 可提交序号 + 真实 FIX full → 激活输出及 empty credit |
| `DispatchNextExpertSlice` | 预取 cursor → 下一个 slice 的本地输入；不代表当前 wave 计算结束 |
| `PrepareMoeExpertTokenCountTable` | 接收 count → 本卡专家计数及来源前缀；用于规划地址，不搬 token 数据 |
| `FinishDispatchOwnedRows` | 本 owner 的剩余路由批次和拷贝 → Drain 后发布目标行贡献 |
| `PublishGmm1TileReady` | 本 owner 已落地行区间 → 对对应 M-group 累加实际行数贡献 |

保留原有小 BS 即时 GMM2、大 BS lag 一个 wave 的条件。GMM1 正向与 GMM2 反向分核分别保留
rolling owner；空任务也要按协议推进，不能把“本核没活”当作“所有核没活”。

## 5. 数据类型与内存视图

- `ExpertContext`：完整专家的形状和本卡紧凑序列起点；M 不是当前 slice 行数，N/K 使用 GMM1 约定。
- `ExpertSlice`：本次专家编号、专家内起始行、行数、全局起始行；确定后只读。
- `ExpertTokenPosition`：下一规划位置；允许推进，不应再用于寻址已经提交的任务。
- `ExpertWaveRange`：跨专家的固定半开范围，供 GMM2 重放。
- `BlockJobContext`：计算任务分工；`BlockWorkspaceContext`：count/flag 物理分区，不能因值相同就混为一个类型。
- `GMMAddrInfo`：一次执行的地址视图，不拥有 GM；持久流水使用具体模板类型，不能通过 void* 猜类型。

记 H 为 token hidden，I 为激活后专家中间维，R 为 `maxOutputSize`。本地主要区域：

| 区域 | 逻辑容量 / 作用 |
|---|---|
| Dispatch data / scale | `[R,H]` FP8 及其物理 MX scale；接收方读远端输入后落地 |
| metadata | 每接收行 32B，记录回传所需源 rank/token/top-k |
| GMM1 result（W4） | `[R,2I]` BF16，AIC 写，AIV1 激活读取 |
| activation data / scale | `[R,I]` FP8 及 MX scale，GMM2 输入 |
| GMM2 result | `[R,H]` BF16，Combine 读取 |
| 连续同步区 | 行数贡献、激活 ready、GMM1 序号、GMM2 slice ready 及共享专家协议 |
| 共享专家区域 | 按共享专家、BS、H/2I/I 分配，不与 routed 生命周期混用 |

shape 是逻辑视图；padding、scale 打包和实际偏移以布局文件为准。R 是保守容量上界，不是平均接收量。
对称窗口另有输入记录、compact routes/count、可选 decode rail 区、BF16 Combine 接收区。
本地 workspace 和对称窗口不是同一块内存，不得用平均 BS×TopK 替代最坏容量校验。

Host 的 Reset UB 预算直接读取 `WorkspaceInfo.flagResetElementCount`，不再单独维护 HostFlagElements 公式。
本次应保持支持配置下区域偏移与基线一致；未启用模式的空指针成员删除不改变 GM 分配。

## 6. 同步契约：提交、完成、可见、可复用

| 生产者 → 消费者 | 必须保证的边界 |
|---|---|
| 输入准备 → peer Dispatch | 输入记录落地后才发布本轮协议标记 |
| Dispatch → AIC GMM1 | 本 owner 所有相关 MTE 拷贝完成后，发布 M-group 实际行数贡献 |
| AIV0 → AIC | 权重写入 L1 后 full；AIC 用完后 empty，才可覆盖对应槽 |
| AIC GMM1 → AIV1 SwiGLU | GM issued 仅用于挑选任务；消费者仍必须等待真正 FIX full |
| AIV1 → GMM1 producer | 最后一次依赖读取完成后归还 empty，不因 Scalar 已推进而提前复用 |
| SwiGLU → GMM2 | 量化数据及 scale 完成后发布输入 ready |
| GMM2 → Combine | wave owner 先 DrainOutput，实际生产核再发布 slice ready |
| Combine → 输出归并 | 保留通信完成及跨 rank 输出协议，不能以发送指令发出代替远端数据可读 |

`HandoffTo` 转移游标和 credit 的唯一 owner，不是结束流水，也不是完成通知。
`DrainOutput` 等输出完成但保留资源；`End` 归还最终资源。三者不可互换。
共享专家沿用自己的生命周期；不得为了统一接口把 routed handoff 强加给共享路径。

`Gmm1AsyncCompletion` 的函数与引擎对应：`InitConsumer` 归还初始 empty；
`AcquireProducer` 在 FIX 队列等待 empty；`PublishCompleted` 在 FIX 队列发布 full；
`WaitConsumer` 在 MTE2 等 full；`ReleaseConsumer` 在 MTE3 归还 empty；
`DrainProducer` 消耗最终 credit 并完成收尾。GM issued 序号不能取代这条链。

## 7. 简洁与可维护性约定

1. 函数保留的理由必须是独立责任、真实复用或资源生命周期；不增加纯参数转发层。
2. 一份配置、一处派生。布局、flag 数量、缓冲预算必须共享事实源。
3. 调度只看任务范围，Stage 绑定数据依赖，物理执行管理引擎和缓冲；禁止互相穿透决策。
4. 不把可推导状态重复保存为 bool。确有独立历史状态（credit、已提交序号、待消费任务）则明确保留。
5. 中文注释解释原因、约束、同步和例外；不逐行翻译，也不写未测的性能承诺。
6. 不支持的模式不做静默 fallback；公开接口、模板约束、文档和测试必须一致。
7. 新增模式先说明实际用户与数值/容量/生命周期契约，再决定是否需要抽象；不恢复一个万能兼容函数。
8. 错误和诊断需要上下文，但热循环中新加 timeout/计时/打印要单独测量，不能借整理代码改变同步开销。

## 8. 验证与交付要求

- CPU：范围推进、空专家/尾块、分核、Dispatch Drain-before-ready、角色生命周期、地址布局和协议解析。
- 编译：完整原生构建必须覆盖实际 W4 配对/非配对以及 A8W8 实例，不能只验证抽取的函数体。
- 设备：独立输出/缓存，固定源码、wheel、SDK、输入；覆盖 BF16/FP8、随机/偏斜路由、共享专家和 BS 切换。
- 性能：与冻结基线同窗口交替对照、timer-off 测完整通信；独立 timer 只定位流水。
- 输出等价不是独立 Golden；CPU 模型不是 DMA/跨核可见性证明；EP8 不外推为 EP128 通过。
- 本文描述设计意图，实际本轮通过/未运行项目必须记录在单独验证报告中，不继承旧报告的 PASS。
