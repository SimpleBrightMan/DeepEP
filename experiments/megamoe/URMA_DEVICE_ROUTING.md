# MegaMoE 内核内 Device 路由

> 当前 commfirst 分支已升级为实验性的 protocol6：见
> [专家唯一 owner 与按需描述符](URMA_EXPERT_OWNER_ROUTING.md)。
> 下文保留 protocol3 的设计和已执行验证记录；这些编译/EP8结果不适用于当前实验版本。

本修改适用于 `megamoe-commfirst-shared` 和 `megamoe-unitflag-shared`。
生产接口只有 Device 路由，不提供 Host 后端或自动回退。

## 调用契约

`topk_idx` 是当前 NPU 上连续的 int32/int64 `[BS, topk]`。
调用者保证专家编号合法，内核不检查编号范围。
H/I/TopK/EP/BS 的静态形状、类型和容量检查保留。

`prepare_routes(topk_idx)` 现在只绑定输入和预分配的工作区，不读取值，
不 sort/count，不 D2H，不通过 TCPStore 交换路由。
`forward` 和 `forward_prepared` 每次均在同一次 MegaMoE kernel 内重建路由。
输入可在两次调用之间改变；调用方按正常 stream 依赖保证写入先于执行。
同一 Buffer 的不同 BS 绑定复用工作区，只能通过该 Buffer 的串行执行流使用。

## Pack 前的执行顺序

1. 使用独立、单调 epoch 的 MTE 启动屏障，防止覆盖上一轮元数据。
2. 56 个 AIV 静态平分扁平 topk 输入，每核仅从 GM 载入一次。
   单 VF 内 Compare/Compact 同时生成计数与稳定 CSR，保留在高 UB；
   计数整行 DMA 写出，再生成跨 worker/专家的全局前缀。
3. 每个目标 rank 收到目标局部专家的计数前缀和来源 expanded 起始偏移。
   EP16 每条有效数据为 108B，使用 512B 邮箱槽，另有 epoch 标志。
   不广播完整 topk_idx，不传 token 编号。
4. leader 生成小 summary，56 个 AIV 静态计算 M-group/CV 描述符长度；
   prefix 确定独占写入区后，并行生成 Fragment/Segment、Send/Recv、
   Link、Pack 表，经过分表 UB 缓冲 DMA 写出。小规模任务表由 leader 生成。
   unitflag 的 Reorder ranges/spans 同样按 CV 的连续行顺序生成。
5. 从保留的 CSR 构造 order、inverse 与 Reduce 依赖，不再重读 topk。
   power-of-two CV 使用向量 Gather/Scatter 和精确整数位移；其他 CV
   保留原 Scalar 计算路径。完成全部 UB 输出后批量 DMA，并统一等待。
6. Config 每次读取前按真实 64B DCache 行清理/失效，然后普通 volatile
   读取；保留必要的三次 reload。共9次本卡同步，所有零任务核也参与。

跨卡元数据采用 MTE DataCopy 的写语义；未引入 GET。
所有前置局部 join 和元数据交换均属于 `RoutePrepare` 时间。
共享 GMM1 在此预处理完成后启动；保持原有 Pack/GMM/通信调度。
commfirst 的 EP4 使用 wave12；unitflag 仅共享开启的 EP4 使用 wave12，S0仍保持专家lookahead。

## 接收地址和返回地址

每个来源固定预留 `source_pitch = max_tokens_per_rank * topk` 行：

```text
receiver_row = source_rank * source_pitch + row_within_source_to_receiver
return_row   = source_expanded_base_for_receiver + row_within_source_to_receiver
```

因此发送端不需要等待接收端回传 prefix。实际计算仍只覆盖真实收到的行。
GMM 输入按专家顺序跨来源 gather；GMM2/Combine 使用上述固定接收布局。
commfirst 本卡结果直接写最终返回区，Reduce 等对应 GMM2 group；
unitflag 本卡结果必须完成 Reorder，Reduce 等对应 self CV，不能仅等 AIC。
unitflag 的 Reorder span 按目标行递增，不使用无序 atomic append。

## 容量与诊断

接收和中间缓冲按 `EP * max_tokens_per_rank * topk` 行分配。
`max_recv_rows` 若显式传入，必须等于这个最大值；不再允许用均衡容量
掩盖热点路由溢出。H/I/EP/BS 较大时内存需求明显增加，必须有足够 HBM。
工作区只分配一次，不根据 Device 路由值重新申请。
优化版在 RouteInfo 尾部预留并行前缀 scratch；routing protocol 升为3，
必须重编完整 wheel，不能仅替换 kernel `.so`。路由 UB 最高182816B，
完成所有 DMA 和末尾同步后才交给 Pack/反量化/Reorder 复用。

`native.route_stats()` 和 `native.debug_snapshot()` 是显式的执行后诊断，
会同步并 D2H；生产 forward 不调用它们。旧 NumPy/C++ 规划器仅供测试
与 benchmark 精度对照，无生产 API 入口。

新增 trace event 62 `RoutePrepare`，粗细模式均记录；纯 profile 的融合核
时间也包含该预处理。原有计算/通信计时类别不包含路由准备。

## 验证

`tests/device_route_plan_test.cpp` 将 Device 描述符编译算法与独立旧规划器
对照：EP4/8/16、BS1/1024/1536/8192/16384、TopK1/6/8、CV128/256/1024，
覆盖均衡、随机、全热点、空来源、极端偏斜，共 675 组、每组逐 rank 检查。
检查 M-group、分段地址、返回地址、CV 依赖、Reorder 顺序和容量上界。
CPU 描述符对照与编译通过均不等价于 NPU 精度验证。

`tests/parallel_route_plan_test.cpp` 另覆盖S0/S1、EP4/8/16、CV384、
零任务worker及最大容量，共1800组逐rank对照；unitflag的Reorder表
逐字段比较独立旧规划器，保留本卡CV完成依赖。
实验参考与独立RoutePrepare结果见个人仓库`megamoe-routeprepare-lab`分支；
其独立核加速比不代表完整融合算子加速比。

2026-11-05 生产集成检查：两个分支均通过 CANN 编译，以及 EP8、
BS1024/8192、H5120/I2304/E384/K6、random-balanced、CV256、
共享专家1/I2304的完整算子运行；每个 rank 的最终 BF16 输出与各自
修改前版本逐位一致。该检查包含 profile/coarse/fine 导出，但仅用
warmup1/iters2、exp0验证功能，不用于性能结论，也不替代独立 Golden。
共享关闭的追加 NPU 回归因空闲门禁发现占用未启动；S0描述符有CPU对照覆盖。
