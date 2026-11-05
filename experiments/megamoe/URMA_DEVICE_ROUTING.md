# MegaMoE 内核内 Device 路由

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
2. 56 个 AIV 静态平分扁平 topk 输入。向量 Compare/GatherMask 统计每块
   的专家数量；向量 Add 形成跨块前缀，生成本卡专家排序偏移。
3. 每个目标 rank 收到一条元数据：目标局部专家的计数前缀，加来源卡
   的 expanded 起始偏移。EP16 每条有效数据为 108B，使用 512B 邮箱槽；
   另有 epoch 标志。不广播完整 topk_idx，不传 token 编号。
4. 接收端根据计数生成实际 M256 分组、source/CV 分段、Pack 轮次、
   GMM 任务序列、Combine 依赖。首版由一个 AIV 编译紧凑描述符；
   这里处理的是计数/区间元数据，不是逐 token payload。
5. 56 个 AIV 再次向量筛选自己的输入块，生成稳定的 expert-major
   order、token-major inverse 和 Reduce 依赖，然后进入原有 Pack/计算流程。

跨卡元数据采用 MTE DataCopy 的写语义；未引入 GET。
所有前置局部 join 和元数据交换均属于 `RoutePrepare` 时间。
共享 GMM1 在此预处理完成后启动；首版不尝试与路由预处理重叠。

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
