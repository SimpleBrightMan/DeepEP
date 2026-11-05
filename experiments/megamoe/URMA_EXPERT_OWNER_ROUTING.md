# Unitflag shared：专家分工路由与异步摘要（protocol6）

把 `megamoe-commfirst-shared@3a7ab63` 的头部方案移入本分支。
仅做源码移植和静态审阅，按用户要求未编译、未运行CPU/NPU测试；
commfirst的BS8192头部实测结果不能直接作为本分支的结果。

## 头部与角色衔接

- 56个AIV均分384个全局专家，每个专家只有一个owner；每批4096个TopK条目。
- VF生成稳定的专家flat-ID列表和计数，保留重复专家/不同topk槽的独立条目。
- 固定专家桶为int32 `[384, maxBS*TopK]`，容量不依赖均衡。
  maxBS8192/K8预留96MiB、maxBS16384/K8预留192MiB，实际只写命中的索引。
- leader只计算384项本地前缀。跨rank仍使用MTE发送计数和epoch，不增加GET。
- 路由下发后直接Pack/Dispatch，不等待接收摘要；helper0非阻塞检查来源epoch，
  未收齐则继续原静态Pack配额。收齐后生成接收前缀和小计算顺序表，发布本卡ready。
- AIC/AIV0获取ready后计算；通信核完成原Dispatch/Pack阶段后获取摘要再进入Reorder/Combine。
- outgoing前缀与接收摘要隔离到不同64B缓存行；仅写后发布的尾部区域，消费者获取后失效缓存。
- 路由仅保留2次本卡SyncAll；入口RankBarrier删除。单槽邮箱复用依赖算子自身的
  所有peer finished观察、空peer通知、最终quiet、尾部本卡join及owner stream串行，
  不依赖benchmark all-reduce。

## 按需描述符与逆映射

不再在头部生成Fragment/Segment、Send/Recv、PackRound、Reorder Range/Span大表。
GMM输入gather和scale gather按当前M256组顺序推导source/CV行段。
Pack从专家桶读取flat ID，并用8B MTE写逆映射；96行metadata双缓冲沿用commfirst方案。
Reduce入场等待本卡Pack完成，再在UB生成自身依赖，并跨H列切片复用。

Reorder按接收CV内的专家/M256边界顺序推导：

```
expert_row = sources[expert][peer] + peer_row - incoming[peer][expert]
group      = groupBase[expert] + expert_row / 256
source     = compactBase[expert] + expert_row
destination = self ? returnBase[peer] + peer_row : peer * sourcePitch + peer_row
```

已复制位图以每个片段的首个peer-local行号为bit位置；不同片段不会重叠，最大位数仍受
`maxBS*TopK`限制，沿用原bitmap空间。初始化改为向量清零。
每次只复制已完成GMM2组的片段，保持目的行顺序检查连续ready前缀；一个未ready的空洞
不会被后面的已复制行掩盖。此按需推导会增加Reorder侧地址计算，应结合整核时间评估。

## 保留的 unitflag 行为

- GMM1每M-group仍为I/256个逻辑tile，GMM2仍连续写 `gmm2Contiguous`，保持Unit Flag流水。
- S0保留原专家lookahead与scale-gather优先级；S1 EP4保留wave12，其余原调度保持。
- S0 Dispatch仍按原CV/quiet节奏；S1保留原合并发送节奏，不套用另一分支的通信策略。
- Combine仍128行PUT、每原CV发布一次SO ready；Reorder完成后才允许发送。
- self Reduce仍等待本卡CV的Reorder完成flag，不用AIC GMM2完成冒充返回数据就绪。
- 全来源完成捷径仍同时检查AIC完成和全部通信核的self Reorder完成计数。

RoutePrepare/RouteSummary/WaitRouteSummary的事件编号为62/63/64。
前者现在仅表示本地准备与路由下发，接收摘要构建/等待另列；不能仅凭前者缩短认定整核收益。
必须重编完整wheel（routing protocol6）；原benchmark命令不变。
显式执行后debug_snapshot从inverse重建compact order用于诊断，生产调用无Host路由或D2H。
route_stats的segment/link项返回-1表示未物化；DeviceRoutes.order为None。
