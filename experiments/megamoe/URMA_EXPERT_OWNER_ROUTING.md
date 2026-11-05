# 专家唯一 owner 与按需路由（protocol6）

仅 `megamoe-commfirst-shared` 的实验。未编译、未运行 CPU/NPU 测试，按用户要求交由用户执行。
`megamoe-unitflag-shared` 未修改。此前 protocol3 的测试结果不能作为本次变更的验证。

## 头部

- 56个AIV静态均分384个全局专家，每核6或7个，专家只有一个写入者。
- 每核遍历本卡全部TopK ID，每批4096个路由条目；支持int32和int64输入。
- 单VF处理本核该批全部专家，Compare/Compact产生flat ID和数量；flat保留token及topk槽身份。
- 专家桶使用int32固定槽 `[384, max_tokens_per_rank * topk]`，按输入顺序追加。
  重复专家路由仍有各自条目，容量不假设均衡或同token专家互异。
- leader只做384项计数前缀；不再合并56×384直方图。
- 跨rank仍MTE写计数前缀、返回起点和epoch；不传全部索引、不引入GET。
- 收到摘要后生成来源前缀、专家M256边界、peer CV前缀和原有小计算任务顺序表。
- 本卡路由同步从protocol5的3次降到2次；取消接收摘要后的全核join，改为本卡发布/获取flag。
  摘要缓存每次调用在写入前失效，发布后只读；没有Host路由或调用前D2H。

固定专家槽的代价是额外预留 `384 * maxBS * TopK * 4` 字节。
maxBS8192/K8为96MiB，maxBS16384/K8为192MiB；实际只写真实命中的索引。
分桶UB最高176KiB；后续摘要使用96KiB区域前，分桶VF/MTE均已结束。

## 单槽邮箱复用与同步

邮箱和routeReady均保留单槽，不新增双缓冲。安全性来自算子内部已有的尾部协议：

1. B必须先把第n轮全部来源的路由邮箱复制到自己的RouteInfo，之后才能进入Pack/计算/Combine。
2. B给A发布第n轮finished，包括B→A没有任何Combine CV的情况。
3. A的peer通信核必须观察到B的finished；A退出前的本卡join等待全部peer通信核。
4. A下一轮由同一owner stream串行执行，因此在覆盖B的邮箱前，已经确认B不再读旧邮箱。

finished可以在对端自身Reduce完成前发布：这里需要确认的是路由邮箱已经消费，
而不是对端整个算子已经退出。后续计算只读私有RouteInfo，不再读取接收邮箱。
本卡出站quiet、入站finished观察、尾部join和内部stream串行都保留；安全性不依赖
benchmark的all-reduce。不能只凭本卡quiet或本卡Reduce完成就省略入站finished观察，
也不能省略空peer的finished。routeReady用MTE覆盖发布epoch，等待精确本轮值，
只在Session创建时清零，避免清掉提前到达的本轮摘要。

RoutePrepare保留的两处SyncAll：

- 专家计数DMA完成＋所有消费者已失效旧摘要缓存 → leader读取计数、写前缀。
- 前缀发布完成 → 各摘要发送核读取前缀。

去掉原入口join、摘要发送后的额外join、Config reload后的join。
摘要发送改读独立且只读的routeOffsets，避免leader重写RouteInfo时与慢发送核读同一区域。
本核路由下发完成后释放本核流水，Dispatch/Combine循环不新增全核屏障。
无共享路径整个kernel现在是4次本卡全核join（路由2＋首批Pack1＋尾部1）、
0次显式RankBarrier；元数据/peer-start/完成flag的必要等待仍在。

## 路由接收与数据发送重叠

本地分桶/前缀完成并发出本核路由摘要后，各核心直接进入原首批Pack。
本地Config的Pack轮数、发送CV数、自卡行范围只由outgoing前缀计算，不等待远端计数。

- 通信AIV1：原Pack/Dispatch批次继续推进；全部Dispatch与本核Pack结束后才获取接收摘要，
  然后进入Combine。即使没有Combine CV，也必须先获取摘要再发送finished。
- 第一个激活AIV1（helper0）：非阻塞检查各来源epoch，缓存已到达者；未齐则做一轮自己的
  静态Pack配额。收齐后Close自己的Pack流水，生成接收摘要/小任务表，完成DMA和Config
  发布，再写本卡route-summary-ready。其余激活核等待的是这个本卡flag，并同样推进Pack。
- 激活核准备好后，沿用已经前移的nextPackRound，不重复Pack；进入原激活/scale gather/Pack循环。
- AIC/AIV0：获取本卡接收摘要后进入原计算/反量化调度。

outgoing与后发布的incoming部分隔开到不同64B缓存行；后者只写自己的尾部区域。
获取方在看到ready后失效接收摘要缓存并载入最终Config，不重写或失效Pack正在使用的
outgoing数据。摘要生成只在helper0上发生，整个过程不依赖benchmark的all-reduce。

计时增加RouteSummary（63）和WaitRouteSummary（64），coarse/fine均可导出。
新版本RoutePrepare（62）只覆盖本地准备与路由发送，不能单独与旧版包含收齐阶段的
RoutePrepare相减来声称整核收益。helper负向轮询之间的Pack在fine正常记录；coarse仍不
把后续Pack伪装成启动Pack。等待记录与Pack/Activation/GMM条不嵌套，避免解析时区间重叠。

## 执行期间

- 不再生成Fragment/Segment、Send/Receive Chunk、Link和PackRound大表。
- GMM1 gather、scale gather和GMM2 scatter按当前专家M-group沿source/CV边界推进游标。
- Combine按当前CV覆盖的专家/M-group检查就绪；Pack按peer CV虚拟拼接保持原静态分工。
- EP4 wave12与其他EP的原lookahead策略保留；发送顺序、批量大小和SO协议保留。
- Pack读取专家桶中的flat ID，复制X/scale时用精确8B MTE写 `inverse[flat]=expanded_row`。
  没有跨核Scalar缓存行散写；正映射桶在头部准备好，逆映射随Pack完成。
- Reduce入场等已有28个AIV1 Pack完成计数，再读取inverse。
  依赖只在入场推导到UB一次，后续H列复用；保留ready扫描、正向缓存和全来源完成捷径。
  本卡行仍等对应GMM2组完成，远端行仍等Combine CV。

Pack逆映射增加每行一条8B DMA。对齐q记录使用双metadata缓冲，跨段保留槽位，
仅复用对应缓冲时等待其捕获的MTE3完成；原6槽token流水和Retire5保留。
每批最多96行，遇专家边界会缩短；这部分有真实开销，需要用户对比Pack与整核时间。
按需描述符也把一部分地址推导移到计算/通信中，不能仅凭头部缩短断言整体更快。

## 接口和诊断

必须完整重编安装wheel，Python/native routing protocol同时为6，测试参数不变。
`prepare_routes()`仍只绑定输入及容量，返回的`DeviceRoutes.order`现在为None；
`reduce_index`只在执行完成后有效。显式`debug_snapshot()`在执行后从inverse重建CPU
compact order供旧诊断消费，不参与生产调用或计时；不再导出未物化的大描述符表。
`route_stats()`的segment/link项返回-1表示未物化，收到的行数与M-group数量仍有效。

建议分别对照BS1024和8192、EP4/EP8、random-balanced；重点观察RoutePrepare、Pack、
GMM包络、Reduce和无打点整核时间。编译及测试由用户执行，本次没有生成性能或精度结论。
