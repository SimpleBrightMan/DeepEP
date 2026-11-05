# 专家唯一 owner 与按需路由（protocol4）

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
- 本卡同步由9次变为6次；原跨rank入口和元数据到达检查保留。
  摘要缓存每次调用在写入前失效，发布后只读；没有Host路由或调用前D2H。

固定专家槽的代价是额外预留 `384 * maxBS * TopK * 4` 字节。
maxBS8192/K8为96MiB，maxBS16384/K8为192MiB；实际只写真实命中的索引。
分桶UB最高176KiB；后续摘要使用96KiB区域前，分桶VF/MTE均已结束。

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

必须完整重编安装wheel，Python/native routing protocol同时为4，测试参数不变。
`prepare_routes()`仍只绑定输入及容量，返回的`DeviceRoutes.order`现在为None；
`reduce_index`只在执行完成后有效。显式`debug_snapshot()`在执行后从inverse重建CPU
compact order供旧诊断消费，不参与生产调用或计时；不再导出未物化的大描述符表。
`route_stats()`的segment/link项返回-1表示未物化，收到的行数与M-group数量仍有效。

建议分别对照BS1024和8192、EP4/EP8、random-balanced；重点观察RoutePrepare、Pack、
GMM包络、Reduce和无打点整核时间。编译及测试由用户执行，本次没有生成性能或精度结论。
