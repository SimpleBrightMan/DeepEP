# random-balanced 路由

用 `--routing random-balanced` 替换原来的 `--routing balanced`。URMA benchmark、普通benchmark、CPU Golden和batch launcher使用同一生成器；默认routing不变，算子kernel不变。

这对应参考脚本的 `hot_multiplier_x2=2`：全局EP×BS个token、每token TopK条路由；每个目标rank严格接收BS×TopK条。按剩余配额加权随机选rank，不放回；剩余token已经不足以容纳某rank配额时，强制分配给当前token。最后随机排列TopK槽位，目标rank内按出现次数循环选择专家。

路由seed固定20261014，随机流为`seed+BS*1000+2`，与参考脚本一致；输入X、权重和topk weights仍由原config seed生成。参考脚本里的physical_stride=6只适合连续布局EP64/E384；此实现用E/EP，所以EP4/8/16分别为96/48/24。

- EP>=TopK：一个token的目标rank不重复；默认配额下与参考脚本逐元素一致。
- EP<TopK：每token、每rank最多ceil(TopK/EP)条，按剩余容量强制分配。EP4/TopK8时每个token必定每rank两条，两个专家不同；EP8/TopK8时每个token必定每rank一条。这两种情况下随机性主要体现为TopK排列，并不制造随机的rank接收负载。
- 仅全局配额保证均衡；不要求每个源rank到每个目标rank的量均衡，也不要求每个wave/CV均衡。专家出现次数全局相差最多1。没有热点倍率参数，热点/偏斜实验继续使用其它模式。

按全局token顺序生成唯一表，再切出本源rank的BS行。每进程缓存最近一张全局表，CPU Golden线程共用，调用者拿独立副本。生成过程属于路由准备，不在融合kernel计时内；不要把生成时间算作通信性能。

Golden contract记录routing并包含生成器源码hash，精度测试需重新生成对应Golden，不能复用旧balanced缓存。这里的均衡指路由条数，并非token去重后的通信字节量。
