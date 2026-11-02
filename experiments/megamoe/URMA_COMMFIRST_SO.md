# Comm-first 合并PUT / 异步Publish / SO Combine

分支 `megamoe-urma-commfirst-so` 基于当前回退后的 `megamoe-urma`。不包含失败的共享GMM上下文、M128或N512实验；保留gate/up条件选择及random-balanced。

## Dispatch

按原通信CV为单位，发送批次目标为 `1,1,2,4,8,8,...`；本通信owner的Pack任务做完后，后续剩余连续数据一次下发（前两批仍各1个CV）。默认CV256即256、256、512、1024、2048…行。其它cv_rows按同样倍数处理。

每批X是一条大PUT，scale是一条独立PUT。通信核先准备本批发送份数+1的原始CV静态Pack配额，再等整批PackReady，然后PUT X、PUT scale、按覆盖的原始CV逐个SO发布epoch。**没有逐批quiet，也没有MTE覆盖写ready flag**。启动仍由全部AIV准备2轮；随后通信核Pack为2、2、3、5、9…份，发送为1、1、2、4、8…份。Pack数量按剩余全局Pack轮次截断，cursor保证每份恰好一次。

例如EP4/BS2048/CV256，16份CV的实际Pack为 `2 | 2,2,3,5,2`，发送为 `0 | 1,1,2,4,8`。这是每轮新增Pack不超过本轮发送+1，不是累计未发送余量不超过1。第一笔PUT会比启动结束额外等待本核2轮Pack，这正是本实验的主动间隔。

Pack只保证本核份额完成，仍须等所有生产者的PackReady。已下发源区在最终quiet前不覆盖，后续Pack写其它行。SO通知使用同一peer QP0的ordered atomic-add，接收端在每次PublishStart前清零Dispatch CV槽位，再等待对应epoch。每个原始CV只发布一次；合并一个数据PUT不代表只发布一个CV的flag。每个Pack轮次提交结束后即可扫描Combine，不等待全部Dispatch提交；期间不新增quiet。最终Combine quiet统一收齐Dispatch和Combine的payload与SO flags。

短peer/空peer原有的余量Pack循环保留，辅助激活核的静态Pack分工及优先级不变。Pack是提交间隔，不能保证上一批已经远端完成；正确性依赖SO发布而不是估算Pack耗时。

## Combine：Pack轮次之后扫描，两CV一批

通信核每完成一个原始PackOneRound调用后扫描Combine；不在该轮Pack内部插入扫描，不强制排空或抢占Pack槽位。没有Pack任务时，仍可在Dispatch等待数据或批次间扫描。不是等整个Pack阶段或全部Dispatch提交完再扫描。

保留原recvChunks、原CV→GMM2组依赖、原Reduce依赖表。按CV对0/1、2/3…扫描，仅相邻且源/目标连续、两份都ready时才发送；某对未ready则跳过，检查后面的对。不把非相邻CV或任意计算M-group拼成一个PUT，不建立独立512行任务表。固定配对避免乱序发送后留下两个互不相邻的孤立CV。

完整PUT大小为2×cv_rows：CV128发送256行，CV256发送512行，CV1024发送2048行。peer尾部不足两CV时按实际行数发，取消旧的最后一个CV强制单发。每个原CV仍对应自己的SO ready bit；发送两CV后一次SO发布两个bit，最后一个单CV只发布一个bit。

继续使用单QP。最终quiet只在本核Dispatch、剩余Pack和Combine都提交完后执行。早发扫描只做Scalar PUT/SO和低地址UB状态维护，不占用高地址Pack UB及DMA槽位事件。fine的外层等待条会在实际Combine下发前结束，避免嵌套；coarse的D/C逻辑包络允许同一lane重叠。

## 计时与使用

- Dispatch粗条使用新事件 `CoarseDispatchSubmit=53`，只计提交跨度；不再产生 `DispatchSendComplete=43`。不以提交时间计算完成带宽。
- Combine从第一次PUT计至通信核的最终quiet，新增可见事件 `CombineSendComplete=42`。最终quiet在timer关闭时也执行。
- 42/43仅画最终等待本身，不从首PUT画重叠条；总体带宽使用聚合timer首尾跨度和真实payload字节数。没有给Reduce加打点。
- 本卡入站完成仍由通信owner轮询peer finished并更新incoming计数；它不是本卡出站带宽的终点。

需重新编译完整wheel。core和experimental扩展都必须带`commfirst-so-bitmap-v1`标记，防止旧协议混装。benchmark命令和配置参数保持不变。

完成了源码协议检查及相关纯Python回归；未进行CANN编译、C++模型执行或NPU功能/精度/性能测试。

## Tile通知与所有场景wave12

GMM1/GMM2都按tile完成发布。FIX_S1事件在目标tile全部计算/FIX指令下发之后立即捕获；若本核下一实际任务仍为同类型且输入已ready，先下发下一tile，再等待旧事件并发布上一tile。若下一类型不同、输入尚未ready或本核结束，则先Flush，不允许扣住GMM1结果导致激活/GMM2自等待。

同类型跨专家保持其原上下文和槽位状态；切换GMM类型才End、完成AIC/AIV0交接、再开启另一上下文。GMM1仍是UnitFlag输出、GMM2仍是普通NZ Scatter；没有引入跨阶段混用输出协议、M128、N512或新槽位。GMM2每完成一个tile增加一次组计数，Combine仍需等该组所有14个N方向结果完成。

所有支持的EP/BS/routing场景，planner按专家顺序遍历真实M256 groups，每12组一个wave（尾wave允许不足12）。EP4/BS1024保持GMM1(w0)→GMM2(w0)→GMM1(w1)→GMM2(w1)…；其它场景提前一波GMM1：GMM1(w0)→GMM1(w1)→GMM2(w0)→GMM1(w2)→GMM2(w1)…，最后补GMM2尾波。只有一波时执行GMM1(w0)→GMM2(w0)，没有任务的rank不生成虚假wave。不增加全卡或全核barrier，也不按“12个专家”假定稀疏/热点路由。所有场景均不再使用--gmm1-ahead-mgroups；该参数仅保留兼容校验，日志明确标记ignored。空目的rank不生成计算任务。

以EP4/BS1024均衡或random-balanced为例，每卡96组、8个wave。GMM1总960 tile、GMM2总1344 tile不变；每核阶段切换15次，上下文开启/关闭16对。通知仍逐tile，不等整个wave结束才通知。

打印新增 `[URMA WAVE]` 和报告字段compute_wave_mgroups/compute_gmm1_ahead_waves/gmm1_ahead_mgroups_applied。不同type切换或依赖等待的Drain条对应上一tile，下一同type Submit可出现在该Drain之前；没有新增PIPE_ALL打点。需要重编译安装匹配新的compute schedule标记。


Pack now retains six slots across consecutive quotas and retires the old ready fence after five new row submissions. See [URMA_PACK_PIPELINE.md](URMA_PACK_PIPELINE.md) for stage boundaries and revised fine trace semantics.
