# Comm-first 合并PUT / 异步Publish / SO Combine

分支 `megamoe-urma-commfirst-so` 基于当前回退后的 `megamoe-urma`。不包含失败的共享GMM上下文、M128或N512实验；保留gate/up条件选择及random-balanced。

## Dispatch

按原通信CV为单位，发送批次目标为 `1,1,2,4,8,8,...`；本通信owner的Pack任务做完后，后续剩余连续数据一次下发（前两批仍各1个CV）。默认CV256即256、256、512、1024、2048…行。其它cv_rows按同样倍数处理。

每批X是一条大PUT，scale是一条独立PUT。通信核先准备本批发送份数+1的原始CV静态Pack配额，再等整批PackReady，然后PUT X、PUT scale、按覆盖的原始CV逐个SO发布epoch。**没有逐批quiet，也没有MTE覆盖写ready flag**。启动仍由全部AIV准备2轮；随后通信核Pack为2、2、3、5、9…份，发送为1、1、2、4、8…份。Pack数量按剩余全局Pack轮次截断，cursor保证每份恰好一次。

例如EP4/BS2048/CV256，16份CV的实际Pack为 `2 | 2,2,3,5,2`，发送为 `0 | 1,1,2,4,8`。这是每轮新增Pack不超过本轮发送+1，不是累计未发送余量不超过1。第一笔PUT会比启动结束额外等待本核2轮Pack，这正是本实验的主动间隔。

Pack只保证本核份额完成，仍须等所有生产者的PackReady。已下发源区在最终quiet前不覆盖，后续Pack写其它行。SO通知使用同一peer QP0的ordered atomic-add，接收端在每次PublishStart前清零Dispatch CV槽位，再等待对应epoch。每个原始CV只发布一次；合并一个数据PUT不代表只发布一个CV的flag。全部Dispatch提交后执行一次最终quiet，`DispatchSendComplete`及coarse结束点均位于该quiet之后，再进入Combine。

短peer/空peer原有的余量Pack循环保留，辅助激活核的静态Pack分工及优先级不变。Pack是提交间隔，不能保证上一批已经远端完成；正确性依赖SO发布而不是估算Pack耗时。

## Combine

每个通信核完成自己对端的Dispatch与自己原有的Pack份额后，直接扫描Combine，不再等待本卡其它通信核或其它Pack worker。原Pack静态划分和短/空peer补完循环保持不变；通信核Pack按当前发送份数+1在PUT前准备；不提前Combine到本核Dispatch阶段。同一peer不交叠Dispatch和Combine，不同peer可以交叠。

每轮扫描未发送CV，跳过未ready者。相邻ready CV的source/destination行都连续时合并为一个PUT；遇空洞、已发送CV、64位bitmap word边界或256MiB PUT上限就另起一段。最后一个实际未发送的通信CV不并入其它PUT。

这里沿用此前“末通信CV单独发送”的定义，不把专家计算M-group与CV视为同一对象。如果要求按最后计算M-group片段切分，需要另行处理跨CV部分完成，不能提前置整CV的ready bit。

payload使用QP0 RO，随后用同QP0的一次SO atomic-add提交一段CV的bit mask。所有bits只提交一次，disjoint mask相加等于OR；可乱序标记，不把未ready空洞误判完成。每次调用在Start前清零本卡Combine bitmap及finished标志，sender等待peer Start后才能写。最后SO finished之后执行最终quiet，覆盖所有数据和通知，保护下一次调用。

Combine中间没有每CV quiet，也不再通过MTE发布Combine ready。Reduce仍检查每个token的原CV依赖，只将ready读取从epoch改为bit，所有正向缓存、本卡GMM2完成条件保持原逻辑。

## 计时与使用

- Dispatch从第一次X PUT计至通信核的最终QP quiet，新增可见事件 `DispatchSendComplete=43`。
- Combine从第一次PUT计至通信核的最终quiet，新增可见事件 `CombineSendComplete=42`。最终quiet在timer关闭时也执行。
- 42/43仅画最终等待本身，不从首PUT画重叠条；总体带宽使用聚合timer首尾跨度和真实payload字节数。没有给Reduce加打点。
- 本卡入站完成仍由通信owner轮询peer finished并更新incoming计数；它不是本卡出站带宽的终点。

需重新编译完整wheel。core和experimental扩展都必须带`commfirst-so-bitmap-v1`标记，防止旧协议混装。benchmark命令和配置参数保持不变。

完成了源码协议检查及相关纯Python回归；未进行CANN编译、C++模型执行或NPU功能/精度/性能测试。

## Tile通知与EP4/BS1024 wave12

GMM1/GMM2都按tile完成发布。FIX_S1事件在目标tile全部计算/FIX指令下发之后立即捕获；若本核下一实际任务仍为同类型且输入已ready，先下发下一tile，再等待旧事件并发布上一tile。若下一类型不同、输入尚未ready或本核结束，则先Flush，不允许扣住GMM1结果导致激活/GMM2自等待。

同类型跨专家保持其原上下文和槽位状态；切换GMM类型才End、完成AIC/AIV0交接、再开启另一上下文。GMM1仍是UnitFlag输出、GMM2仍是普通NZ Scatter；没有引入跨阶段混用输出协议、M128、N512或新槽位。GMM2每完成一个tile增加一次组计数，Combine仍需等该组所有14个N方向结果完成。

仅EP4且每rank BS1024时，planner按专家顺序遍历真实M256 groups，每12组一个wave（尾wave允许不足12）：GMM1(w0)→GMM2(w0)→GMM1(w1)→GMM2(w1)…。不增加全卡或全核barrier，也不按“12个专家”假定稀疏/热点路由。此case不使用--gmm1-ahead-mgroups；其它EP/BS仍用原前瞻调度。

均衡或random-balanced时每卡96组、8个wave。GMM1总960 tile、GMM2总1344 tile不变；每核阶段切换15次，上下文开启/关闭16对。通知仍逐tile，不等整个wave结束才通知。

打印新增 `[URMA WAVE]` 和报告字段compute_wave_mgroups/gmm1_ahead_mgroups_applied。不同type切换或依赖等待的Drain条对应上一tile，下一同type Submit可出现在该Drain之前；没有新增PIPE_ALL打点。需要重编译安装匹配新的compute schedule标记。
