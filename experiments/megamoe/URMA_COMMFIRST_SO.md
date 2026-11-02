# Comm-first 合并PUT / 异步Publish / SO Combine

分支 `megamoe-urma-commfirst-so` 基于当前回退后的 `megamoe-urma`。不包含失败的共享GMM上下文、M128或N512实验；保留gate/up条件选择及random-balanced。

## Dispatch

按原通信CV为单位，发送批次目标为 `1,1,2,4,8,8,...`；本通信owner的Pack任务做完后，后续剩余连续数据一次下发（前两批仍各1个CV）。默认CV256即256、256、512、1024、2048…行。其它cv_rows按同样倍数处理。

每批先等待已准备的PackReady，然后PUT X、PUT scale、逐CV SO发布epoch；**本批发送完成下发后**，才执行该轮后续Pack。没有逐批quiet。启动仍由全部AIV准备2轮，首批Dispatch不再额外等待本通信核的新Pack。

发送批次仍为 `1,1,2,4,8,8,...`（Pack全部完成后保留原来合并剩余发送的规则）。发送后Pack的基础配额是本批发送份数+1，并补足下一批所需的累计前缀。例如第四批发送4份后，如果只Pack5份，累计准备仅14份，下一批8份发送需要累计16份，因此该轮Pack增加至7份。

```text
启动：Pack 2
随后：Dispatch 1 → Pack 2 → Dispatch 1 → Pack 2 → Dispatch 2 → Pack 3
      → Dispatch 4 → Pack 7 → Dispatch 8 → Pack 9 → Dispatch 8 → Pack 9 …
```

Pack按剩余全局轮次截断，不重复任何静态配额。EP4/BS2048/CV256共有16份：实际为 `P2 | D1 P2 | D1 P2 | D2 P3 | D4 P7 | D8`。短/空peer仍补完属于其它目的rank的本核Pack配额；不在当前PUT之前补新Pack。

Pack只保证本核份额完成，仍须等所有生产者的PackReady。已下发源区在最终quiet前不覆盖，后续Pack写其它行。SO通知使用同一peer QP0的ordered atomic-add，接收端在每次PublishStart前清零Dispatch CV槽位，再等待对应epoch。每个原始CV只发布一次；合并一个数据PUT不代表只发布一个CV的flag。全部Dispatch提交后不再quiet，本核补完原有剩余Pack配额后直接扫描Combine。最终Combine quiet统一收齐Dispatch和Combine的payload与SO flags。

短peer/空peer原有的余量Pack循环保留，辅助激活核的静态Pack分工及优先级不变。Pack是提交间隔，不能保证上一批已经远端完成；正确性依赖SO发布而不是估算Pack耗时。

## Combine

每个通信核提交完自己对端的Dispatch并完成自己原有的Pack份额后，直接扫描Combine，不再等待本卡其它通信核或其它Pack worker。原Pack静态划分和短/空peer补完循环保持不变；通信核Pack在当前PUT和SO之后进行，并保证下一批前缀已经准备；不提前Combine到本核Dispatch阶段。同一peer的Dispatch仍在途时即可提交Combine，共用QP0；接收端仍由SO flag保护。

每轮扫描未发送CV，跳过未ready者。相邻ready CV的source/destination行都连续时合并为一个PUT；遇空洞、已发送CV、64位bitmap word边界或256MiB PUT上限就另起一段。最后一个实际未发送的通信CV不并入其它PUT。

这里沿用此前“末通信CV单独发送”的定义，不把专家计算M-group与CV视为同一对象。如果要求按最后计算M-group片段切分，需要另行处理跨CV部分完成，不能提前置整CV的ready bit。

payload使用QP0 RO，随后用同QP0的一次SO atomic-add提交一段CV的bit mask。所有bits只提交一次，disjoint mask相加等于OR；可乱序标记，不把未ready空洞误判完成。每次调用在Start前清零本卡Combine bitmap及finished标志，sender等待peer Start后才能写。最后SO finished之后执行最终quiet，覆盖所有数据和通知，保护下一次调用。

Combine中间没有每CV quiet，也不再通过MTE发布Combine ready。Reduce仍检查每个token的原CV依赖，只将ready读取从epoch改为bit，所有正向缓存、本卡GMM2完成条件保持原逻辑。

## 计时与使用

- Dispatch粗条在最后一批SO提交后立即结束，不包含其后的尾部Pack；使用事件 `CoarseDispatchSubmit=53`，只计提交跨度；不再产生 `DispatchSendComplete=43`。不以提交时间计算完成带宽。
- Combine从第一次PUT计至通信核的最终quiet，新增可见事件 `CombineSendComplete=42`。最终quiet在timer关闭时也执行。
- 42/43仅画最终等待本身，不从首PUT画重叠条；总体带宽使用聚合timer首尾跨度和真实payload字节数。没有给Reduce加打点。
- 本卡入站完成仍由通信owner轮询peer finished并更新incoming计数；它不是本卡出站带宽的终点。

需重新编译完整wheel。core和experimental扩展都必须带`commfirst-dispatch-so-pack-after-v6`标记，防止旧协议混装。benchmark命令和配置参数保持不变。

本次先发后Pack修改通过160项Python回归及纯CPU C++调度穷举：覆盖0..512全局/peer CV数量、空peer、短peer和批次拆分，验证发送前已准备好本核配额、每个Pack配额恰好执行一次。未进行CANN编译或NPU功能、精度、性能测试。

## Tile通知与EP4所有BS的wave12

GMM1/GMM2都按tile完成发布。FIX_S1事件在目标tile全部计算/FIX指令下发之后立即捕获；若本核下一实际任务仍为同类型且输入已ready，先下发下一tile，再等待旧事件并发布上一tile。若下一类型不同、输入尚未ready或本核结束，则先Flush，不允许扣住GMM1结果导致激活/GMM2自等待。

同类型跨专家保持其原上下文和槽位状态；切换GMM类型才End、完成AIC/AIV0交接、再开启另一上下文。GMM1仍是UnitFlag输出、GMM2仍是普通NZ Scatter；没有引入跨阶段混用输出协议、M128、N512或新槽位。GMM2每完成一个tile增加一次组计数，Combine仍需等该组所有14个N方向结果完成。

仅EP4，所有支持的BS场景，planner按专家顺序遍历真实M256 groups，每12组一个wave（尾wave允许不足12）：GMM1(w0)→GMM2(w0)→GMM1(w1)→GMM2(w1)…。不增加全卡或全核barrier，也不按“12个专家”假定稀疏/热点路由。EP4不使用--gmm1-ahead-mgroups；EP8/EP16仍用原前瞻调度。

以EP4/BS1024均衡或random-balanced为例，每卡96组、8个wave。GMM1总960 tile、GMM2总1344 tile不变；每核阶段切换15次，上下文开启/关闭16对。通知仍逐tile，不等整个wave结束才通知。

打印新增 `[URMA WAVE]` 和报告字段compute_wave_mgroups/gmm1_ahead_mgroups_applied。不同type切换或依赖等待的Drain条对应上一tile，下一同type Submit可出现在该Drain之前；没有新增PIPE_ALL打点。需要重编译安装匹配新的compute schedule标记。


Pack now retains six slots across consecutive quotas and retires the old ready fence after five new row submissions. See [URMA_PACK_PIPELINE.md](URMA_PACK_PIPELINE.md) for stage boundaries and revised fine trace semantics.
