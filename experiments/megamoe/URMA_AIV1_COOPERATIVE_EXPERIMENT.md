# 实验③：AIV1合作调度与FIX完成消费

仅实验分支 `megamoe-unitflag-exp-consumer`。必须基于本次生产头部更新（AIV0并行RouteSummary）；
不把实验实现回写到两个shared生产分支。编译与NPU双标杆状态以总实验报告为准；CPU协议检查本身不代表NPU精度或性能通过。

## 分工

- 28个AIV1逐一对应同block AIC，消费真实FIX完成通知，并处理同AIC拥有的路由GMM1 tile激活。
- Scale gather按M-group编号模28静态分配，每组只有一个发布inputScaleDone的owner。
- Pack仍按原28个出站静态配额分配。出站token与本卡接收专家tile不等价，没有宣称它与AIC tile一一对应。
- 原EP-1通信owner保持peer/QP0独占；其它AIV1也参与completion、scale、激活、Pack。
- 共享激活仍按28核静态分配，最低优先级，每次最多一个slice。

每一轮合作调度：

1. 尝试一个已ready的Dispatch单元；PeerStart、PackReady或共享头限流未就绪则立即返回。
2. 本核Pack已完成且Dispatch已提交完时，尝试一个remote CV和一个self CV的Reorder/Combine推进。
3. 若同block AIC已有issued，关闭本核Pack事件生命周期，消费一个真实FIX full，发布GMM完成并归还empty信用。
4. 做一个ready scale gather，否则一个本核ready激活tile，否则一轮静态Pack，否则一个共享激活slice。

第3步具有有界推进机会，不能让持续ready的通信无限饿死完成消费，否则AIC会被2槽credit反压。
消费者发布完成严格晚于硬件full等待；Scalar issued计数只用于非阻塞探测，不代表数据已落地。

## 通信与尾部

- Dispatch使用RO payload + SO epoch，移除S0每CV quiet。S0保持一CV提交；S1保持1/1/2/4/8 ramp，
  每次合作turn最多8CV，以保证completion公平。S1 BS<4096保留共享GMM1未完成时的批数限流。
- 即使Dispatch为空，Combine与finished发送也必须先观察PeerStart，避免写入被对端本轮reset擦除。
- Combine保持128行PUT、每原CV一个SO ready，且只有连续已Reorder前缀可发送。
- Reorder bitmap占用Pack高UB，所以本核Pack结束前不初始化/使用它；没有冒险与Pack同时复用。
- self必须完成实际Reorder MTE3写入后才发布self CV ready；不以AIC GMM2完成代替。
- 空peer仍发送finished。S0在合作任务结束后执行最后quiet和入站完成观察；S1保持外层Reduce后quiet。
- 所有AIV1包括零任务core都初始化2个empty信用，并消费到Done。AIC最终结束仍排空信用和FIX。

## 计时与代价

Dispatch coarse条改为submission-only（ID53），不再把没有quiet的下发结束误称为完成。
通信core的coarse Dispatch/Combine包络可能包含期间的本地工作；其激活/完成细分请看fine。
非通信core可用CoarseActivation观察合作阶段包络。AIC矩阵tile与最终drain用于统一GMM包络。
AIV1上的Gmm1TileDrain/Gmm2Drain明确标成paired FIX completion，不与AIC tile跨泳道合并，也不计作AIC计算时间。
benchmark manifest记录28个completion/scale/路由激活worker，保留真实S0/S1通信批次与收尾区别。

频繁关闭Pack以让completion/激活复用MTE事件，会带来真实成本；额外Scalar调度、
Reorder和通信任务也可能让AIV1成为瓶颈。因此本方案是待测实验，不能提前声称AIC一直满载。
