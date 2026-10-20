# AIV1 跨任务异步事件生命周期

2026-10-20。分支 `codex/megamoe-aiv1-async-1020`，父版为常驻UB + 公平调度 `a772852`。

## 结论

迁移已实现并在真实EP8上运行：Dispatch/Combine行槽的credit跨任务保留，SwiGLU提交与就绪发布分离；不把“已提交”当“已完成”。均衡BS512独立ABBA：806.03→807.69μs，慢0.21%，没有稳定收益，不合入master。

集中路由BS96扩展A/B/B/A：906.81→868.33μs，降耗4.24%。但小BS还伴随Combine槽数6→2，不能把这4.24%全部归因于异步生命周期，也不是已超越DS的证明。

## 迁移的边界

借鉴DS的跨任务槽位/事件持续存活与异步推进原则，不是整体照搬DS调度器。

- 保留原有expert-slice分工、Wave游标、GM地址、矩阵乘、SwiGLU/量化数学计算及最终跨卡同步。
- AIV1的Dispatch、激活、Combine使用父版已隔离的常驻UB；本次进一步隔离事件ID，避免切任务后槽位状态被重置。
- 仍是我们的64行激活chunk、GMM1结果经GM中转；没有迁移DS的32行让出、双CV输入stage或CV计算后端。DS在末个激活step完成后发布ready；本版将最终发布推迟到下次调度访问，不能称为完全相同的调度。
- AIC GMM1 FIX full/empty协议保持原样：消费者先确认issued，再等待真实FIX完成；EMPTY仍排在MTE3上，不能由Scalar提前归还。

## 先看哪些代码

以下位置均相对 `experiments/megamoe/csrc/kernels/mega_moe/`：

|位置|阅读重点|
|---|---|
|`common/aiv1_async_events.hpp`|8个本地event的分区；相同ID在不同HardEvent域不是同一个事件|
|`common/aiv1_activation_ub.hpp`|激活UB的初始化、获取、释放、最终消费credit|
|`mega_moe_wave_a8w4.hpp`：`ProcessAiv1PriorityStages`、`RetireActivation`|issue游标与完成状态分开；末个pending任务也必须退休|
|`stage/mega_moe_token_dispatch.hpp`：`FinishDispatchOwnedRows`、`RetireDispatchReady`|末行补写→设置完成事件→以后等待完成→发布GMM1 ready|
|`stage/mega_moe_gmm1_activation.hpp`、`epilogue/block_epilogue_activation_mx_quant.hpp`|64行块提交后通过UB credit保证安全覆盖；输出真正完成才发布GMM2 ready|
|`stage/mega_moe_gmm2_combine.hpp`：`RunTokenCombineSlice`、`DrainCombineRowBuffers`|行槽sequence跨slice递增，仅复用槽时等待，循环结束统一收尾|

### 数据与事件依赖

|事件域/ID|生产者→消费者|保护什么|
|---|---|---|
|MTE2→MTE3，0..5|Dispatch远端读→本地写|token/scale在UB已可读|
|MTE3→MTE2，0..5|Dispatch写出→下次槽复用|不能覆盖尚未读完的行槽|
|S→MTE3、MTE3→S，0..5|metadata填充/写出→下一次填充|Scalar不能覆盖在途metadata|
|MTE3→S，7|Dispatch末次写出→`RetireDispatchReady`|完整token/scale/metadata落地后，才原子贡献GMM1 ready行数|
|V→MTE2，0；MTE3→V，0|初始credit或量化写出→下轮激活读入|输入/输出重叠的激活UB安全复用，等待在硬件目标队列，不逐chunk排空Scalar|
|MTE3→S，6|激活输出/scale写出→`RetireActivation`|真实完成后才原子递增正确的GMM2 ready槽|
|MTE2→MTE3、MTE3→MTE2，6..7|Combine输入读→远端写→复用|两个行槽跨slice保持在途状态|

Dispatch不在每个slice结束消费全部empty事件；最后一行必须先提交，然后保存owner范围。下一次轮到Dispatch时等专用完成事件并发布ready，不重新初始化行环。Combine同理保持sequence，不在slice末排空。激活单独持有pending输出；下一次主循环先退休，再提交下一块。

循环退出条件包含pending Dispatch与激活，防止最后一块ready丢失；最后消费激活UB credit、排空两种通信行环。原来的Export、全核/跨卡完成与Unpermute边界仍保留，未因为打点增加屏障。

### 容量及行为变化

本地事件每域只有8个：保留Dispatch最多6槽，Combine固定2槽，激活借用其他HardEvent域。BS512的两种槽数与父版相同；BS≤256父版Combine为6，本版改为2，因此小BS不是严格单因素生命周期实验。H7168/EP8/BS512常驻UB仍为191680字节，未扩大UB。

## 实测结果

柜05 CPU7，Ascend950DT，CANN B080 9.2.0，驱动25.6.rc1.b010。H7168/I3072/E384/TopK6/shared0，FP8输入、FP4权重，seed20260922，完整通信。warmup10，采20取后10；逐轮rank-min，再对轮次/独立启动求均值，同时保存rank-max。正式性能全部timer-off，与PMU/流水诊断分开。

|路由、BS/卡|父版 μs|异步版 μs|降耗|证据级别|
|---|---:|---:|---:|---|
|balanced，512|806.028|807.692|-0.21%|同窗口A/B/B/A，各两次独立启动|
|balanced，96|655.417|656.859|-0.22%|各一次，回归筛选|
|balanced，1024|1146.802|1152.972|-0.54%|各一次，回归筛选|
|balanced，4096|4129.298|4126.755|+0.06%|各一次，不宣称收益|
|random，96|655.717|657.240|-0.23%|各一次，含负路由权重|
|skewed，96|906.807|868.332|+4.24%|扩展A/B/B/A，追加两次之间有控制器间隔；槽数变化未隔离|

skewed每个token都路由到专家0..5：rank0每专家768行，大量专家和rank1..7没有计算，覆盖空路径及严重不均衡。balanced、random、skewed的8卡输出均有限且与父版逐位一致；没有独立Golden，不能替代数学精度验收。未验证其他EP、共享专家及A8W8的原生性能/正确性。

### 为什么均衡BS512没有变快

完整被动timer：父版开销1.14%，异步版2.17%，均为仅rank0记录、32组×3角色完整96轨道；默认隐藏WAIT/SYNC/DRAIN，原始事件仍保存。ISSUE不是DMA完成或硬件纯计算时间；新旧区间含义发生变化，不可直接把长条减少当端到端节省。

`msprof --task-block=on --aic-metrics=PipeUtilization`：device0后10次调用、block0/15/16/31，每角色40条原生记录，与官方V6BlockPmu逐项对齐。下表是各流水“计数/total_cycles×duration”的平均活跃等效μs，流水会重叠，不能相加或当成独占耗时。

|角色/指标|父版|异步版|
|---|---:|---:|
|AIC Cube|530.01|529.79|
|AIC MTE2|570.85|568.04|
|AIV0 MTE2|689.92|689.11|
|AIV0 MTE3|522.62|523.26|
|AIV1 Scalar|413.17|471.53|
|AIV1 Vector|58.43|58.47|
|AIV1 MTE2|164.09|166.59|
|AIV1 MTE3|147.13|182.38|

GMM1 ISSUE累计454.57→454.40μs，GMM2 ISSUE205.54→206.61μs，权重Prologue也基本不变。虽然Dispatch和SwiGLU提交能更早返回，后续真实完成等待、游标/ready状态管理仍有成本；当前数据没有显示关键计算链被缩短。PMU不能单独证明增加的活跃计数全部变成暴露延迟，也不能据此断言通信带宽变差。

下一步优先：集中路由做Combine2-only消融，隔离4.24%来源；balanced进一步审查DS的32行yield及双CV stage，在有时间线证据时改，不能继续只删等待。保留真实完成、UB释放与最终排空三条正确性约束。

## 复现与离线证据

新分支需重编host扩展与device库，timer动态类型24→29，不能混用旧_C/旧raw。实际Python/host均为57024000个int64；计数区长度不是整个扩展buffer长度。

```bash
bash experiments/megamoe/scripts/build.sh
# 性能；默认full_moe，不是compute replay
python3 experiments/megamoe/scripts/batch_launch.py --ep-size 8 --bs-values 512 --no-timer
# 默认完整被动打点，单独测量并核对开销
python3 experiments/megamoe/scripts/batch_launch.py --ep-size 8 --bs-values 512
```

本轮私有环境的可复现原始命令在本目录脚本中，机器绝对路径不是公共部署要求；不含凭据。CPU实际执行8项协议/布局/调度测试、17项timer/parser测试通过。设备编译通过；初次CANN配置、首次host链接、错误ABI预检均失败在NPU运行前，失败日志保留，未把失败批次当成功。

身份：父版kernel SHA `b01905f8b456a624a793d2731a5a1e3268fe79abeae43c0c065ace2f05a9f21f`；异步kernel `9d3cb0756decace35496b4dffb056dba1f350f64677546d4a85faca87157d8a9`；重编host_C `6e0242f090dc1cac3b71ead0d7afcd0a57779feff323d53d50759284898628da`。编译后仅更新Python trace导出元数据，逐文件overlay SHA另记，未变更已测C++。

本地根目录 `C:/Project/Infra/ops-transformer-hs/outputs/aiv1-async-1020/`：

- `evidence.tar.gz` 与 `final/manifest.json`：1415文件/160份原始kernel_details.csv，逐份大小和SHA已核验，含两个版本实际SO与失败记录。
- `final/summary.json`：各次启动、shape筛选、路由复查、全部32组阶段账本、分核原始PMU计数及采集配置。
- `final/control-trace/timer_trace.json`、`final/async-trace/timer_trace.json`：父版/异步版完整被动流水；同目录含on/off性能与原始事件，其子目录下的`kernel_details.csv`为配套CSV（各自区分timer-on/off）。
- `final/pmu_{control,async}_PipeUtilization/`：对应msprof CSV、device0原生计数和官方metric_summary.db。

`verify_archive.py`核验归档；`check_identity.py <源码根> <final目录>`核验编译C++、Python overlay及二进制身份。原始证据不进入Git，不覆盖离线站冻结的七shape基线。
