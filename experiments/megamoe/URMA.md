# MegaMoE URMA 固定 CV 实验

本实验是完整的 Prefill 算子：真实路由准备、本地展开、Dispatch、GMM1、SwiGLU/MXFP8 量化、GMM2、Combine PUT 和加权 Reduce。没有屏蔽计算，也没有使用预置 balanced 专用路由。

本分支为 `megamoe-urma-tailCover`，基于 `megamoe-urma` 的多槽版本 `a35b4d3`。原分支保留尾部全卡同步和56核Reduce；本分支采用每CV完成信号和41核Reduce，详见下文。两版均须各自重新编译安装完整wheel，不能混用rank或Python/native版本。

首版范围：默认单机 EP16，固定使用28个AIC / 56个AIV、每个AIV至少192 KiB UB、H=3584、I=1280、384个专家、TopK=8、无共享专家、FP8输入/FP4权重、BS=2048/4096/8192。物理核心数更多的兼容机型仍只启动这28组计算核。任意合法路由均走同一流程；超过配置接收容量的偏斜路由会在启动计算前明确报错。

GMM按专家聚合不同来源的行，使用M256计算块，仅专家尾块不足256行；通信CV不再决定计算分块。SwiGLU在每个已就绪GMM1 tile内按最多128行处理，全部子块写出后只发布一次该tile的完成贡献。其两个BF16输入缓冲合计128 KiB，AIV0的W4解量化流水使用192 KiB，后者决定当前UB最低要求。资源不足时分别报告核心数或UB容量，并打印实际查询值。

通信接口与固定 CV 调度参考 [ops-transformer PR11602](https://gitcode.com/cann/ops-transformer/pull/11602)（分析版本 `3ab6b8bba2cbdaf3751596c3e4b0488e1991df5f`）。计算复用本仓主线的 FP8/W4 CATLASS 和 SwiGLU 算术；返回方向改为 PUT。

## 数据与通信顺序

1. 每个输入 `[token, topk_slot]` 都是独立路由实例。按 `[目标rank][目标本地expert][原始token/topk位置]` 稳定排序，建立 `order` 和逆映射 `reduce_index[token,slot]=expanded_row`。
2. 56个AIV共同展开FP8 token和E8M0 scale：远端路由写连续发送区，self路由直接写本rank接收区；不再从发送区执行一次SelfDispatch拷贝。一个token命中多个expert时复制对应多行，本版不做去重。
3. 每个目标rank的连续行流按 `cv_rows` 截断，通过 URMA PUT 发送。token与scale使用各自连续平面。
4. 对每个本地expert，把16个来源的行按source编号连接，按M256划分计算组；专家尾组可以不足256，不与下一个专家混算。每组的Segment记录来源、CV、接收行、组内行和返回展开行。Segment在source/CV边界切分，但这些边界不再切分矩阵计算。
5. GMM1按Segment把X分段加载到同一L1计算块，只等待该组实际涉及的Dispatch CV及scale准备。scale由13个AIV1独立准备，每个M组只聚合一次到expert-major GM；AIC沿用连续MX scale加载。GMM1输出、SwiGLU及GMM2输入使用连续的expert-major计算行序。GMM2按同一映射将远端结果分段写回原source-major连续输出区，self结果直接写Reduce最终读取的位置。各peer的Combine CV等待所有覆盖它的M组完成全部H列，再按原连续区间PUT。
6. 本rank的15个远端Dispatch全部完成且self接收区已就绪后才提交Combine PUT。每个Combine CV执行 `PUT → quiet(peer) → MTE DataCopy发布epoch flag`。计算侧AIV完成自己的任务后即可按token依赖Reduce，不再等待最终全卡同步。

15个通信AIV1各独占一个远端rank。通信worker `w` 的peer为 `w < rank ? w : w+1`；剩余13个AIV1执行SwiGLU。self就绪标志在Pack本地同步后发布，原来承担self拷贝的block15随后直接进入SwiGLU。远端两个阶段都只使用PUT，队列同一peer始终只有一个提交者。

Pack使用全部56个AIV，任务编号为`block*2+subblock`。Reduce只使用28个AIV0和block15～27的13个AIV1，共41核；15个通信AIV1不参与Reduce。计算侧每核结束自己的反量化或scale/SwiGLU任务后，先排空本核流水以交还UB/event，再独立进入Reduce。当前内核固定启动28组；即使物理设备有64个AIV，也只使用已启动的56个。

本地阶段使用多槽流水，benchmark打印`local_pipeline=pack6-reduce6-output2 tail=combine-ready-scan-reduce41`。Pack按56核、Reduce按41核分配连续行区间：

- Pack使用6个token/scale槽，按槽等待上一次MTE3完成，再下发新的MTE2；不同槽的读取与写出可以重叠。`order`批量DMA到UB，当前支持的最大BS8192每核最多1171行，一次预取即可，payload循环不再从GM逐行标量读取路由。UB占用65.5 KiB。
- Reduce使用6个BF16/FP32输入槽、独立FP32累加器和2个BF16输出槽。每核自己的`inverse + weights + readiness dependencies`一次搬入UB，BS8192时最多200个token，不再用128行metadata批次限制扫描范围。每个token的8路结果全部就绪后才预取payload；槽位按实际提交顺序复用，写回仍使用原始token行号。数据、metadata、正向缓存、提交位图和六项route ring合计191328 B，低于192 KiB。
- 返回布局、FP32乘加顺序和最终BF16舍入不变。self直接写receive/return，远端PUT的CV粒度和数量不变。Pack仍全部完成后才进入Dispatch；Reduce可以与其他核心的计算和Combine重叠。

Python会拒绝不含本分支tail调度标记的native扩展。正式耗时仍取关闭timer/trace的`CACHED_FORWARD`。Pack跨度覆盖56核；本分支Reduce跨度覆盖41核且包含等待结果就绪的时间，不能直接当作纯本地Reduce耗时，也不能与Combine跨度相加。原timer的Pack字节统计写入的FP8+scale，Reduce字节仅统计输出BF16，均不是完整HBM读写流量。

### Combine 完成信号与 Reduce 重叠

- 路由准备时为每个expanded row生成依赖：远端为`producer_rank * maxPeerChunks + cv_id`，self为带高位标记的GMM2组编号。再用已有`reduce_index`在NPU上执行一次`index_select`，缓存token-major `[BS,8]`依赖表。forward不重建它；native绑定inverse的地址和版本，修改后必须重新准备路由。
- 远端每CV的`quiet`完成后，通信owner用MTE发布32B信号到独立128B槽。Reducer只在本地轮询槽，要求epoch精确相等；self等待对应组的全部GMM2列tile完成计数。AIV0结束不能代替self结果完成。
- 每个通信owner在每次Combine `PUT`之后、`quiet`之前，非阻塞读取本卡上对应peer的`finished(epoch)`。首次观察到本轮完成才给本卡入站计数器`AtomicAdd(1)`，用核内bool记住已计入，后续不再读取该peer的finished。
- 本核所有Combine CV均完成`quiet + CV ready发布`后，用MTE发布自己面向该peer的`finished(epoch)`，再带200周期退避地观察尚未计入的入站完成；空edge也发一次通知。必须先发后等，且每个owner只处理自己的peer，避免相互等待形成环。该通知使用独立phase 1的128B槽，不覆盖逐CV ready。
- 第16份是self：28个AIC各自完成全部GMM2/FIX输出后，给本卡AIC完成计数器加一。通信owner 0非阻塞检查它是否到28，再且仅再给入站计数器加一；不能预置self为完成，也不能用AIV0退出代替GMM2输出就绪。此门槛偏保守，self输出可能更早完成。
- Reducer轮询入站计数，观察到16时，执行S→MTE2门禁并锁存“全部就绪”。剩余未提交token不再读取就绪flag、不再逐token做ready门禁，直接进入跨token六槽流水；metadata、输入槽、Vector和输出槽事件仍保留。普通payload GM读取不受影响。
- 没有尾部全卡barrier或Reduce之前的全本机汇合。只保留Pack后的本机汇合和函数退出前的本机MIX汇合；后者包括28 AIC、56 AIV和完成计数的所有贡献者，保护workspace及下一轮复用。本轮计数器在发布Start前由原有清零操作重置；finished区保留epoch，不逐轮清空。
- 下一次调用仍在Pack完成后进行peer Start握手。Start接受`>= epoch`，防止双向空路由的peer提前进入下一轮、覆盖旧Start而造成死等。有效CV的ready仍使用`== epoch`，并且不逐轮清空对称ready区。
- 每个worker循环扫描自己的未提交token。遇到首个未就绪依赖就跳过该token，遇到可执行token立即提交，不等待凑满一批；提交时即标记，避免重复执行。只在整轮扫描没有进展时退避200个周期。不同token允许乱序执行，同一token的8路FP32乘加顺序保持不变。
- 已观察到就绪的远端`(producer_rank, CV)`和self计算组记录在本核UB正向缓存中，本次调用不再重复读取对应GM flag；未就绪结果不缓存，下轮重新检查。提交标记每次初始化；正向缓存在需要依赖检查时清零，不跨epoch复用。该流程使用实际路由依赖，不要求balanced或固定token分布。
- 扫描与下发共用Scalar，GM flag读取仍可能延迟后续指令下发。已有MTE/Vector任务可与扫描重叠，但不保证流水始终饱和；切换token不会主动排空流水，只有最终输出完成后统一退出。全入站计数只用于提前解锁连续流水，不要求Reduce先整体等到16。性能以关闭timer/trace的完整`CACHED_FORWARD`实测判断。

trace保留每CV的`CombinePublish`。Reduce在每个有任务的reducer上只记录一条：从Reduce函数入口到既有的最终流水排空完成，包含metadata预取、token结果就绪等待、计算及输出写回；在当前三个BS下共41条Reduce。这条跨度不能解释为纯Vector工作时间，也不能通过隐藏等待条扣除其中的ready等待。

新版不再逐token记录`WaitReduceReady`和`Reduce`，也不单独记录`ReduceDrain`，减少循环中的scalar日志写出与cache flush。计时结束使用原有排空位置，不添加同步；其它trace事件不变。`WaitReduceReady`和`ReduceDrain`编号仍保留供历史bin解码，当前设备不会生成这两类事件。`--trace-include-waits`只显示实际记录的等待，不能恢复新版已不记录的逐token等待。旧版本尾部同步事件仍可解码，本分支不会生成已删除的全卡同步事件。

self直接写地址按现有`send_base/receive_base`映射；同一计算组可同时含self和远端行，逐Segment决定输出地址。远端PUT的地址、顺序、CV和数量不变，计算按expert顺序推进多个peer的返回结果。当前Pack完成前不会对peer发布新epoch的Start；上一轮本地计算/返回/Reduce由同一runtime stream串行完成，因此self直接写不会覆盖尚在使用的上一轮数据。

接收数据/中间结果/返回结果不按CV覆盖复用。所有BS共享固定对称heap布局，epoch由`UrmaBuffer`统一递增，不因创建新Plan或切换BS重新归零。

通信域启用MTE和UDMA，每个对端独立QP0，不使用共享jetty。MTE只用于完成标志与本地搬运；远端token、scale和FFN结果均走UDMA PUT。每次调用在同一runtime stream串行执行，并在下一次写入前与各peer握手，保护上一轮Reduce尚未读完的结果。

### 专家 M256 与通信 CV 解耦

通信CV仍是**单个来源rank发送的一段连续行**，用`(source_rank, cv_id)`唯一标识。计算批次改为一个本地专家的全部M256组，同专家内不混入其它专家。每个M组有独立的输入依赖列表，不需要等齐全16个来源。例如BS2048均衡时，专家0第一个M256由来源0～5分别贡献43、43、42、43、43、42行。

计算前瞻按 **M256 group 数量**配置，默认 `--gmm1-ahead-mgroups 12`。每次执行下一个专家的 GMM2 前，先按专家顺序补充 GMM1，直到已排入 GMM1、尚未排入 GMM2 的 group 数达到下限；同时保留原有至少一个后续专家的前瞻。为复用上下文，同一专家仍整体计算，窗口在专家边界向上取整；专家尾组不足256行也计一个 group。它是每核遵循的任务顺序，不引入全核同步，也不表示这些 group 在同一时刻全部完成。

```text
pending_groups = 0
next_gmm1 = 0
for e = 0 .. localExperts-1:
    while next_gmm1 < localExperts and (
        next_gmm1 < min(e + 2, localExperts) or pending_groups < ahead_mgroups
    ):
        GMM1(expert[next_gmm1])
        pending_groups += actual_group_count[next_gmm1]
        next_gmm1 += 1
    GMM2(expert[e])
    pending_groups -= actual_group_count[e]
```

H3584/I1280/E384/TopK8、EP16均衡时，每个M256 group有5个GMM1逻辑tile（gate/up成对），让28个AIC每核至少领到2个tile需要 `ceil(28*2/5)=12` 个group；至少3个需要17个group，可设置18进行实验。默认12时，BS2048首轮从原6个group增加到12个（60tile，每核2～3个）；BS4096原本就是12个、BS8192原本是22个，两个规模的整条专家任务顺序与旧版相同。真实路由按实际group计数，不能用此均衡算例保证任意路由的并行度。

`--gmm1-ahead-mgroups 0` 精确恢复原调度，便于对照。窗口持续补充，不仅影响启动预热；所有GMM1和GMM2仍各执行一次。Host在路由准备时生成任务数组，AIC和对应AIV0读取同一份顺序；中间结果使用既有独立GM行，不新增L1/L0/UB槽位，专家结束握手保留。更深窗口可能推迟Combine/Reduce或提前遇到尚未到达的Dispatch CV，应同时比较完整算子耗时。新的起跑对齐和连续采样方式见 [URMA_TIMING.md](URMA_TIMING.md)。

MX scale在L1中的zZ布局按16行组织，任意来源边界可能落在2字节偏移，不能直接调用要求32字节LocalTensor起址的DMA。为此只增加`receiveRows*(H/32)`字节的scale GM工作区：准备时UB每行补齐到32字节，写回GM时压紧；不新增完整X重排。该缓冲在每次调用重新填充，组完成标志逐次清零。13个AIV1分别推进scale准备和activation两个游标，只有依赖就绪才执行任务，避免等待GMM1时阻止后续scale准备。trace中的`InputScaleGather`记录该额外搬运，`WaitInputScale`记录AIC等待。

Scale准备只在详细trace中独立计时，未归入原有8个aggregate phase；因此不能用这些phase的累计时间代表完整算子耗时。正式`CACHED_FORWARD`和`FULL_E2E`均包含该步骤。

空专家没有计算任务。各AIC及其配对AIV0按本核任务独立推进，专家批次之间没有全核barrier；GMM1下一批可以与上一批的SwiGLU/GMM2重叠。13个SwiGLU核使用新的M组任务编号，GMM2等待同组全部1280列activation完成，不能只等待其中一个N tile。

返回CV保存它所依赖的M组编号列表，同一组去重后只出现一次。每组全部GMM2 N tile完成分段输出后才满足依赖；仅提交Cube/FIX指令不能提前放行PUT。BS2048时一个peer的CV0大致跨6个专家，新顺序让多个peer按CV波次逐渐就绪，不保证15个peer同时开始。

GMM1和GMM2分别持有可复用的上下文：同一专家、同一GMM阶段内，本核相邻的M256任务沿用L1/L0缓冲、事件及ping-pong槽位状态，仅更新当前组的地址和形状；本核在该专家上的最后一个有任务组才执行End和配对AIV0确认。两个阶段仍互斥使用同一组硬件资源，切换阶段或专家时释放旧上下文。无任务核不启动上下文，该确认不涉及通信或SwiGLU AIV1。

GMM1仍逐tile等待FIX输出完成并发布就绪标志，以便SwiGLU及时开始。GMM2改为在本核完成该专家全部任务后统一等待FIX输出完成，再按各M组实际完成的tile数批量累计`gmm2Done`；中间M256组不再执行FIX_S或发布计数，内部FIX_M事件继续保护L0C槽位。Combine和self Reduce仍检查每组的完整N方向计数，不会提前读取结果，但通知时间推迟到本核的专家批次末尾。此处不增加等待Dispatch的门槛，也不改变远端每CV的`PUT → quiet → ready flag`；通信核保留本rank所有Dispatch完成后才启动Combine的原有规则。

trace中的最后一个参与组包含本专家该阶段的上下文收尾；若与tile条合并，该条也包含这段收尾时间。GMM2中间组不再单独产生Drain条。

GMM1直接Gather的X，以及AIC读取的已整理输入scale，均恢复默认L2策略，不再强制禁用L2。AIV对接收scale的一次整理读取、Reduce读取及完成flag的缓存策略保持原样。

默认均衡模型的每rank任务数如下，CV128/256/1024等取值不改变这些计算组数：

| BS | M256组（含专家尾组） | GMM1逻辑tile | GMM2 tile |
|---|---:|---:|---:|
| 2048 | 72 | 360 | 1008 |
| 4096 | 144 | 720 | 2016 |
| 8192 | 264 | 1320 | 3696 |

benchmark打印`[URMA SCHEDULE] compute=expert-m256-mgroup-lookahead-v1`及M-group窗口并写入manifest；Python会拒绝不匹配的native扩展，需要重新编译安装完整wheel。流水图和正式耗时仍需实测验证；不能将tile数量减少倍数视为耗时加速倍数。

### 本地多槽流水 A/B 验证（2026-10-20）

以下数据属于原分支 `megamoe-urma` 的多槽版本 `a35b4d3`，不是本分支tailCover结果。在两台Ascend950DT（每机8卡）上对比该版本与`e319740`对应的单槽本地实现，通信和专家M256计算不变。H3584/I1280/E384/TopK8、balanced、CV256，每次调用仍启动28 AIC/56 AIV。两机全部设备健康OK且空闲后才运行，旧新分别创建独立通信域。

| BS | Pack：旧→新 μs | Reduce：旧→新 μs | 完整forward：旧→新 μs | forward耗时下降 |
|---:|---:|---:|---:|---:|
| 2048 | 140.716→35.317 | 171.034→61.518 | 1388.829→1172.365 | 15.59% |
| 4096 | 279.727→77.510 | 337.898→119.865 | 2396.253→1977.776 | 17.46% |
| 8192 | 642.160→169.378 | 675.963→235.742 | 4614.845→3708.643 | 19.64% |

Pack/Reduce来自单独aggregate timer调用：先计算每rank全部56 AIV的首尾跨度，再对16个rank求均值，按1000 ticks/μs换算。完整forward是关闭timer/trace的`CACHED_FORWARD rank-max-mean`，不含路由准备；warmup=2、iters=10，取后5轮。两类计时来自不同调用，不能精确相加。

16rank×3BS的全部BF16输出与旧版逐位一致且均为有限值；这是旧新回归，不是独立Golden验证。841项CPU测试、CANN9.2编译和native import通过。本次为一轮双机A/B，表中性能不能视为UBX单机结果。控制机测试目录为`/home/codex-deepep/megamoe-urma-local-pipeline-ab-20261020-b07`，`comparison.json`与`phase_records.json`保留统计口径及逐rank记录。

## Python 接口与路由成本

```python
from deep_ep_experimental.megamoe.urma import UrmaBuffer

buffer = UrmaBuffer(group, num_max_tokens_per_rank=8192, cv_rows=256, gmm1_ahead_mgroups=12)
# x = (FP8 E4M3 tensor, E8M0 scale tensor)
# weights1/2 来自 transform_weights_for_mega_moe；不要拆掉GMM1准备对象。
y = buffer.forward(x, topk_idx, topk_weights, weights1, weights2)

# 仅用于显式复用不变路由，benchmark单独标成CACHED_FORWARD：
prepared = buffer.prepare_routes(topk_idx)
y = buffer.forward_prepared(x, topk_weights, weights1, weights2, prepared)
buffer.destroy()
```

`prepare_routes`每次在输入NPU上执行排序、计数、前缀和及逆映射。随后把384个counts复制到CPU，经Store交换完整计数表，native据此生成通信CV、专家M组、Segment及CV输出依赖表并送回NPU。它包含设备同步和host控制开销，是后续需要优化的一部分。

`forward`始终包含这一步。`forward_prepared`只允许复用同一buffer的不变路由。修改prepared中的任何路由tensor后拒绝继续使用。默认`max_recv_rows=2*max_BS*topk`；非均衡路由可能需要在所有rank一致增大该参数，并承担额外HBM占用。

同一buffer的native调用统一入队到owner专用NPU stream。准备Reduce依赖表的异步`index_select`也在owner stream运行，避免在另一条caller stream使用prepared plan时读取尚未生成的依赖。调用前等待caller stream的输入/描述符写入，返回时在caller stream插入输出就绪等待；不同caller stream也不会并发覆盖同一对称workspace。`destroy()`同步专用stream后再集体清理。

校验或启动异常会向Store发布abort，并禁止再次使用该buffer，避免各rank的epoch或调用序列分叉。Store abort能唤醒host协调等待，不能取消已经运行的NPU kernel；发生异常后应由启动器终止整个作业，不应捕获异常后继续调用或让单个rank进入清理屏障。正常cached forward不插入额外host集合通信。

## 编译和运行

使用本分支的完整MegaMoE wheel构建流程，URMA随`DEEPEP_BUILD_MEGAMOE=ON`编入，没有额外URMA编译开关。先激活Python环境并source匹配的CANN包，在仓库根目录执行：

```bash
export PYTHON="$(command -v python3)"
export DEEPEP_BUILD_MEGAMOE=ON
export DEEPEP_MEGAMOE_TIMER=ON
# 若使用已经安装的外部依赖，设成本机实际路径；否则保持未设置，
# build.sh会准备仓库固定版本的bundled依赖。
# export SHMEM_ROOT=/actual/shmem/install/shmem
# export CATLASS_ROOT=/actual/catlass
bash experiments/megamoe/scripts/build.sh
```

只能在已分配给本作业的全部16卡上运行。以下命令在仓库根目录执行，随后切到`/tmp`，避免源目录的`deep_ep`遮蔽已安装wheel：

```bash
REPO="$PWD"
OUT="$REPO/benchmark-runs/megamoe_urma/$(date +%Y%m%d_%H%M%S)"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:19091
export SHMEM_ROOT="${SHMEM_ROOT:-$REPO/third-party/shmem/install/shmem}"
export CATLASS_ROOT="${CATLASS_ROOT:-$REPO/third-party/catlass}"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib${CONDA_PREFIX:+:$CONDA_PREFIX/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
(
  cd /tmp || exit 1
  python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=16 \
    -m deep_ep_experimental.megamoe.urma_benchmark \
    --bs-values 2048,4096,8192 --routing balanced --cv-rows 256 \
    --warmup 2 --iters 10 --timer --output-dir "$OUT"
)
```

同一命令将三种BS顺序跑完，整个进程只创建、销毁一次通信域。原来的`DEEPEP_MEGAMOE_PREFILL_OPT`、4/24核配比、XOR顺序、MTE槽位数不控制这个独立URMA接口。

`cv_rows`首版接受128到4096之间128的整数倍。原始输入scale可以为`[BS,H/32]`或`[BS,H/64,2]`，Python在native调用前统一为后者，不改变scale内容。

默认只测性能，明确打印`VALIDATION skipped`，不将其写成精度通过。如需数值对照，先按照现有CPU Golden流程生成FP8、相同config/seed/routing/BS/check-tokens的缓存，再增加`--golden-dir <目录> --check-tokens <同一数值>`。CPU路由单元测试不等于NPU数值验证。

## 同UB域的双机功能验证

双机8+8可在两边torchrun使用`--nnodes=2 --nproc-per-node=8`，分别设置`--node-rank=0/1`，共用可达的`--master-addr`、`--master-port`和`DEEPEP_SHMEM_ENDPOINT`，并给benchmark增加`--allow-multi-host`。API对应`UrmaBuffer(..., allow_multi_host=True)`；必须所有rank一致开启。每台的`ASCEND_RT_VISIBLE_DEVICES`只列本机8张卡。

此开关仅用于同一可映射UB内存域的功能验证。初始化后按仓库固定版本SHMEM的真实topology标志检查所有peer的MTE可达性，再经Host Store做全rank就绪核验。`aclshmem_ptr`非空本身不能证明MTE可达。拓扑检查失败时中止整次作业，不进入设备通信；也不尝试在不可达域上执行设备barrier清理。普通跨机RoCE网络不满足本实现的MTE完成信号条件。

不同机型跑出的耗时只证明流程能够执行，不能作为UBX性能结论；打印的本组/跨组仍按连续4个逻辑rank分组，并在manifest中注明这一口径。

## 计时和带宽

每种BS输出三项相互独立的实测，正式统计取后半轮：

| 指标 | 计入范围 | 时钟 |
|---|---|---|
| `ROUTE_PREP` | NPU路由操作、D2H计数、Store交换、native描述符构建/拷贝 | Host wall time，完成后同步 |
| `CACHED_FORWARD` | 本地展开、两次GMM、SwiGLU、双向阶段PUT、Reduce与本机退出汇合 | NPU Event区间，不含预先准备路由 |
| `FULL_E2E` | 每次全新路由准备 + 完整forward + 完成同步 | Host wall time |

输入/权重生成、权重格式转换、通信域创建、计时前的host对齐屏障均不计入上述区间。输入契约是预量化FP8，BF16到FP8输入量化不在本接口内。`FULL_E2E`是单独执行的完整测量，不是将前两项相加。

`--timer`另跑一次诊断调用，保存每rank原始`urma_timer.bin`及schema。URMA格式为`int64[28,3,8,16]`，引擎顺序AIC/AIV0/AIV1，阶段为Pack/Dispatch/Gmm1/Activation/Gmm2/Combine/Barrier/Reduce。每记录前七个字段为`first_tick,last_tick,sum_ticks,payload_bytes,count,reserved,schema`。不要用旧Wave timer parser读取。

带宽分开报告Dispatch/Combine各自的INTRA/CROSS：

```text
单卡某阶段某通信类有效带宽 = 该类实际payload_bytes总和
                             / (该类最晚结束时刻 - 最早开始时刻)
```

只使用同一rank的时钟；不平均核时、不乘2、不跨rank相减。当前INTRA仅包含同组另外3个rank的远端PUT，CROSS包含其余12个rank；self直接写属于Pack/GMM2，不计入Dispatch/Combine通信字节，输出标为`payload=remote-peers`。绝对均衡时，两种通信阶段分别按原始全量的`3/16`和`12/16`计算这两类payload。不同类别拥有不同时间窗口，不能把两类带宽直接相加。这个数值包含该阶段调度/等待间隙，是单向聚合有效带宽，不是链路瞬时峰值。

解析器仍兼容旧版本含SelfDispatch/SelfCombine的timer：检测到self通信记录时继续计入INTRA，并在JSON中设置`includes_self_copy`、输出标为`payload=remote-peers+self-copy`。因此新旧INTRA带宽的payload范围不同，不应直接横向比较。Pack字节仍统计全部展开行，包含self直接写；GMM2按计算事件计时，不把相同self输出再作为通信payload重复累加。

原始tick始终保留。换算默认沿用现有MegaMoE `timer/protocol.py`的1000 ticks/us；这不是硬件频率自动检测。必要时通过`--timer-clock-mhz`覆盖。

## 逐事件流水 trace

增加`--trace`会隐含开启`--timer`。每种BS先完成关闭timer和trace的正式测量，再单独执行一次原有aggregate timer诊断，最后额外执行一次详细trace诊断。因此，详细打点开销不会混入正式耗时或原有带宽统计。它会改变诊断调用自身的耗时，应使用正式测量判断优化收益。

`--trace`现在自动合并全部rank，每种BS生成`$OUT/bs_<BS>/allrank.json`，可直接在Perfetto/Chrome Trace Viewer打开。每个rank作为独立process，内部保留84条AIC/AIV轨道，按rank编号排列；原有每rank JSON也保留。默认导出的两种JSON均隐藏`Wait*`及普通`LocalSync`；解析旧trace时仍保留尾部`FinalPeerHandshake`、`RankBarrier`和`LocalSync(sync_stage=2)`，本分支已不生成这三类尾部同步事件。`Quiet/Drain`继续显示实际通信完成等待和流水排空。过滤不移动时间戳、不压缩空白，不修改设备同步、原始bin或带宽统计。

如需显示所有已记录的等待条，运行时增加`--trace-include-waits`；也可从已有bin通过下方离线导出命令增加`--include-waits`恢复被导出器隐藏的事件。新版Reduce只有每核一条完整跨度，内部ready等待没有单独事件，这些选项不会重新生成逐token等待；历史bin中的`WaitReduceReady`仍可显示。只传`--timer`仍仅导出aggregate统计，逐事件流水及`allrank.json`需要`--trace`。

合并在诊断数据导出之后由rank0执行，不进入正式性能测量。单机/共享文件系统直接读取；双机目录不共享时，通过Host Store分块汇集JSON到rank0临时目录，再合并并清理临时副本。只有全部rank导出成功且文件完整才发布`allrank.json`，失败会通知全部rank，不静默生成缺rank的文件。双机的合并文件位于rank0所在机器。

在已配置上述CANN、SHMEM环境并分配全部16张UBX卡后，从仓库根目录运行：

```bash
REPO="$PWD"
OUT="$REPO/benchmark-runs/megamoe_urma/trace_$(date +%Y%m%d_%H%M%S)"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:19091
(
  cd /tmp || exit 1
  python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=16 \
    -m deep_ep_experimental.megamoe.urma_benchmark \
    --bs-values 2048,4096,8192 --routing balanced --cv-rows 256 \
    --warmup 2 --iters 10 --trace --timer --trace-capacity 4096 \
    --timer-clock-mhz 1000 --output-dir "$OUT"
)
```

`--trace-capacity`是**每个物理引擎**最多保存的事件数，范围1到65536，默认4096；84个引擎的默认raw缓冲约42 MiB/rank。超过容量时记录dropped，所有rank先交换解析状态，然后统一报错，保留原始bin和错误信息，不会静默导出不完整流水。可在新的输出目录用更大容量重跑。

每rank、每BS的实际产物如下；例如rank0的BS4096位于`$OUT/rank_0/bs_4096/`：

| 文件 | 内容 |
|---|---|
| `urma_timer.bin`、`timer.json` | 原有aggregate timer及schema；来自独立timer调用 |
| `urma_trace.bin` | 逐事件原始little-endian int64存储，按uint64解释 |
| `urma_trace.json` | 可在Perfetto或Chrome Trace Viewer打开的流水 |
| `trace_metadata.json` | rank、容量、各引擎记录数、时钟设置、raw路径或解析错误 |
| `summary.json` | 正式耗时、aggregate带宽及各rank trace导出信息 |

此外，`$OUT/bs_<BS>/allrank.json`保存该BS全部rank的合并流水，`summary.json`的`trace_allrank`记录其路径及统计。不同BS分别导出，避免混用不同调用的时间轴。各rank的零点依旧独立；合并只方便浏览，不能用跨rank横坐标差计算通信延迟或推断真实同时性。

每条轨道对应`block 00..27`的`AIC/AIV0/AIV1`，引擎编号为`block*3+0/1/2`。Pack出现在全部56条AIV轨道；Reduce在28个AIV0及block15～27的13个AIV1上各有一条完整跨度，15个通信AIV1不记录Reduce。SelfDispatch/SelfCombine不再产生拷贝事件，旧事件编号保留供历史trace解析。事件参数包含可用的peer、CV、fragment、expert、行起点/行数、M/N tile位置、列数和字节数；不适用字段显示null。Reduce的行起点/行数表示该reducer的完整token区间，字节数仅为输出BF16写入量。`LocalSync`的CV字段另解释为`sync_stage`：0=pack、3=本次调用的本机退出汇合；1=compute/return、2=rank barrier仅供历史trace解析。

编译和trace运行仍使用上面的命令，不增加调度或采样参数。新计算事件中的`fragment`字段表示专家M组编号，`row`表示expert-major计算行；一个组可以跨多个peer/CV，因此这两个字段为null。`WaitDispatch`保留实际等待的peer/CV及接收行，通信事件仍表示原peer/CV。核对前瞻时按`expert,fragment`观察GMM1与GMM2先后，不能再按peer解释计算轨道。单个核心只记录分给自己的tile，缺少某组的条不等于该组未计算。`WaitActivation`仍表示执行GMM2前尚未满足的依赖；aggregate timer阶段与逐事件schema不变。

| 图中标签 | 应如何理解 |
|---|---|
| `DispatchPutX/DispatchPutScale/CombinePut [PUT submit]` | PUT指令提交区间，不代表远端数据已经完成写入 |
| `DispatchQuiet/CombineQuiet [completion wait]` | 等待之前peer PUT完成的区间 |
| `Gmm1Tile/Gmm2Tile [compute + drain]` | 默认将能够明确对应的AIC提交与排空记录合为一条，包含等待及上下文收尾，不是纯Cube计算时间 |
| `Gmm1Tile/Gmm2Tile [Cube submit]` | 无法明确配对或选择原始显示时，保留任务提交区间 |
| `Gmm1TileDrain/Gmm1Drain/Gmm2Drain [drain]` | 未合并的显式排空记录；AIV上的记录仍单独显示 |
| `Wait*`、`LocalSync`、`RankBarrier`等灰色条 | 数据依赖、核间或rank间同步等待 |
| `Gmm*Dequant`、`ActivationSlice`、`Pack`等 | 对应设备代码实际打点范围 |
| `Reduce` | 每reducer一条完整跨度，含metadata预取、ready等待、计算和最终输出排空，不是纯计算时间 |

每个rank以自己的最早事件归零，**不同rank的零点没有全局对齐**，不能将两个文件的横坐标相减计算跨卡延迟。转换先用整数tick相减再转浮点，保留大时间戳的低位；原始tick在JSON参数中用字符串保存。默认1000 ticks/us仍只是显式约定，可修改`--timer-clock-mhz`，离线解析时应使用同一值。

GMM条合并只影响JSON显示，不改变kernel、同步或打点开销。GMM1的逐tile drain与同一tile合并；组级GMM1/GMM2 drain仅在该AIC于该组只有一个tile时并入，多个tile共享的组级drain保持独立，避免伪造每个tile的完成时间。当前M256、H3584/I1280配置满足单核每组至多一个tile。合并要求原始记录连续且同一引擎/专家/计算组，不能跨越隐藏的等待。原始bin和计数保留；JSON的33/34为合成事件ID，`source_records`保留源事件编号、序号及时间戳，`coalesced_record_count`记录减少的显示条数。

可以在CPU环境离线重新导出单rank，无需启动NPU：

```bash
python3 -m deep_ep_experimental.megamoe.urma_trace \
  "$OUT/rank_0/bs_4096/urma_trace.bin" \
  --output "$OUT/rank_0/bs_4096/urma_trace.json" --rank 0 --clock-mhz 1000
```

已经跑完、各rank JSON保存在同一目录树的历史结果，无需重新跑NPU或重新编译，可从仓库根目录直接合并：

```bash
python3 experiments/megamoe/scripts/urma_merge_traces.py \
  /实际路径/已有运行目录 --bs-values 2048,4096,8192
```

该脚本只需Python标准库，输出同样是`已有运行目录/bs_<BS>/allrank.json`，默认隐藏上述等待条并合并能对应的GMM条；省略`--bs-values`会发现`rank_0/bs_*`下的全部case。历史双机结果需先收集全部16个`rank_*`目录；新版本在线运行则自动完成汇集。离线`--include-waits`只能保留输入JSON已有的等待，恢复已过滤事件需使用原始bin；原始bin本来没有记录的逐token Reduce等待无法恢复。两种离线导出命令均支持`--split-gmm-events`关闭进一步合并；已合并的JSON要恢复原始分条，应从bin重新导出。

逐事件schema为每引擎16个uint64 header，后接`capacity*16`个uint64 record。header前五项为`schema,capacity,count,dropped,engine_id`；record前13项为`start_ticks,end_ticks,event_id,peer,cv,fragment,expert,row,rows,tileM,tileN,columns,bytes`，其余保留位为0。未知元数据为`UINT64_MAX`。解析器检查全部引擎的schema、容量、编号、计数、保留位和事件时间；同引擎事件应按平铺顺序不重叠。

### 压缩成便于复制的文本

脚本只用Python标准库，无需重新编译或安装wheel。优先输入原始`urma_trace.json`，保留等待信息用于归因：

```bash
python3 experiments/megamoe/scripts/urma_trace_to_text.py \
  "$OUT/rank_0/bs_4096/urma_trace.json"
```

默认生成同目录`urma_trace.txt`：包含事件字典、各操作的耗时分布、各引擎记录区间的累计时长/并集/跨度、各peer的提交字节数，以及wait/quiet/drain各自最长的8条。默认不列出数万条重复事件。累计核时不当作墙钟耗时，也不以PUT下发耗时计算链路带宽；记录空白不能直接解释成硬件空闲。

需要逐条还原流水时，用`--detail`；字段名和事件名缩写，保留原始起始时间、耗时及CV/tile信息，不拼接或缩短时间轴。可限制block及引擎，减少复制内容：

```bash
python3 experiments/megamoe/scripts/urma_trace_to_text.py \
  "$OUT/rank_0/bs_4096/urma_trace.json" \
  --detail --cores 0,1,15-17 --engines AIC,AIV1 \
  --output "$OUT/rank_0/bs_4096/selected_cores.txt"
```

详细列表默认省略`wait.sync`，但摘要与最长等待仍保留输入JSON已有的这些数据；增加`--include-waits`可输出其中全部等待条。quiet/drain始终保留。新导出器声明的等待过滤显示`FILTERED_WAITS`；其它源记录数不一致显示`INCOMPLETE`，均不会把缺失的等待误报为0。需要分析完整等待时，先从bin导出含等待的JSON。文本没有同步不同rank的时钟；当前文本脚本只接受单rank URMA trace，不接受`allrank.json`或旧Wave合并格式。

## 下一步优化方向

- 将计数交换、前缀和、CV描述符生成移到设备，减少同步和host工作；完整E2E应始终计入实际准备成本。
- 扫描通信CV行数和专家计算前瞻深度，两者现已独立；进一步改善不均衡路由下固定专家顺序的等待。
- 优化分段L1加载及FIX输出的指令数量，在完整M256收益和非对齐片段搬运成本之间实测取舍；本地展开可按原始token一次读入后复制多个目标行。
- 增加每peer队列的完成进度控制，同时保持独占提交；不能把15个peer owner直接改成多个核竞争同一队列。
- 在正确性与真实全算子收益确认后，才考虑受credit保护的接收/返回缓冲复用，以降低全量中间结果的HBM占用。

## 验证范围

2026-10-19 已通过 CANN 9.2.0 的设备核、Host扩展和完整wheel编译；MegaMoE及runtime的CPU回归为449项通过。CPU检查涵盖路由逆映射、CV/expert交集、28/13核任务覆盖、完成顺序、stream串行及通信域生命周期。尚未进行EP16 UBX上的数值与性能实测，不能将CPU检查或编译通过视为设备精度通过。

2026-10-20 将SwiGLU改为128行子块后，483项CPU回归及CANN 9.2.0完整wheel编译通过。在两台各8张Ascend950DT、同SuperPod的机器上完成EP16功能验证：`--bs-values 2048,4096,8192 --routing balanced --cv-rows 256 --warmup 1 --iters 2 --timer --allow-multi-host`。三个BS均完成路由准备、完整forward、timer诊断及通信域销毁，两台worker正常退出；没有触发原256 KiB UB检查。测试使用Python 3.12.14、Torch 2.12.0及torch_npu 2.12.0.post2，未启用Golden数值校验，短轮次耗时也不作为UBX性能结论。

2026-10-20 逐事件trace版本通过569项CPU回归、全仓pre-commit及CANN 9.2.0完整wheel编译。在另外两台空闲的8卡Ascend950DT、同SuperPod机器上，使用上述命令增加`--trace --trace-capacity 4096`完成EP16验证。三个BS共48份rank/BS流水图均通过完整性校验，84个引擎均有记录，dropped均为0；BS2048/4096/8192每rank分别记录48174/55586/71690条事件，单引擎最大分别为954/1094/1378。正式性能、aggregate timer和详细trace分别运行，两台worker正常退出。本次同样没有Golden数值校验，不据此宣称UBX性能或数值精度通过。

2026-10-20 专家M256版本通过737项MegaMoE/runtime CPU回归、修改文件的pre-commit检查及CANN 9.2.0完整wheel编译。测试覆盖实际C++路由规划器、任意source/CV边界的行映射、L1/L0C分段布局、scale压紧、独立游标调度、self直接返回及旧路径回归；隔离wheel导入确认`urma_compute_schedule=expert-m256`与timer启用。本次只编译和执行CPU测试，没有启动NPU，也没有进行设备数值或性能验证。编译产物与日志保存在`177.2.21.1:/home/codex-deepep/megamoe-urma-expert-m256-20261020/`。

2026-10-20 自动合并/等待过滤版本通过115项Host Store、trace及文本导出测试，包含真实CPU TCPStore传输、单机与非共享目录合并、错误传播、16-rank身份与独立时钟、尾部同步保留和原始时间坐标保持。离线合并脚本另经`python -I -S`验证，不依赖Torch/NPU安装。本次仅修改Python导出层，未修改或启动设备核；测试日志位于`177.2.21.1:/home/codex-deepep/megamoe-urma-allrank-20261020/cpu-tests.log`。

2026-10-20 本分支的`combine-ready-scan-reduce41`通过990项MegaMoE/runtime CPU回归、CANN 9.2.0完整wheel编译和native导入。新增检查直接运行生产Reduce头文件的异步流水模型，覆盖非单调token提交、最后一个token越过阻塞的第一个token、ready正缓存、跨epoch重置、六槽事件复用、精确计数16快路径及完整BF16输出；真实Host planner另验证极端偏斜/专家尾块下的UB与依赖位图容量。

同版wheel在两台各8卡Ascend950DT上完成EP16运行回归。对照`db7d395`，balanced和random各执行BS2048/4096/8192、CV256、warmup=1、iters=2及独立timer调用；两种路由共96个rank/BS的完整BF16输出均有限且逐位相同，各进程正常结束。本次未使用独立Golden；短轮次双机耗时用于功能诊断，不作为UBX单机性能结论。通信协议、41个Reduce worker和每核一条Reduce trace保持不变。
