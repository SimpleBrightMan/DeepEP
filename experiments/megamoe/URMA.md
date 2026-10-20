# MegaMoE URMA 固定 CV 实验

本实验是完整的 Prefill 算子：真实路由准备、本地展开、Dispatch、GMM1、SwiGLU/MXFP8 量化、GMM2、Combine PUT、最终同步和加权 Reduce。没有屏蔽计算，也没有使用预置 balanced 专用路由。

首版范围：默认单机 EP16，固定使用28个AIC / 56个AIV、每个AIV至少192 KiB UB、H=3584、I=1280、384个专家、TopK=8、无共享专家、FP8输入/FP4权重、BS=2048/4096/8192。物理核心数更多的兼容机型仍只启动这28组计算核。任意合法路由均走同一流程；超过配置接收容量的偏斜路由会在启动计算前明确报错。

GMM按专家聚合不同来源的行，使用M256计算块，仅专家尾块不足256行；通信CV不再决定计算分块。SwiGLU在每个已就绪GMM1 tile内按最多128行处理，全部子块写出后只发布一次该tile的完成贡献。其两个BF16输入缓冲合计128 KiB，AIV0的W4解量化流水使用192 KiB，后者决定当前UB最低要求。资源不足时分别报告核心数或UB容量，并打印实际查询值。

通信接口与固定 CV 调度参考 [ops-transformer PR11602](https://gitcode.com/cann/ops-transformer/pull/11602)（分析版本 `3ab6b8bba2cbdaf3751596c3e4b0488e1991df5f`）。计算复用本仓主线的 FP8/W4 CATLASS 和 SwiGLU 算术；返回方向改为 PUT。

## 数据与通信顺序

1. 每个输入 `[token, topk_slot]` 都是独立路由实例。按 `[目标rank][目标本地expert][原始token/topk位置]` 稳定排序，建立 `order` 和逆映射 `reduce_index[token,slot]=expanded_row`。
2. 56个AIV共同展开FP8 token和E8M0 scale：远端路由写连续发送区，self路由直接写本rank接收区；不再从发送区执行一次SelfDispatch拷贝。一个token命中多个expert时复制对应多行，本版不做去重。
3. 每个目标rank的连续行流按 `cv_rows` 截断，通过 URMA PUT 发送。token与scale使用各自连续平面。
4. 对每个本地expert，把16个来源的行按source编号连接，按M256划分计算组；专家尾组可以不足256，不与下一个专家混算。每组的Segment记录来源、CV、接收行、组内行和返回展开行。Segment在source/CV边界切分，但这些边界不再切分矩阵计算。
5. GMM1按Segment把X分段加载到同一L1计算块，只等待该组实际涉及的Dispatch CV及scale准备。scale由13个AIV1独立准备，每个M组只聚合一次到expert-major GM；AIC沿用连续MX scale加载。GMM1输出、SwiGLU及GMM2输入使用连续的expert-major计算行序。GMM2按同一映射将远端结果分段写回原source-major连续输出区，self结果直接写Reduce最终读取的位置。各peer的Combine CV等待所有覆盖它的M组完成全部H列，再按原连续区间PUT。
6. 本rank的15个远端Dispatch全部完成且self接收区已就绪后才提交Combine PUT，此处不插入全卡同步。Combine延后quiet，所有PUT完成后参加最终全卡同步，再由56个AIV按`reduce_index`读取各top-k结果、乘路由权重并求和。

15个通信AIV1各独占一个远端rank。通信worker `w` 的peer为 `w < rank ? w : w+1`；剩余13个AIV1执行SwiGLU。self就绪标志在Pack本地同步后发布，原来承担self拷贝的block15随后直接进入SwiGLU。远端两个阶段都只使用PUT，队列同一peer始终只有一个提交者。

Pack和Reduce是计算前后的独立本地阶段，使用28组中的全部AIV0/AIV1，共56个AIV；任务编号为`block*2+subblock`，按56核划分行。阶段结束仍保留原有DMA完成及本地同步，再复用UB进行计算或结束调用。当前内核固定启动28组；即使物理设备有64个AIV，也只使用已启动的56个，不扩大计算/通信核数量。

本地阶段使用多槽流水，benchmark打印`local_pipeline=pack6-reduce6-output2`。每核分配连续行区间，各核行数与原来的跨步分配完全一致：

- Pack使用6个token/scale槽，按槽等待上一次MTE3完成，再下发新的MTE2；不同槽的读取与写出可以重叠。`order`批量DMA到UB，当前支持的最大BS8192每核最多1171行，一次预取即可，payload循环不再从GM逐行标量读取路由。UB占用65.5 KiB。
- Reduce使用6个BF16/FP32输入槽、独立FP32累加器和2个BF16输出槽。`inverse + weights`按最多128个token批量搬入UB，输入预取与向量乘加、前一个token输出写入可以重叠。UB占用166 KiB，低于现有192 KiB最低要求。
- 返回布局、FP32乘加顺序和最终BF16舍入不变。self直接写receive/return，远端PUT和通信CV不变。Pack仍全部完成后才进入Dispatch，Reduce仍在尾部全卡同步后执行；本次不将它们融入通信/计算阶段。

若打印`local_pipeline=single-slot`，加载的native扩展仍为旧版，需重新编译安装。正式耗时仍取关闭timer/trace的`CACHED_FORWARD`；比较Pack/Reduce阶段时使用同rank全部56个AIV记录的`max(last_tick)-min(first_tick)`。原timer的Pack字节统计写入的FP8+scale，Reduce字节仅统计输出BF16，均不是完整HBM读写流量。

self直接写地址按现有`send_base/receive_base`映射；同一计算组可同时含self和远端行，逐Segment决定输出地址。远端PUT的地址、顺序、CV和数量不变，计算按expert顺序推进多个peer的返回结果。当前Pack完成前不会对peer发布新epoch的Start；上一轮本地计算/返回/Reduce由同一runtime stream串行完成，因此self直接写不会覆盖尚在使用的上一轮数据。

接收数据/中间结果/返回结果不按CV覆盖复用。所有BS共享固定对称heap布局，epoch由`UrmaBuffer`统一递增，不因创建新Plan或切换BS重新归零。

通信域启用MTE和UDMA，每个对端独立QP0，不使用共享jetty。MTE只用于完成标志与本地搬运；远端token、scale和FFN结果均走UDMA PUT。每次调用在同一runtime stream串行执行，并在下一次写入前与各peer握手，保护上一轮Reduce尚未读完的结果。

### 专家 M256 与通信 CV 解耦

通信CV仍是**单个来源rank发送的一段连续行**，用`(source_rank, cv_id)`唯一标识。计算批次改为一个本地专家的全部M256组，同专家内不混入其它专家。每个M组有独立的输入依赖列表，不需要等齐全16个来源。例如BS2048均衡时，专家0第一个M256由来源0～5分别贡献43、43、42、43、43、42行。

计算前瞻按专家批次进行：

```text
GMM1(expert0)
for e = 0 .. localExperts-2:
    GMM1(expert[e+1])
    GMM2(expert[e])
GMM2(expert[last])
```

MX scale在L1中的zZ布局按16行组织，任意来源边界可能落在2字节偏移，不能直接调用要求32字节LocalTensor起址的DMA。为此只增加`receiveRows*(H/32)`字节的scale GM工作区：准备时UB每行补齐到32字节，写回GM时压紧；不新增完整X重排。该缓冲在每次调用重新填充，组完成标志逐次清零。13个AIV1分别推进scale准备和activation两个游标，只有依赖就绪才执行任务，避免等待GMM1时阻止后续scale准备。trace中的`InputScaleGather`记录该额外搬运，`WaitInputScale`记录AIC等待。

Scale准备只在详细trace中独立计时，未归入原有8个aggregate phase；因此不能用这些phase的累计时间代表完整算子耗时。正式`CACHED_FORWARD`和`FULL_E2E`均包含该步骤。

空专家没有计算任务。各AIC及其配对AIV0按本核任务独立推进，专家批次之间没有全核barrier；GMM1下一批可以与上一批的SwiGLU/GMM2重叠。13个SwiGLU核使用新的M组任务编号，GMM2等待同组全部1280列activation完成，不能只等待其中一个N tile。

返回CV保存它所依赖的M组编号列表，同一组去重后只出现一次。每组全部GMM2 N tile完成分段输出后才满足依赖；仅提交Cube/FIX指令不能提前放行PUT。BS2048时一个peer的CV0大致跨6个专家，新顺序让多个peer按CV波次逐渐就绪，不保证15个peer同时开始。

每段GMM结束后仍由配对AIV0向本组AIC发送End确认，确保旧段消费完流水槽信号后才复用；该确认不涉及通信或SwiGLU AIV1。接收数据读取保留URMA所需的cache bypass。

默认均衡模型的每rank任务数如下，CV128/256/1024等取值不改变这些计算组数：

| BS | M256组（含专家尾组） | GMM1逻辑tile | GMM2 tile |
|---|---:|---:|---:|
| 2048 | 72 | 360 | 1008 |
| 4096 | 144 | 720 | 2016 |
| 8192 | 264 | 1320 | 3696 |

该调度默认生效，无需新开关。benchmark打印`[URMA SCHEDULE] compute=expert-m256`并写入manifest；若显示`source-cv`，当前Python加载的还是旧native扩展，需要重新编译安装。流水图和正式耗时仍需实测验证；不能将tile数量减少倍数视为耗时加速倍数。

### 本地多槽流水 A/B 验证（2026-10-20）

在两台Ascend950DT（每机8卡）上对比本次修改与`e319740`对应的单槽本地实现，通信和专家M256计算不变。H3584/I1280/E384/TopK8、balanced、CV256，每次调用仍启动28 AIC/56 AIV。两机全部设备健康OK且空闲后才运行，旧新分别创建独立通信域。

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

buffer = UrmaBuffer(group, num_max_tokens_per_rank=8192, cv_rows=256)
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

同一buffer的native调用统一入队到owner专用NPU stream。调用前等待caller stream的输入/描述符写入，返回时在caller stream插入输出就绪等待；不同caller stream也不会并发覆盖同一对称workspace。`destroy()`同步专用stream后再集体清理。

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
| `CACHED_FORWARD` | 本地展开、两次GMM、SwiGLU、双向阶段PUT、最终同步和Reduce | NPU Event区间，不含预先准备路由 |
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

`--trace`现在自动合并全部rank，每种BS生成`$OUT/bs_<BS>/allrank.json`，可直接在Perfetto/Chrome Trace Viewer打开。每个rank作为独立process，内部保留84条AIC/AIV轨道，按rank编号排列；原有每rank JSON也保留。默认导出的两种JSON均隐藏`Wait*`及普通`LocalSync`，保留尾部`FinalPeerHandshake`、`RankBarrier`和`LocalSync(sync_stage=2)`。`Quiet/Drain`继续显示实际通信完成等待和流水排空。过滤不移动时间戳、不压缩空白，不修改设备同步、原始bin或带宽统计。

如需完整等待条，运行时增加`--trace-include-waits`；也可从已有bin通过下方离线导出命令增加`--include-waits`恢复。只传`--timer`仍仅导出aggregate统计，逐事件流水及`allrank.json`需要`--trace`。

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

每条轨道对应`block 00..27`的`AIC/AIV0/AIV1`，引擎编号为`block*3+0/1/2`。Pack和Reduce现在同时出现在AIV0/AIV1轨道；SelfDispatch/SelfCombine不再产生拷贝事件，旧事件编号保留供历史trace解析。事件参数包含可用的peer、CV、fragment、expert、行起点/行数、M/N tile位置、列数和字节数；不适用字段显示null。`LocalSync`的CV字段另解释为`sync_stage`：0=pack、1=compute/return、2=rank barrier、3=reduce。

编译和trace运行仍使用上面的命令，不增加调度或采样参数。新计算事件中的`fragment`字段表示专家M组编号，`row`表示expert-major计算行；一个组可以跨多个peer/CV，因此这两个字段为null。`WaitDispatch`保留实际等待的peer/CV及接收行，通信事件仍表示原peer/CV。核对前瞻时按`expert,fragment`观察GMM1与GMM2先后，不能再按peer解释计算轨道。单个核心只记录分给自己的tile，缺少某组的条不等于该组未计算。`WaitActivation`仍表示执行GMM2前尚未满足的依赖；aggregate timer阶段与逐事件schema不变。

| 图中标签 | 应如何理解 |
|---|---|
| `DispatchPutX/DispatchPutScale/CombinePut [PUT submit]` | PUT指令提交区间，不代表远端数据已经完成写入 |
| `DispatchQuiet/CombineQuiet [completion wait]` | 等待之前peer PUT完成的区间 |
| `Gmm1Tile/Gmm2Tile [Cube submit]` | Cube任务提交区间，不伪造异步Cube执行完成时刻 |
| `Gmm1TileDrain/Gmm1Drain/Gmm2Drain [drain]` | 显式等待计算流水排空的区间 |
| `Wait*`、`LocalSync`、`RankBarrier`等灰色条 | 数据依赖、核间或rank间同步等待 |
| `Gmm*Dequant`、`ActivationSlice`、`Pack`、`Reduce`等 | 对应设备代码实际打点范围 |

每个rank以自己的最早事件归零，**不同rank的零点没有全局对齐**，不能将两个文件的横坐标相减计算跨卡延迟。转换先用整数tick相减再转浮点，保留大时间戳的低位；原始tick在JSON参数中用字符串保存。默认1000 ticks/us仍只是显式约定，可修改`--timer-clock-mhz`，离线解析时应使用同一值。

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

该脚本只需Python标准库，输出同样是`已有运行目录/bs_<BS>/allrank.json`，默认隐藏上述等待条；省略`--bs-values`会发现`rank_0/bs_*`下的全部case。历史双机结果需先收集全部16个`rank_*`目录；新版本在线运行则自动完成汇集。离线`--include-waits`只能保留输入JSON已有的等待，恢复已过滤事件需使用原始bin。

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
