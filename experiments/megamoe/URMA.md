# MegaMoE URMA 固定 CV 实验

本实验是完整的 Prefill 算子：真实路由准备、本地展开、Dispatch、GMM1、SwiGLU/MXFP8 量化、GMM2、Combine PUT、最终同步和加权 Reduce。没有屏蔽计算，也没有使用预置 balanced 专用路由。

首版范围：默认单机 EP16，固定使用28个AIC / 56个AIV、每个AIV至少192 KiB UB、H=3584、I=1280、384个专家、TopK=8、无共享专家、FP8输入/FP4权重、BS=2048/4096/8192。物理核心数更多的兼容机型仍只启动这28组计算核。任意合法路由均走同一流程；超过配置接收容量的偏斜路由会在启动计算前明确报错。

GMM保留256行tile；SwiGLU在每个已就绪GMM1 tile内按最多128行处理，全部子块写出后只发布一次该tile的完成贡献。其两个BF16输入缓冲合计128 KiB，AIV0的W4解量化流水使用192 KiB，后者决定当前UB最低要求。通信CV大小、专家行顺序及GMM调度不因子块划分改变。资源不足时分别报告核心数或UB容量，并打印实际查询值。

通信接口与固定 CV 调度参考 [ops-transformer PR11602](https://gitcode.com/cann/ops-transformer/pull/11602)（分析版本 `3ab6b8bba2cbdaf3751596c3e4b0488e1991df5f`）。计算复用本仓主线的 FP8/W4 CATLASS 和 SwiGLU 算术；返回方向改为 PUT。

## 数据与通信顺序

1. 每个输入 `[token, topk_slot]` 都是独立路由实例。按 `[目标rank][目标本地expert][原始token/topk位置]` 稳定排序，建立 `order` 和逆映射 `reduce_index[token,slot]=expanded_row`。
2. 56个AIV共同展开FP8 token和E8M0 scale：远端路由写连续发送区，self路由直接写本rank接收区；不再从发送区执行一次SelfDispatch拷贝。一个token命中多个expert时复制对应多行，本版不做去重。
3. 每个目标rank的连续行流按 `cv_rows` 截断，通过 URMA PUT 发送。token与scale使用各自连续平面。
4. 每个CV与expert区间取交集，产生 `source_rank、cv_id、expert、cv_row、expert_row、rows、source_expanded_row` 描述符。GMM1、SwiGLU、GMM2共用行身份；跨expert边界必须分别计算。
5. 计算按通信CV前瞻一批：先执行首个CV的GMM1，随后交替执行下一个CV的GMM1和上一个CV的GMM2，最后执行末个CV的GMM2。SwiGLU按已完成的gate/up tile独立推进；GMM2仍等待对应CV/expert交集的全部SwiGLU tile完成。远端结果写按来源rank排列的连续GMM2输出区，返回PUT等待整个CV的完整H列；self结果直接写Reduce最终读取的返回区，不再执行SelfCombine拷贝。
6. 本rank的15个远端Dispatch全部完成且self接收区已就绪后才提交Combine PUT，此处不插入全卡同步。Combine延后quiet，所有PUT完成后参加最终全卡同步，再由56个AIV按`reduce_index`读取各top-k结果、乘路由权重并求和。

15个通信AIV1各独占一个远端rank。通信worker `w` 的peer为 `w < rank ? w : w+1`；剩余13个AIV1执行SwiGLU。self就绪标志在Pack本地同步后发布，原来承担self拷贝的block15随后直接进入SwiGLU。远端两个阶段都只使用PUT，队列同一peer始终只有一个提交者。

Pack和Reduce是计算前后的独立本地阶段，使用28组中的全部AIV0/AIV1，共56个AIV；任务编号为`block*2+subblock`，按56核划分行。阶段结束仍保留原有DMA完成及本地同步，再复用UB进行计算或结束调用。当前内核固定启动28组；即使物理设备有64个AIV，也只使用已启动的56个，不扩大计算/通信核数量。

self直接写地址按现有`send_base/receive_base`映射，远端PUT的地址、顺序、CV和数量不变，计算仍按来源rank外循环、CV内循环。没有在本次改动中加入跨peer交错。当前Pack完成前不会对peer发布新epoch的Start；上一轮本地计算/返回/Reduce由同一runtime stream串行完成，因此self直接写不会覆盖尚在使用的上一轮数据。

接收数据/中间结果/返回结果不按CV覆盖复用。所有BS共享固定对称heap布局，epoch由`UrmaBuffer`统一递增，不因创建新Plan或切换BS重新归零。

通信域启用MTE和UDMA，每个对端独立QP0，不使用共享jetty。MTE只用于完成标志与本地搬运；远端token、scale和FFN结果均走UDMA PUT。每次调用在同一runtime stream串行执行，并在下一次写入前与各peer握手，保护上一轮Reduce尚未读完的结果。

### 计算按通信 CV 前瞻一批

这里的一个CV是**单个来源rank发送的一段连续行**，用`(source_rank, cv_id)`唯一标识。不同来源rank的同号CV彼此独立，不会合成一批。计算沿现有描述符顺序遍历；一个CV跨越多个expert时，先完成该CV内所有expert交集的GMM1，再按下面的顺序安排GMM2：

```text
GMM1(CV0)
for i = 0 .. 最后一个CV之前:
    GMM1(CV[i+1])
    GMM2(CV[i])
GMM2(最后一个CV)
```

只有一个非空CV时直接执行`GMM1→GMM2`；没有接收行时没有计算任务。前瞻把下一个CV的GMM1放到当前CV的GMM2之前，让13个SwiGLU核在这段时间处理已就绪的当前CV。28组AIC/AIV0仍按原有全局tile编号分工，不改变每个CV的行、专家边界、tile数量或输出位置。

各AIC及其配对AIV0按本核任务独立推进，CV批次之间没有barrier。这里`GMM1(CV)`完成指本核负责的任务结束，并不代表全部核心同时完成该CV；GMM2仍检查对应fragment的真实SwiGLU完成计数。每段GMM结束后，配对AIV0向本组AIC发送End确认，确保旧段已消费完6–9号流水槽信号后才复用；此确认只涉及该对AIC/AIV0，不让15个通信AIV1或13个SwiGLU AIV1参与同步。

该调度默认生效，无需新开关。它保留Dispatch就绪检查、`WaitActivation`、原有计算流水完成等待及计数发布，不增加全核或全卡同步；PUT顺序、payload、量化算术、路由权重和Reduce公式均保持原有定义。它能否减少等待需要由正式耗时与trace验证，不能仅凭前瞻顺序推定收益。

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

`prepare_routes`每次在输入NPU上执行排序、计数、前缀和及逆映射。首版随后把384个counts复制到CPU，经Store交换完整计数表，native据此生成CV描述符并送回NPU。它包含设备同步和host控制开销，是后续需要优化的一部分。

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

每条轨道对应`block 00..27`的`AIC/AIV0/AIV1`，引擎编号为`block*3+0/1/2`。Pack和Reduce现在同时出现在AIV0/AIV1轨道；SelfDispatch/SelfCombine不再产生拷贝事件，旧事件编号保留供历史trace解析。事件参数包含可用的peer、CV、fragment、expert、行起点/行数、M/N tile位置、列数和字节数；不适用字段显示null。`LocalSync`的CV字段另解释为`sync_stage`：0=pack、1=compute/return、2=rank barrier、3=reduce。

编译和trace运行仍使用上面的命令，不增加调度或采样参数。核对前瞻顺序时，在AIC/AIV0轨道按`(peer, cv)`合并同一CV的fragment，观察GMM1与GMM2条的先后；fragment/expert元数据沿用原编号，因此重排后编号可以往回跳。单个核心只记录分给自己的tile，缺少某CV的条不等于该CV未计算。`WaitActivation`仍会出现，表示该核心实际执行GMM2前尚未满足的依赖；aggregate timer阶段与逐事件schema均不变。

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

详细列表默认省略`wait.sync`，但摘要与最长等待仍保留这些数据；增加`--include-waits`可输出全部等待条。quiet/drain始终保留。已过滤的JSON仍可读取，源记录数不一致时标注`INCOMPLETE`，不会把缺失的等待误报为0。文本没有同步不同rank的时钟；当前脚本只接受单rank URMA trace，不接受旧Wave合并格式。

## 下一步优化方向

- 将计数交换、前缀和、CV描述符生成移到设备，减少同步和host工作；完整E2E应始终计入实际准备成本。
- 扫描CV行数和Mtile：当前均衡场景每来源每expert平均仅42.7/85.3/170.7行，大CV不会消除不同expert或来源的M尾块。
- 将GMM2就绪条件从整个CV/expert交集细化到完整I列的Mtile，缩短等待；本地展开可按原始token一次读入后复制多个目标行。
- 增加每peer队列的完成进度控制，同时保持独占提交；不能把15个peer owner直接改成多个核竞争同一队列。
- 在正确性与真实全算子收益确认后，才考虑受credit保护的接收/返回缓冲复用，以降低全量中间结果的HBM占用。

## 验证范围

2026-10-19 已通过 CANN 9.2.0 的设备核、Host扩展和完整wheel编译；MegaMoE及runtime的CPU回归为449项通过。CPU检查涵盖路由逆映射、CV/expert交集、28/13核任务覆盖、完成顺序、stream串行及通信域生命周期。尚未进行EP16 UBX上的数值与性能实测，不能将CPU检查或编译通过视为设备精度通过。

2026-10-20 将SwiGLU改为128行子块后，483项CPU回归及CANN 9.2.0完整wheel编译通过。在两台各8张Ascend950DT、同SuperPod的机器上完成EP16功能验证：`--bs-values 2048,4096,8192 --routing balanced --cv-rows 256 --warmup 1 --iters 2 --timer --allow-multi-host`。三个BS均完成路由准备、完整forward、timer诊断及通信域销毁，两台worker正常退出；没有触发原256 KiB UB检查。测试使用Python 3.12.14、Torch 2.12.0及torch_npu 2.12.0.post2，未启用Golden数值校验，短轮次耗时也不作为UBX性能结论。

2026-10-20 逐事件trace版本通过569项CPU回归、全仓pre-commit及CANN 9.2.0完整wheel编译。在另外两台空闲的8卡Ascend950DT、同SuperPod机器上，使用上述命令增加`--trace --trace-capacity 4096`完成EP16验证。三个BS共48份rank/BS流水图均通过完整性校验，84个引擎均有记录，dropped均为0；BS2048/4096/8192每rank分别记录48174/55586/71690条事件，单引擎最大分别为954/1094/1378。正式性能、aggregate timer和详细trace分别运行，两台worker正常退出。本次同样没有Golden数值校验，不据此宣称UBX性能或数值精度通过。
