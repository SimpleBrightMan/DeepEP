# MegaMoE EP16 通信解耦模拟与核心配比实验

默认运行独立 `mega_moe_communication_kernel`，复用 MegaMoE 的真实路由、Dispatch、Combine、Unpermute 与 Wave 规划。
通信 kernel 不执行 GMM1、SwiGLU、GMM2，不创建权重。Combine 所需 BF16 专家输出在测试前一次性生成并写入原工作区，测量中没有 payload 生成或 H2D。
原完整 `benchmark` 保留，作为显式开启的全算子通信时间对照。

固定 H=3584、I=1280、E=384、K=8、S=0、EP16、BS=2048/4096/8192，沿用仓库 rank-first balanced 路由。
全部 28 个物理核心保持启动。连续四个 rank 为一组，INTRA 包含本卡搬运。

## 构建和运行

激活匹配的 Python/Torch 环境并 source CANN，设置完整 SHMEM SDK 的 `SHMEM_ROOT`。
此次包含新设备 kernel 与 Python API，必须重新构建安装，不能复用旧二进制。
审计计数已改为每核独占 128B 缓存行，timer buffer 布局随之更新；Python 包和设备库必须一起更新。

```bash
git fetch origin
git switch codex/megamoe-comm-core-sweep
git pull --ff-only
export DEEPEP_BUILD_MEGAMOE=ON
export DEEPEP_MEGAMOE_TIMER=ON
bash experiments/megamoe/scripts/build.sh

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:19091
export LD_LIBRARY_PATH="${SHMEM_ROOT:-$PWD/third-party/shmem/install/shmem}/lib:$ASCEND_HOME_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 仅运行通信模拟，保留 timer 原始文件；不做数据/逐 token 核对或完整算子时间对比。
python3 experiments/megamoe/scripts/comm_core_sweep.py \
  --mode baseline --bs-values 2048,4096,8192 \
  --dispatch-metadata-mode dma \
  --no-validation --keep-raw-timers

# 同一构建的原 scalar metadata 对照，保持任务、槽位及其它参数相同。
python3 experiments/megamoe/scripts/comm_core_sweep.py \
  --mode baseline --bs-values 2048,4096,8192 \
  --dispatch-metadata-mode scalar \
  --no-validation --keep-raw-timers
```

`--mode baseline` 清除继承的 Dispatch / Combine 分核变量，使用原任务分配。
不加 `--run-full-reference` 时只运行通信模拟，适合后续快速实验。
`--no-validation` 跳过 CPU Golden、数据 snapshot、逐 token audit 和任务一致性核对，仍采集 timer-off kernel 耗时和普通 timer。
当前 GB/s 指标的分子来自普通 timer 中实际记录的行数，分母为全部 28 个 AIV1 的通信累计时间均值；Dispatch、Combine 分开计算。
这是历史的平均核时归一化指标，不是真实链路带宽，也不等于按通信首尾时间跨度计算的有效带宽。比较优化效果以 timer-off 通信 kernel 耗时为主，结合原始 timer 检查瓶颈；本轮不修改统计实现。
报告标记 `validation=skipped`，保留每 Wave/每核时间和行数，不生成逐 token 审计 CSV。
需要流水图时，可对结果内 `bs_<BS>/baseline/bs_<BS>/timer/` 运行 `python3 -m deep_ep_experimental.megamoe.timer /实际timer目录 --ranks 0-15`。

## Dispatch metadata DMA 实验

通信模拟默认 `--dispatch-metadata-mode dma`，`scalar` 保留原逐 token metadata 写法供同构建对照。
此开关只作用于通信实验；完整 MegaMoE 算子和 `--run-full-reference` 仍使用原 scalar metadata 路径。
`baseline` 指原核心任务分配；metadata 算法由该独立开关指定。

DMA 模式在计时前生成本地 GM 只读身份表，每行 32B，含源 rank、源 token、TopK 槽位。
Dispatch 通过 MTE2 把对应表项读到 UB，然后随 token/scale 一起由 MTE3 写入原 metadata 位置，去掉逐 token scalar metadata 写入及其 MTE3→S 等待。
每卡表大小为 `EP × BS × TopK × 32`：BS2048/4096/8192 分别为 8/16/32 MiB。
每个 dispatch token 额外读取 32B **本地** GM，不增加跨卡通信 payload；带宽分子保持原定义。
表的生成与准备在 timer-off profiler 和普通 timer 调用之前完成，表读取包含在通信计时内。

两种模式沿用相同 Wave、每核任务、槽位数量、ready 协议和 Dispatch/Combine timer 边界。
默认 Combine 仍为原 2 槽；下面的独立开关可测试 4/6 槽。批量读取路由索引所需 `MTE2_S`、slice ready 同步和槽位复用等待仍保留；metadata 实验不代表去除了全部 scalar 依赖。
`plan.json`、rank manifest、详细通信报告、`summary.json` / `summary.csv` 和控制台均记录 `dispatch_metadata_mode`，防止结果混淆。

## Combine 流水槽位实验

`--combine-buffer-count {2,4,6}` 只改变通信模拟的 Prefill Combine 流水槽位，默认 2。
完整 MegaMoE 及 `--run-full-reference` 仍按原策略使用 2 槽；Dispatch metadata 默认仍为 DMA，Dispatch 槽位和核心任务分配不受此开关影响。
Wave、每核 token 任务和传输字节数、ready 协议、两类通信 timer 边界保持原定义。
增加槽位扩大可同时在途的 Combine 搬运数量，是否改善带宽需要实测，不能由槽位数直接推断。

重新编译安装一次后，在同一构建下循环测试 2/4/6 槽，每个槽位跑 BS2048/4096/8192：

```bash
for slots in 2 4 6; do
  python3 experiments/megamoe/scripts/comm_core_sweep.py \
    --mode baseline --bs-values 2048,4096,8192 \
    --dispatch-metadata-mode dma --combine-buffer-count "$slots" \
    --no-validation --keep-raw-timers || break
done
```

每轮自动创建独立结果目录，不画图。控制台和 plan、manifest、详细报告、汇总 JSON/CSV 均记录 `combine_buffer_count`。
输出仍分别给出 Dispatch、Combine 的带宽和通信 kernel 总耗时；以相同 BS 的三种槽位结果比较。
省略 `combine_buffer_count` 字段的旧报告按原 2 槽读取，不能作为 4/6 槽结果接受。

## 分核 Combine metadata 批量预取

共享的 `RunCommSplitCombineSlice` 路径默认合并同一 slice 内、本核心负责的各 peer 段 metadata 预取，每批最多 1536 行。
先下发这一批各段的 metadata MTE2 搬运，再统一执行一次 `MTE2_S`，随后依原段顺序消费 metadata 并发起 payload PUT；不足一批的尾部使用实际行数。
不同 peer 段仍可对应不同 GM 地址，不要求它们在 GM 连续。优化减少原先逐 peer 段等待 metadata 的次数，保留每批 metadata 就绪所需的 scalar 同步，不代表消除了全部 scalar 依赖。

此优化自动用于设置了 Combine INTRA 核数的分核路径，没有新增 CLI 开关；完整 MegaMoE 若显式开启 Combine 分核，同样采用此优化。
原有 peer/组顺序、每 Wave 每核的 token 任务及字节数、ready 协议和 timer 边界保持不变，也不增加全卡同步或切组排空。
Combine 槽位仍由 `--combine-buffer-count` 控制：全局默认保持 2，下面的测试显式使用 6 槽；本次 metadata 优化不改变所选槽位数量。
非分核 `baseline` 及未开启 Combine 分核的完整 MegaMoE 路径不改；默认完整算子与 `--run-full-reference` 保持未开启 Combine 分核的行为。

重新编译安装后，用当前已测的 4/24、XOR、Dispatch DMA、Combine 6 槽配置跑三个 BS：

```bash
python3 experiments/megamoe/scripts/comm_core_sweep.py \
  --mode paired-groups --bs-values 2048,4096,8192 \
  --group-intra-cores 4 --group-order paired-xor \
  --dispatch-metadata-mode dma --combine-buffer-count 6 \
  --no-validation --keep-raw-timers
```

每个 BS 只跑一个 `paired_groups_d4_c4` case，不做数据核对或完整算子时间对比，保留原始 timer。
与更新前相同命令的结果比较 timer-off `COMM-KERNEL rank-max-mean`，再检查 Combine 各核的累计时间和尾部；控制台旧 GB/s 指标只作为既有口径记录，不据此宣称链路峰值提升。

## 配对组轮转与核心配比控制实验

`--mode paired-groups` 默认每个 BS 运行一次 12/16 配置：Dispatch、Combine 均由 AIV1 核 0..11 处理 INTRA，核 12..27 处理 CROSS。
可用 `--group-intra-cores 12,4,6` 按列出的顺序运行三种配置；Dispatch、Combine 的 INTRA 核数同时取该值，CROSS 为 `28-INTRA`。
每种配置各运行一次，不增加 baseline、选优或 confirm。核心数取不同的整数 `1..27`，两种通信分别按各自任务量做均分。

INTRA 包含本卡和同组其它三卡；`--group-order paired-xor` 为默认顺序，CROSS 按 `peer_group = own_group XOR phase`，依次 `phase=1,2,3` 处理：

| 逻辑阶段 | 配对组 |
|---|---|
| 1 | G0↔G1，G2↔G3 |
| 2 | G0↔G2，G1↔G3 |
| 3 | G0↔G3，G1↔G2 |

这只是各核发起通信的先后顺序，没有新增阶段全卡同步、Host 同步或切组流水排空。
各 rank/核心可因执行速度不同进入不同逻辑阶段，因此运行时不保证严格只有两组互通。
`--group-order reverse` 则使用原错峰顺序 `(own_group-phase+4)%4`；可在相同配比下对比两种顺序。
同一 kernel 内 INTRA 与 CROSS 并发；保留每 Wave 的 token 集合、搬运字节和现有 timer 边界，核心归属按所选配比重新划分。
组内任务继续采用余数轮转均分，避免无法整除的余数一直落在固定核心。

更新并按前文重新编译后运行（显式选择当前已测的 4/24 配比、XOR 顺序和 Combine 6 槽；CLI 默认配比仍为 12/16）：

```bash
python3 experiments/megamoe/scripts/comm_core_sweep.py \
  --mode paired-groups --bs-values 2048,4096,8192 \
  --group-intra-cores 4 --group-order paired-xor \
  --dispatch-metadata-mode dma --combine-buffer-count 6 \
  --no-validation --keep-raw-timers
```

针对 BS4096，一次编译后做两种组顺序 × 12/4/6 INTRA 核的控制实验：

```bash
for order in paired-xor reverse; do
  python3 experiments/megamoe/scripts/comm_core_sweep.py \
    --mode paired-groups --bs-values 4096 \
    --group-intra-cores 12,4,6 --group-order "$order" \
    --dispatch-metadata-mode dma --combine-buffer-count 6 \
    --no-validation --keep-raw-timers || break
done
```

总计六个 case，每种组顺序自动生成一个独立结果根目录；case 路径为 `bs_<BS>/paired_groups_d<N>_c<N>/`，例如 `paired_groups_d4_c4`。
添加 `--plan` 可只打印顺序和 case 数量，不访问设备、不创建文件。没有绘图步骤。
带宽仍按 Dispatch 和 Combine 分别统计。`plan.json`、rank manifest、详细报告、汇总 JSON/CSV 及控制台记录实际 `group_order`。
固定分工时额外输出四项分类带宽：Dispatch INTRA/CROSS、Combine INTRA/CROSS。
INTRA 为该类字节数除以 **N 个 INTRA 核累计通信耗时的均值**，CROSS 则除以 **28−N 个 CROSS 核累计通信耗时的均值**；两者单位仍为 GB/s/card，控制台同时显示 workers。
这是分类的平均核时归一化指标，不是整段 class-envelope 时间，也不是真实链路带宽；两类指标不能直接相加得到总带宽。
原 Dispatch/Combine 总指标仍用全部 28 核累计通信时间的均值作分母，定义不变。
原 baseline 核心同时承担混合拓扑任务，不生成无法从 timer 区分的分类带宽，CSV 对应列留空。
driver 在选择 paired-xor 时设置 `DEEPEP_MEGAMOE_COMM_GROUP_ORDER=paired-xor`；选择 reverse、其它模式及完整 reference 都清除此继承变量，保持原 reverse 顺序。
旧报告缺少 `group_order` 字段时按 `reverse` 解释，不能作为 paired-xor 结果接受。

`--mode baseline` 仍保留原 28 核分工；`--combine-buffer-count` 的全局默认仍为 2。
`--group-intra-cores`、`--group-order` 仅用于 paired-groups；新的配比列表不能与旧 `--intra-cores` / Dispatch / Combine 扫描参数混用。
旧的 `--mode paired-groups --intra-cores 12` 仍兼容默认 12/16 配置。
paired-groups 不接受原算子时间比较参数；需要数据/逐任务验收时移除 `--no-validation` 即可，验收使用相同配比和组顺序的 oracle。

## 可选的解耦验收

下面命令在每个 BS 先跑本分支完整 MegaMoE，再跑通信模拟，最后比较每核时间；它不是只跑 main 原算子。
两者均使用原 Dispatch/Combine 任务分配，完整版本还执行 GMM1、SwiGLU 和 GMM2。

```bash
python3 experiments/megamoe/scripts/comm_core_sweep.py \
  --mode baseline --bs-values 2048,4096,8192 \
  --dispatch-metadata-mode scalar \
  --run-full-reference --timing-rtol 0.10 --require-timing-match
```

验收与 `--no-validation` 互斥；无需每轮性能实验都执行验收。
上例显式选择 scalar 以对照原通信路径；也可用 DMA 对照完整算子，比较器不要求 metadata 模式相同，仍按相同任务与指定时间容差比较。

`--timing-rtol 0.10` 表示 10% 相对容差，可增加 `--timing-atol-us`。
对每个 rank、每个核、每种通信分别检验：

```text
abs(simulated_core_total_us - full_megamoe_core_total_us)
    <= timing_atol_us + timing_rtol * full_megamoe_core_total_us
```

比较前先检查任务完全一致，包括 Wave、物理核、专家、专家行、源 rank/token、TopK 槽位。
任务不匹配直接报错。`--require-timing-match` 使任意核时间偏差超标也返回失败。
不设严格退出时，超标最终显示 `COMPLETED_TIMING_MISMATCH`，不会宣称时间一致。
报告 `bs_<BS>/full_megamoe_comparison.json` 保存每个核的原算子时间、模拟时间、差值、比值与逐 Wave 诊断。

也可复用之前完整 MegaMoE 扫描的输出根目录，要求包含 `bs_<BS>/baseline/`：

```bash
python3 experiments/megamoe/scripts/comm_core_sweep.py \
  --mode baseline --bs-values 2048,4096,8192 \
  --dispatch-metadata-mode scalar \
  --full-reference-dir /path/to/full-megamoe-run \
  --timing-rtol 0.10 --require-timing-match
```

代码复用和任务一致不保证物理时间天然一致。移除 GMM 会改变 HBM/UB/发指令竞争、rank 相位、Combine ready 节奏与通信重叠。
本实现没有插入人为 sleep 来拟合时间。必须先在同机完成对照验收，再把模拟实验的性能结论投入 MegaMoE。

## 独立通信实验

```bash
# 只运行通信模拟。
python3 experiments/megamoe/scripts/comm_core_sweep.py \
  --mode baseline --bs-values 2048,4096,8192

# 原 Combine，仅扫描 Dispatch 的组内/跨组核心配比。
python3 experiments/megamoe/scripts/comm_core_sweep.py \
  --mode dispatch --bs-values 2048,4096,8192 \
  --intra-cores 10,12,16,20 --combine-intra 0
```

普通扫描中，`0` 或不设置表示该阶段原调度；`1..27` 表示 INTRA 核数，CROSS 为 `28-INTRA`。
例如 Dispatch=10 时 AIV1 0..9 负责 INTRA，10..27 负责 CROSS。跨组按本组−1、本组−2、本组−3 处理，组切换不添加全卡屏障或排空流水。
Prefill Dispatch 保留生产槽位策略（通常 6 槽）；通信模拟的 Combine 默认为原 2 槽，可用 `--combine-buffer-count 4` 或 `6` 做独立实验。
默认模式为 baseline，扫描必须显式选 dispatch；仍保留显式 `--mode staged` / `--mode fullgrid`。
候选选择标准为 timer-off **通信 kernel** 的每轮最慢 rank 耗时均值，最优候选在新进程中复测。
这个时间输出为 `COMM-KERNEL`，包含路由准备、Dispatch、Combine 与 Unpermute，不再标为完整算子 E2E。

每个 case 使用新的 torchrun 进程组、SHMEM runtime 和 bootstrap 端口。父进程等上个进程组退出，默认额外等待 2 秒再启动下一轮；`--case-gap-seconds 0` 可关闭，均在设备计时之外。
`--shmem-port-policy fixed` 可显式复用 endpoint；默认 fresh 保留 IP、每轮选择新端口。
`npu-smi info` 仅保存只读快照；16 张卡必须已分配给本作业。

## 数据与计时验收

未设置 `--no-validation` 时，每个通信 case 包含 timer-off 性能采样、普通 timer 批量调用（仅最后一次带 timer）、独立 audit kernel。
audit 记录逐 token 实际任务，详细标量日志开销不进入带宽。

FP8 输入字节和 scale 由源 rank/token/lane 确定，身份位覆盖全部 8192 个 token。
模拟 GMM2 输出是可精确表示的 BF16 整数，按 rank 和专家平铺行的多位编码生成，路由权重为 1/8。
每个阶段检查所有 Dispatch 数据字节、scale、metadata 的 rank/token/TopK、专家计数和完整 Combine 输出。
CPU 独立重建路由及逆向归并，检查不依赖设备 metadata 生成期望值。这是通信数据面验证，不是模型计算精度验证。
完整 MegaMoE 对照沿用原 benchmark 的模型输入与权重。

CPU Wave oracle 检查原生产 Wave 预算、专家切片、逻辑核轮转和源 rank 行布局。
任务缺失、重复、跨 Wave 搬移、错误物理核/peer/token、计数不一致均拒绝报告。
候选保留每 Wave 任务集合，但按候选分核算法验收新的核心归属。

## Dispatch 与 Combine 独立带宽

本节记录当前实现保留的历史 GB/s 公式。它使用平均核时而非实际通信完成跨度，因此只表示归一化指标，不应用来判断物理链路利用率；本轮没有把报表切换为新的时间口径。
每个 rank 分别计算两类指标：

```text
core_time[op, core] = 该核所有 Wave 的 op 通信 slice 时间之和
mean_core_time[op]  = sum(core_time[op, core] for core in 0..27) / 28
GB/s[op]           = total_payload_bytes[op] / mean_core_time_us[op] / 1000
```

无任务核心时间为零，分母仍是全部 28 核。整轮总字节除以整轮平均核心通信时间，不能平均各 Wave 带宽。
先逐 rank 计算，再汇总 16 rank 的均值、最小、最大。Dispatch 与 Combine 的字节数和耗时均独立。

| 阶段 | 每行有效 payload | H=3584 |
|---|---|---:|
| Dispatch GET | `align256(H) + align32(ceil(H/32))`，FP8 token 与 scale | 3712 B |
| Combine PUT | `H × sizeof(BF16)`，原始专家输出 | 7168 B |

| BS | 每阶段行数/卡 | Dispatch 字节/卡 | Combine 字节/卡 |
|---:|---:|---:|---:|
| 2048 | 16384 | 60817408 | 117440512 |
| 4096 | 32768 | 121634816 | 234881024 |
| 8192 | 65536 | 243269632 | 469762048 |

公共 timer 复用完整 MegaMoE 边界，包含本次非空任务的 owner/peer 分配、prefix 恢复、metadata、搬运和完成等待。
Combine 包含成功进入时的 owner 计算和 ready 探测，失败重试不计入；slice 之间计算/等待空档不计入累计通信时间。
未额外添加流水屏障，保留原 timer 已有同步。当前 `TopkWeightsPrefetch=false`，路由权重在后续 Unpermute 中应用。
Dispatch/Combine 带宽时间不包含前置路由准备与后续 Unpermute。
payload 分子不含 metadata、控制标记与本地暂存，不做 RX+TX 翻倍。

```text
[CASE] BS=2048 original schedule COMM-KERNEL rank-max-mean=...us rank-min-mean=...us
[BW] DISPATCH per-card mean=... rank-min/max=.../... GB/s
[BW] COMBINE per-card mean=... rank-min/max=.../... GB/s
[WAVE_CORE] strict per-wave/core validation passed; rank JSON/CSV paths in: .../communication.json
```

前置对齐计算沿用 benchmark，默认临时区约 12 GB/卡，可调 `--alignment-rows/cols/repeats`；对照和模拟必须保持相同配置。
通信 snapshot、CPU oracle、H2D 初始化均在计时之外。

## 输出与资源

输出根目录为 `benchmark-runs/megamoe_comm_core_sweep/<时间_随机号>/`：

- `plan.json`、`npu_smi.txt`：计划和设备快照。
- `summary.json` / `summary.csv`：两类带宽、通信 kernel 耗时、选择结果。
- `bs_*/<case>.log`：命令、SHMEM endpoint、worker Python、错误及退出码。
- `bs_*/<case>/bs_*/timer/communication.json`：schema 2 汇总和每个 rank 的详细 JSON/CSV 路径。
- 详细 JSON：每个 Wave、operation、物理核的行数、字节、累计时间、peer 计数。
- 详细 CSV：audit 实测逐 token 任务身份；时间取普通 timer JSON。
- `rank_*/bs_*/payload_accuracy.json`、`timer_accuracy.json`、`communication_audit_accuracy.json`：完整数据验证。
- `bs_*/full_megamoe_comparison.json`：显式对照的每核时间匹配结果。

默认不保存庞大 raw timer；需要流水图时加 `--keep-raw-timers`，再使用 timer CLI。
普通 timer 为 346,104,064 B/卡，释放后申请 audit 648,093,952 B/卡；`--no-validation` 不申请 audit。
16 rank 同时解析 audit 的原始 D2H 数据约 10.37 GB 主机内存，另需任务对象和 snapshot 内存。
TCPStore 仅交换紧凑报告，详细数据保存在各 rank 文件。
数据与任务检查失败立即停止，时间偏差是否强制退出由 `--require-timing-match` 控制。
CPU 测试不代替 CANN 编译和实际 UBX 的无死锁、数据正确性、时间匹配验证。
