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
带宽分子来自普通 timer 中实际记录的行数，分母为全部 28 个 AIV1 的通信累计时间均值；Dispatch、Combine 分开计算。
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
Combine 搬运实现保持原样。批量读取路由索引所需 `MTE2_S`、slice ready 同步和槽位复用等待仍保留；本实验不代表去除了全部 scalar 依赖。
`plan.json`、rank manifest、详细通信报告、`summary.json` / `summary.csv` 和控制台均记录 `dispatch_metadata_mode`，防止结果混淆。

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

`0` 或不设置表示该阶段原调度；`1..27` 表示 INTRA 核数，CROSS 为 `28-INTRA`。
例如 Dispatch=10 时 AIV1 0..9 负责 INTRA，10..27 负责 CROSS。跨组按本组−1、本组−2、本组−3 处理，组切换不添加全卡屏障或排空流水。
保留生产槽位策略：Prefill Dispatch 通常 6 槽，Prefill Combine 2 槽。
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

每个 rank 分别计算两类带宽：

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
