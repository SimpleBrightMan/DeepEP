# MegaMoE EP16 原调度通信基线与核心配比实验

本分支默认运行生产 W4A8 MegaMoE 的原调度，分别测试 BS=2048、4096、8192，
导出每个 rank、每个 Wave、每个 AIV1 核的 Dispatch / Combine 任务和带宽。
量化、路由、GMM1、SwiGLU、GMM2、Combine、Unpermute 均实际执行，全部 28 个物理核心保持启动。
固定模型 H=3584、I=1280、E=384、K=8、S=0、EP16，FP8 输入、FP4 权重，沿用仓库 rank-first balanced 路由。
连续四个 rank 为一组，本组 INTRA 包含本卡搬运。

## 编译与运行

在已经分配给本作业的单机 16 卡、每卡 28 核机器上运行。激活匹配的 Python/Torch 环境并 source CANN，
设置完整 SHMEM SDK 的 `SHMEM_ROOT`。构建可复用已配置的外部 SHMEM/CATLASS，或按依赖锁准备子模块。
此次修改同时包含设备 timer 和 Python 报告，须重新构建安装本分支，不能复用旧二进制。

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

# 预览计划，不访问设备、不创建输出目录。
python3 experiments/megamoe/scripts/comm_core_sweep.py --plan

# 默认只测原生产调度；三个 BS 各启动一次完整测试。
python3 experiments/megamoe/scripts/comm_core_sweep.py \
  --mode baseline --bs-values 2048,4096,8192
```

`--mode baseline` 会清除继承的 Dispatch / Combine 核心配比变量。
每个 case 依次执行 timer-off 整算子性能测量、普通通信 timer、独立详细任务 audit、全量输出/计数回归。
普通 timer 阶段按原有对齐计算与批量调用逻辑运行，仅最后一次 kernel 带 timer；
audit 额外执行一次 kernel，记录实际 Wave / core / peer / token 分配。
带宽只读取普通 timer 调用的时间；audit 的时间不进入带宽或 E2E 指标。
各调用使用相同输入与路由，普通 timer 与 audit 输出分别和 timer-off 输出比较。

`npu-smi info` 只保存只读快照；16 张可见卡需由使用者分配。
每个 case 使用新的 torchrun 进程组、SHMEM runtime 和 bootstrap 端口。
正常退出完成流同步、SHMEM barrier/free/finalize、TCPStore 关闭；父进程等待进程退出后才运行下一轮。
默认轮间额外等待 2 秒，位于算子计时之外；`--case-gap-seconds 0` 可关闭，
`--shmem-port-policy fixed` 可显式复用 endpoint。默认 `fresh` 保留配置的 IP、为每轮选择新端口。

## 在基线上实验

Dispatch 与 Combine 的开关已解耦。`0` 或不设置表示该阶段保持原生产调度，
`1..27` 表示该阶段的 INTRA 核数，CROSS 核数为 `28-INTRA`。
默认 Dispatch 扫描保持原 Combine，因此候选与 baseline 的差异集中在 Dispatch 任务分工。

```bash
# 原 Combine，扫描 Dispatch 的本组核数；每个 BS 包含 baseline、4 个候选、1 次最佳复测。
python3 experiments/megamoe/scripts/comm_core_sweep.py \
  --mode dispatch --bs-values 2048,4096,8192 \
  --intra-cores 10,12,16,20 --combine-intra 0

# 显式研究固定新 Combine 调度；此时相对原 baseline 同时改变了两种通信阶段。
python3 experiments/megamoe/scripts/comm_core_sweep.py \
  --mode dispatch --bs-values 2048 --intra-cores 7,8,12 --combine-intra 12

# 原 Combine，遍历全部 Dispatch 配比。
python3 experiments/megamoe/scripts/comm_core_sweep.py \
  --mode dispatch --intra-cores "$(seq -s, 1 27)"

# 保留显式的两阶段与全网格实验入口。
python3 experiments/megamoe/scripts/comm_core_sweep.py --mode staged \
  --dispatch-intra-cores 6,7,8,10,12 \
  --combine-intra-cores 6,7,8,10,12,16 --initial-combine-intra 12
python3 experiments/megamoe/scripts/comm_core_sweep.py --mode fullgrid --intra-cores 6,7,8,12
```

默认模式已改为 `baseline`。执行扫描必须显式加 `--mode dispatch`，否则只运行三个 BS 的原调度。
默认 Dispatch 候选仍为 `2,4,6,7,8,10,12,16,20,24,26`；选择标准为 timer-off 整算子耗时。
候选逐元素检查全量输出和专家计数，最优候选使用新进程复测。
`staged` 可能错过两阶段交互；`fullgrid` 也只给出本次采样组合中的最优值。

例如单独配置 Dispatch：

```bash
export DEEPEP_MEGAMOE_DISPATCH_INTRA_CORES=10
unset DEEPEP_MEGAMOE_COMBINE_INTRA_CORES
```

物理 AIV1 0..9 负责 INTRA、10..27 负责 CROSS；Combine 继续使用原来的连续行归属与 Wave 轮转。
跨组 cohort 按本组−1、本组−2、本组−3 的顺序处理，组切换处不添加全卡屏障或排空流水。
保留生产槽位策略：Prefill Dispatch 通常 6 槽，Prefill Combine 2 槽。
`SymmBuffer` 初始化检查各 rank 配置一致；创建后不能通过改环境变量复用旧 buffer 改计划。

## Wave 与核心分配验收

原调度直接执行生产通信路径，并记录实际任务。CPU 独立模型根据 balanced 路由、生产 Wave 预算、
专家切片、原始逻辑核轮转和源 rank 行布局构建预期记录，与设备 audit 对照。
验收同时覆盖 Dispatch 和 Combine 的 Wave、物理核心、专家、目标行和 peer 分配。
候选必须满足原始每个 Wave 的工作集，并满足指定分核算法的核心归属。
缺失、重复、跨 Wave 搬移、错误核心/peer 或计数不一致会拒绝报告。

原调度的 CPU 核分配 oracle 是主线算法的独立复现；运行的是本分支编译后的生产路径。
这些校验保证通信任务相符，不表示与单独编译的 main 拥有相同指令布局或耗时。
详细任务日志只在 audit 调用开启，带宽调用不写这份日志。
当前固定 timer 协议的普通 buffer 为 346,092,304 B/卡，audit buffer 为 648,082,192 B/卡；
普通 timer 释放后才申请 audit。audit 的各 rank 原始 buffer 在本地 D2H 并解析，
16 个 rank 同时解析时原始数据约 10.37 GB 主机内存，另需任务身份对象的内存。
TCPStore 仅交换已校验的紧凑汇总与文件路径，逐 token 明细写到各 rank 的独立文件。

## Dispatch 与 Combine 独立带宽

对每个 rank，分别对 Dispatch 和 Combine 计算用户指定的口径：

```text
core_time[op, core] = 该核所有 Wave、所有 op 通信 slice 的耗时之和
mean_core_time[op]  = sum(core_time[op, core] for core in 0..27) / 28
bandwidth[op]      = total_payload_bytes[op] / mean_core_time[op]
GB/s              = total_payload_bytes[op] / mean_core_time_us[op] / 1000
```

28 是启动的物理 AIV1 核数；某个 Wave 内无任务的核心仍计入分母，通信时间为零。
逐 Wave 同样使用全部 28 个核计算，并导出每核行数、字节数、通信累计时间及 peer 明细。
整轮带宽用整轮总字节除以整轮平均核心通信时间，不能对各 Wave 的带宽直接求平均。
终端和 CSV 的值是先逐 rank 计算，再汇总 16 个 rank 的均值、最小、最大。
Dispatch 与 Combine 各有自己的字节数与耗时，二者不合并，也不相加。

| 阶段 | 每行有效 payload | H=3584 |
|---|---|---:|
| Dispatch GET | `align256(H) + align32(ceil(H/32))`，FP8 token 加 E8M0 scale | 3712 B |
| Combine PUT | `H × sizeof(BF16)`，应用路由权重后的专家输出 | 7168 B |

均衡路由下每 rank 的预期整轮任务如下，包含本卡、本组、跨组任务：

| BS | 每阶段行数 | Dispatch payload B | Combine payload B |
|---:|---:|---:|---:|
| 2048 | 16384 | 60817408 | 117440512 |
| 4096 | 32768 | 121634816 | 234881024 |
| 8192 | 65536 | 243269632 | 469762048 |

此指标使用有效 payload，未把 metadata、ready 控制信息、路由权重加载和本地暂存读写计入分子，
也不进行 RX+TX 翻倍。分母是通信 slice 的实测累计时间，未把两条通信 slice 之间的 GMM/激活空档算入。
因此它与物理链路峰值、最长核心耗时、整算子时长或旧版 Wave 时间包络带宽是不同口径。
新的公共通信区间包含本次有数据任务的 owner/peer 分配、prefix 恢复、metadata 处理、搬运与完成等待。
Combine 从成功尝试进入时计时，包含该次 owner 计算与成功 ready 探测；此前未就绪的失败探测不计入。
公共区间不额外增加流水屏障，保留原 timer 路径已有的同步；timer-off 时不执行新增的 cycle 读取。
零行任务不贡献通信时间，相关调度成本与阶段间空档仍体现在 timer-off 整算子耗时中。

输出示例：

```text
[CASE] BS=2048 original schedule E2E rank-max-mean=...us rank-min-mean=...us
[BW] DISPATCH per-card mean=... rank-min/max=.../... GB/s
[BW] COMBINE per-card mean=... rank-min/max=.../... GB/s
[WAVE_CORE] strict per-wave/core validation passed; rank JSON/CSV paths in: .../communication.json
```

算子耗时取原 Torch profiler 的 MegaMoE kernel Duration，不传 timer 指针，使用正式轮次后半段样本。
选择指标为每轮 16 rank 中最大的 kernel Duration，再对轮次求均值；同时保留 rank-min 及全 rank 均值。
前置对齐计算默认沿用 benchmark，临时区每卡约 12 GB，可调整 `--alignment-rows/cols/repeats`，
但所有对照 case 必须使用相同值。

## 输出与验证边界

输出在 `benchmark-runs/megamoe_comm_core_sweep/<时间_随机号>/`：

- `plan.json`、`npu_smi.txt`：计划与只读设备快照。
- `summary.json`、`summary.csv`：每个 case 的 Dispatch / Combine 独立带宽、整算子耗时、选择结果。
- `bs_*/<case>.log`：命令、环境、SHMEM endpoint、worker Python、错误及退出码。
- `bs_*/<case>/bs_*/timer/communication.json`：schema 2 汇总；`operations` 保存两阶段带宽，`ranks[].validation.strict_wave_core` 必须为 true，`ranks[].detail_files` 给出详细 JSON/CSV 路径。
- 详细 JSON：每个 rank、Wave、operation、core 的行数、payload、累计时间与 peer 计数。
- 详细 CSV：audit 采集到的逐 token 任务身份，包括 Wave、operation、core、expert、目标行、源 rank/token 与 TopK 槽位；计时取 JSON 中的普通 timer 汇总。
- `rank_*/bs_*/kernel_samples.json`：timer-off 原始 kernel 样本。
- `output_comparison.json`、`timer_accuracy.json`、`communication_audit_accuracy.json`：全量输出、专家计数及 timer / audit 对输出影响的检查。
- `baseline/rank_*/bs_*/output_reference.npz`：用于候选全量回归的原调度输出。

默认不保留庞大的 raw timer；逐 Wave / core 详细 JSON/CSV 仍会保存。
需要完整流水图时加 `--keep-raw-timers`，然后使用 timer CLI；本入口不绘制扫描折线图。
所有检查失败立即停止。默认候选输出要求完全一致，`--comparison-rtol/atol` 可显式放宽并记录阈值。
CPU 测试、CANN 编译、任务守恒与输出回归各验证不同问题；实际 UBX 精度、死锁和性能仍以设备运行结果为准。
