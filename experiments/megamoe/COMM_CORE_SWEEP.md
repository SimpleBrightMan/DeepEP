# MegaMoE EP16 通信核心配比扫描

本分支在生产 W4A8 MegaMoE 中分别设置 Dispatch 和 Combine 的本组/跨组 AIV1 配比。
不是独立通信微基准：量化、路由、GMM1、SwiGLU、GMM2、Combine 和 Unpermute 均执行，
GMM 与激活计算仍使用全部 28 个物理核心。连续四个 rank 为一组，共四组；本组 INTRA 包含本卡搬运。

## 编译与运行

在已经分配给本作业的单机 16 卡、每卡 28 核机器上运行。先激活匹配的 Python/Torch 环境并 source CANN，
设置 `SHMEM_ROOT`（完整 SDK）及 `LD_LIBRARY_PATH`。构建入口可复用已配置的外部 SHMEM/CATLASS，
也可按仓库依赖锁准备子模块。安装的 wheel 必须来自此分支，不能沿用独立 Dispatch benchmark 的二进制。

```bash
git fetch origin
git switch codex/megamoe-comm-core-sweep
git pull --ff-only
export DEEPEP_BUILD_MEGAMOE=ON
export DEEPEP_MEGAMOE_TIMER=ON
bash experiments/megamoe/scripts/build.sh

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:19091  # 默认复用此 IP，每轮自动选新端口。
export LD_LIBRARY_PATH="${SHMEM_ROOT:-$PWD/third-party/shmem/install/shmem}/lib:$ASCEND_HOME_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 只预览，不访问设备、不创建输出目录。
python3 experiments/megamoe/scripts/comm_core_sweep.py --plan

# 一键扫描默认三组 BS，同时检查全量输出，最终生成 summary.json/summary.csv。
python3 experiments/megamoe/scripts/comm_core_sweep.py
```

固定模型：H=3584、I=1280、E=384、K=8、S=0、EP16，输入 FP8、权重 FP4，
使用仓库原有 rank-first balanced 路由；默认 BS=2048/4096/8192。每种配比单独启动 torchrun，
避免修改运行中的设备计划；每个算子调用仍只启动一个融合 kernel。
`npu-smi info` 只保存只读快照，入口不会杀掉、抢占已有作业；16 张可见卡需由使用者分配。

每个实际启动的 case 都使用新的 SHMEM runtime 和不同的 bootstrap 端口，包括 baseline 和最优复测。
同一轮已使用的端口不会再次分配；默认保留 `DEEPEP_SHMEM_ENDPOINT` 的 IP，环境变量未设置时使用 127.0.0.1。
正常 worker 路径执行流同步、SHMEM barrier/free/finalize，再关闭 TCPStore；父进程等待 torchrun 完整退出后才启动下一轮。
轮间默认额外等待 2 秒，位于算子计时区间之外；`--case-gap-seconds 5` 可增加等待，`0` 可关闭。
`--shmem-port-policy fixed` 可显式恢复固定 endpoint，便于对照；默认 `fresh`。
各 case 日志记录 SHMEM endpoint、worker Python 和 torchrun 返回码。

若错误为 `communication report ... Object of type int64 is not JSON serializable`，
则是计时报告的 NumPy 整数序列化错误，通常发生在 baseline 后第一个带 timer 的候选，
不表示已进入后续核数或 SHMEM 初始化失败。修复在 `timer/protocol.py`：
从原始 timer 解码 metadata 前转换为 Python `int`，使报告可经 TCPStore JSON 交换。
这是 Python 修复；已安装包也需要更新，单纯 `git pull` 不会更新 `site-packages`。

```bash
# 首轮少量点验证；数字代表 INTRA 核数，CROSS=28-INTRA。
python3 experiments/megamoe/scripts/comm_core_sweep.py \
  --bs-values 2048 --intra-cores 7,8,12 --warmup 5 --iters 20

# 分别指定两个阶段的候选；Dispatch 扫描期间先固定 Combine INTRA=12。
python3 experiments/megamoe/scripts/comm_core_sweep.py \
  --dispatch-intra-cores 6,7,8,10,12 \
  --combine-intra-cores 6,7,8,10,12,16 --initial-combine-intra 12

# 全部 1..27 配比：分阶段，默认不会展开 27×27。
python3 experiments/megamoe/scripts/comm_core_sweep.py --intra-cores "$(seq -s, 1 27)"

# 需要研究两者交互时才使用 fullgrid；下例是 4×4，共 16 个候选/BS。
python3 experiments/megamoe/scripts/comm_core_sweep.py --mode fullgrid --intra-cores 6,7,8,12
```

默认候选为 `2,4,6,7,8,10,12,16,20,24,26`，按照以下顺序运行：

1. 不设置配比环境变量，执行原始生产调度作为 baseline，保存全量输出和专家计数。
2. 固定 Combine 的 INTRA=12，扫描 Dispatch 候选，按算子整体耗时选择 Dispatch 配比。
3. 固定上一步的 Dispatch 最优值，扫描 Combine 候选；已经执行的相同配比复用结果。
4. 从两个阶段全部已测候选选择最优配比，使用新进程复测，报告相对 baseline 的总耗时加速比。

默认每个 BS 23 次启动，三个 BS 共 69 次；包括 baseline 和最优组合复测。
分阶段搜索可能错过 Dispatch/Combine 的交互最优点。`fullgrid` 遍历给定候选的笛卡尔积，
也只能选出本次采样中最优的一组，不能消除温度、运行噪声等影响。

## 配比如何生效

```bash
export DEEPEP_MEGAMOE_DISPATCH_INTRA_CORES=8
export DEEPEP_MEGAMOE_COMBINE_INTRA_CORES=7
```

两个变量必须同时设置为 1..27；均不设置则保留原调度。
只影响对应通信阶段：Dispatch 的物理 AIV1 0..7 为 INTRA，8..27 为 CROSS；
Combine 的 0..6 为 INTRA，7..27 为 CROSS。没有减少 launch 的物理核心数量。
每个跨组 cohort 按本组−1、本组−2、本组−3 的 GET/目的组次序处理，
组之间没有全卡屏障，也不在组切换处排空流水。
各组任务按商/余数分配，尾数分配轮转，避免固定核长期承担全部尾行。

限制为 W4A8、EP16、28 个 AICore、四卡一组；不匹配会报错。
`SymmBuffer` 初始化时检查各 rank 配比一致，创建以后不能改变环境变量并复用旧 buffer。
扫描入口为每个候选启动新进程。

保持生产槽位策略：Prefill Dispatch 通常为 6 槽，Prefill Combine 为 2 槽；
此分支只扫描核心配比，没有把 Combine 改成独立 benchmark 的 6 槽。
tiling ABI 增加两个 uint32，224 → 232 字节；设备 timer 新增四类事件，须同步重编 C++ 与安装 Python。

## 如何读结果

每个候选打印完整算子的 timer-off 耗时和四个通信值：

```text
[CASE] BS=4096 D=8/20 C=7/21 E2E rank-max-mean=...us rank-min-mean=...us
[BW] DISPATCH_INTRA per-card mean=... rank-min/max=.../... GB/s
[BW] DISPATCH_CROSS per-card mean=... rank-min/max=.../... GB/s
[BW] COMBINE_INTRA per-card mean=... rank-min/max=.../... GB/s
[BW] COMBINE_CROSS per-card mean=... rank-min/max=.../... GB/s
```

**算子耗时**来自原有 Torch profiler 的 MegaMoE kernel Duration，不传 timer 指针。
使用正式轮次的后半部分；选择指标为“每轮 16 个 rank 中最大 kernel Duration，再对轮次取均值”。
同时保留每轮 rank-min 的均值以及所有 rank/轮次的均值。
这比只用最快 rank 更适合选集体算子的配比；所有计算、通信和 kernel 内同步都包含在耗时内。
权重准备、H2D 输入、CPU 建表、输出比较及 profiler 初始化在区间之外。
默认前置对齐计算沿用现有 benchmark，临时区每卡约 12 GB；可用 `--alignment-rows`、
`--alignment-cols`、`--alignment-repeats` 调小，比较时各候选必须保持相同设置。

**通信带宽**来自另一次单独打点调用，计算真实搬运行数，不按核心比例估算字节：

| 阶段 | 每行单向 payload | H=3584 |
|---|---|---:|
| Dispatch GET | `align256(H) + align32(ceil(H/32))`，FP8 token 加 E8M0 scale | 3712 B |
| Combine PUT | `H × sizeof(BF16)`，已应用路由权重的专家输出 | 7168 B |

对每个 rank、每个 wave、每种 INTRA/CROSS，取这一类全部有效 slice 的
`max(end)-min(start)` 时间包络。该 wave 带宽等于该 wave 的真实 payload 除以这个包络。
终端四个值是：每个 rank 各 wave payload 总和 / 各 wave 包络之和，再汇总 16 个 rank 的均值/最小/最大。

**这不是物理链路忙时带宽**：wave 包络可能包含生产者等待、slice 之间的计算调度间隔、起止偏斜；
不同 wave 的包络也可能重叠。不能把四种带宽相加，不能把包络和当成整算子耗时，
也不能直接与独立通信 benchmark 的连续窗口结果视作相同测量。
不进行 RX+TX 翻倍，metadata/ready 控制流量及本地暂存读写不计入有效 payload。
Dispatch slice 区间包含提交、最终完成等待和 tile-ready 发布；Combine 在生产者 ready 成功后开始，
包含 metadata 预取、搬运和最终完成等待。重建 peer 段表及 cumsum 恢复不在四类通信条内，
但这些开销计入 timer-off 完整 kernel Duration。

设备事件为 `wave_N_AIV_DISPATCH_INTRA_*`、`DISPATCH_CROSS_*`、
`COMBINE_INTRA_*`、`COMBINE_CROSS_*`。每条记录的 `row_count` 是该核实际搬运的行数，
不表示从 `row_start` 开始的一段连续输出行。所有 rank 的所有 wave 行数会与独立 balanced 路由计数校验，
缺失事件、错误 cohort、字节不守恒或缺失 rank 会直接拒绝结果。

## 精度与输出文件

每个候选增加一次不计时调用，在 16 个 rank 上逐元素比较全部 `[BS,H]` 输出，默认要求完全一致，
并逐专家比较累计 token 数。baseline 自身检查有限输出和独立路由计数。
另外比较 timer-on 与 timer-off 输出，避免 timer 诊断掩盖数值异常。
`--comparison-rtol/--comparison-atol` 可显式放宽候选相对 baseline 的比较阈值，报告保留其值。
baseline 是原生产算子的回归参考，不能替代已有 CPU Golden 量化参考验收；完整 Golden 流程见主 README。

输出在 `benchmark-runs/megamoe_comm_core_sweep/<时间_随机号>/`：

- `plan.json`、`npu_smi.txt`：配置和设备快照。
- `summary.json`、`summary.csv`：全部配比、四类带宽、总耗时及每个 BS 的选择结果。
- `bs_*/<候选>.log`：启动命令、配比环境、原生错误及验证日志。失败立即停止，不掩盖为低性能。
- `bs_*/<候选>/bs_*/timer/communication.json`：每 rank、每 wave 的字节数、行数、包络及带宽。
- `bs_*/<候选>/rank_*/bs_*/kernel_samples.json`：timer-off 原始 kernel 样本。
- `.../output_comparison.json`、`timer_accuracy.json`：全量输出/计数及 timer-on/off 验收。
- `bs_*/baseline/rank_*/bs_*/output_reference.npz`：baseline 全量输出，保存一次供全部候选复用。

默认不保留庞大的 raw timer 二进制，仍保留分 wave 的通信 JSON；`--keep-raw-timers` 可以开启，
随后使用 `python -m deep_ep_experimental.megamoe.timer <timer目录> --ranks 0-15` 生成完整流水。
每个候选的 16 份 raw timer 可达数 GB，完整扫描可达数百 GB。此入口不画折线图。

## 验证边界

CPU 测试覆盖分组分核任务守恒、行计数/字节公式、wave 包络口径、core cohort、扫描计划、
最优选择指标和全元素输出比较。通过 CPU 测试或 CANN 编译不代表已在 UBX 上通过精度、死锁和性能验收。
最终以本机运行的全 rank 回归检查、通信行数守恒、timer-on/off 检查和 timer-off 整算子耗时为准。
