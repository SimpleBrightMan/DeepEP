# BS2048/4096/8192 CATLASS GMM-only

分支 `megamoe-gmm-only-bs2048`，基于远端 `megamoe-urma-tailCover` 的 `60642fb`。
只新增独立算子和性能入口，原 URMA/主线核函数不变。

这是 **单卡复现一个逻辑 EP16 rank 的计算量**，无需启动16个进程、初始化通信域或占用其它卡。
使用已分配的空闲卡；`--device` 是当前进程可见的逻辑设备编号。

```bash
git fetch origin
git switch megamoe-gmm-only-bs2048
git pull --ff-only origin megamoe-gmm-only-bs2048
# 已激活 Python 并 source CANN；与原分支相同的 SHMEM/CATLASS 构建环境。
DEEPEP_MEGAMOE_TIMER=OFF bash experiments/megamoe/scripts/build.sh

python3 -m deep_ep_experimental.megamoe.gmm_only_benchmark \
  --bs 2048 --device 0 --logical-rank 0 --cv-rows 256 --gmm1-ahead-mgroups 12 \
  --modes both,gmm1,gmm2 --warmup 10 --iters 20 --peak-tflops 765
```

`--bs` 支持 2048（默认）、4096、8192。改变真实路由 counts、接收行数、M-group、
tile 任务表和缓冲区大小，不是重复执行 BS2048。三种规模依次运行：

```bash
export SHMEM_ROOT="$PWD/third-party/shmem/install/shmem"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for bs in 2048 4096 8192; do
  python3 -m deep_ep_experimental.megamoe.gmm_only_benchmark \
    --bs "$bs" --device 0 --logical-rank 0 \
    --cv-rows 256 --gmm1-ahead-mgroups 12 \
    --modes both,gmm1,gmm2 --warmup 10 --iters 20 --peak-tflops 765 || break
done
```

| BS | 每卡接收行数 | M-group | GMM1逻辑tile（每个gate+up） | GMM2 tile |
| --- | --- | --- | --- | --- |
| 2048 | 16384 | 72 | 360 | 1008 |
| 4096 | 32768 | 144 | 720 | 2016 |
| 8192 | 65536 | 264 | 1320 | 3696 |

本版本 `gmm_only_protocol=2`，升级旧版本需要重新编译安装，不能只复制 Python 入口。
下文具体行数和 FLOPs 示例以 BS2048 为例，其余规模按实际接收行数计算。

`both` 在**一次 kernel 启动**中执行原 host plan 的 GMM1/GMM2 专家交替顺序；
`gmm1`、`gmm2` 过滤同一任务表，分别测两个阶段。
默认三种模式都测，分别打印设备核函数平均/最小/最大耗时、有效 TFLOPS 和 MFU。
独立阶段之和不应冒充 `both` 的时间：上下文切换和缓存热度不同。

每张卡保留：

- BS2048、H3584、I1280、E384、TopK8、EP16、24个本地专家。
- 直接调用仓库 `benchmark.inputs(..., routing="balanced")` 生成16个来源的 counts，再调用**同一个 C++ BuildExpertPlan**。
- 每个 rank 16,384 行，专家各682或683行；每个专家3个 M-group，M=256、256、170/171。
- 28 AIC + 配对28 AIV0；AIV1空闲。AIV0继续执行真实 FP4 权重反量化。
- GMM1逻辑360个tile，每个分别计算gate、up，共720次projection；GMM2 1008个tile。
- 原 CATLASS 模板、M256/N256/K配置、每核tile轮转、专家上下文复用、来源/CV分段Gather/Scatter、本卡结果直写布局、L2策略。

删除的是通信、激活就绪、完成通知等**外部 GM flag**：不运行Pack、Dispatch、SwiGLU、Combine、Reduce，
不等待inputScaleDone/activationDone，不写gmm1Done/gmm2Done，也不做GMM1逐tile发布所需的FIX_S排空。
保留CATLASS内部MTE/MMAD/FIX依赖、AIC/AIV0权重缓冲区信用同步，以及每个专家上下文最终输出drain；删除这些会导致竞争和错误耗时。

两层输入和scale在计时前准备在本地HBM，均是合法、有限、非零的FP8/E8M0值。
GMM2输入独立预置，并不消费GMM1的输出。权重使用原生成与NZ转换流程，GMM1明确选Prefill planar布局。
这样无需精度目标或SwiGLU依赖，但保留矩阵形状与真实访存方式。

计时仅使用 `torch_npu.profiler` 的 `kernel_details.csv` 中精确匹配的设备任务。
不使用Event，不记录内部timer；即使整个wheel打开timer，该独立核也单独关闭timer宏。
连续预热后取所有正式轮次，不插每轮同步，也不只选最快轮次。

有效FLOPs（乘加记2，不把M尾块padding计成有效工作）：

```text
GMM1 = 2 × rows × H × (2I) = 300,647,710,720 FLOPs
GMM2 = 2 × rows × I × H    = 150,323,855,360 FLOPs
both = 450,971,566,080 FLOPs
TFLOPS = FLOPs / (mean_us × 10^-6) / 10^12
MFU = TFLOPS / peak_tflops × 100%
```

分母是整个独立MIX核函数设备耗时，包含GMM内部数据搬运、权重反量化和上下文收尾，
因此是GMM计算路径的有效MFU，并非只量Cube MMAD指令忙碌周期。`--peak-tflops` 应填该设备及当前数据类型的单卡峰值；
若物理卡核心更多，本测试依旧只启28核，默认765不自动按物理核心数缩放。
所有输入反复复用，结果代表预热后的此工作集，不代表无缓存冷启动或融合算子端到端性能。

只查看计算量、不访问NPU：

```bash
python3 -m deep_ep_experimental.megamoe.gmm_only_benchmark --plan-only
```

结果在 `benchmark-runs/gmm_only/<时间戳>/summary.json`，每种模式保留原始profiler目录。
原始报告保留每轮耗时和代入公式。这个入口不做数值精度判定。
