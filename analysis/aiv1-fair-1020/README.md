# AIV1 常驻 UB 与 DS 式交替推进：实验结果

结论：**实现已生效，但 EP8/BS512 两项独立 ABBA 都没有稳定收益，不合入 master。** 不把这版称为已完整迁移 DS 后端。

## 版本、协议与改动

父版本是 `93e5805`（设备代码 `2adf28c`，A 预取候选），不是前一轮无收益的 GMM1 descriptor 分支。柜05 CPU7，Ascend950DT，CANN B080 9.2.0。

1. `codex/megamoe-aiv1-resident-1020`：隔离 SwiGLU、Dispatch、Combine 三套 UB 工作区，保留原有严格优先级；移除 Dispatch prefix 的 GM 备份/恢复，Combine 视图只初始化一次。为放下三套缓冲，激活内部工作块改为64行，GMM tile/ready计数/数值算法不变。BS512 每专家64行，激活分段数不增加。
2. `codex/megamoe-aiv1-fair-1020`：在1之上，每轮尝试一个就绪激活，再交替推进 Dispatch/Combine；Combine未就绪则退回Dispatch。任务游标、输入就绪协议和通信 owner 不变。

这里“常驻”是三种任务的**工作缓冲互不覆盖**，不是让全部 token、权重或中间结果留在 UB。输入仍按任务读入，输出仍写 GM。

EP8/BS512 的 UB 分配见父实验 `analysis/aiv1-resident-1020/README.md`，共191680字节，约187.19KiB；host/device使用同一个容量规划函数。实验版明确拒绝共享专家和超容量配置，没有静默回退；A8W8路径未修改。

正式性能为完整通信融合 kernel Duration；FP8/FP4，H7168/I3072/E384/TopK6，无共享专家，balanced，seed20260922；warmup10，采20取后10，逐轮rank-min后求均值，另存rank-max。每项ABBA四次独立启动，timer-off。分核PMU和流水另采，不能拿诊断耗时替代正式性能。

## EP8/BS512 独立 ABBA

| 单因素实验 | 当前对照 μs | 候选 μs | 降耗 |
| --- | ---: | ---: | ---: |
| 原版 → 三套 UB 常驻 | 806.580 | 806.157 | 0.052% |
| 常驻 → 交替推进 | 804.826 | 804.668 | 0.020% |

两项控制窗口不同，不能相加收益。第一项候选两次为804.699/807.615μs；均值变化小于启动波动。第二项rank-max均值806.401→807.622μs，也不支持“稳定变快”。

## 流水证明调度已改变，而非未加载候选

仅rank0打点，保留32组AIC/AIV0/AIV1共96轨道，原生事件不因计时新增完成等待。三版分别实测开销1.66%、1.21%、1.74%。下列时间是最后一次timer-on的观察，不是ABBA均值。

以group0、相对该次最早AIC计时起点：

| 观察点 | 原版 μs | 常驻 μs | 常驻+交替 μs |
| --- | ---: | ---: | ---: |
| 第一个Combine开始 | 208.546 | 209.613 | 138.002 |
| 最后Dispatch结束 | 207.627 | 208.841 | 243.122 |

交替版确实早服务Combine，但也推迟后续Dispatch。不能仅凭首个输出提前70μs，就声称端到端节省70μs。

32组阶段区间累计的组均值（各阶段可重叠，不能相加）：

| 阶段 | 原版 μs | 常驻 μs | 常驻+交替 μs |
| --- | ---: | ---: | ---: |
| GMM1 ISSUE | 450.896 | 450.734 | 451.264 |
| GMM2 ISSUE | 204.138 | 206.031 | 206.115 |
| Dispatch | 124.690 | 126.185 | 131.531 |
| SwiGLU | 154.271 | 159.062 | 161.850 |
| Combine | 61.515 | 60.523 | 48.276 |

ISSUE是提交窗口，不是纯MMAD或设备完成时间。原版与交替版GMM1推进没有相应缩短，Combine局部改善没有转化为端到端收益。

## `msprof --task-block=on` 分核证据

PipeUtilization，device0的block0/15/16/31，每角色40条原生记录（末10次调用×4个block），与官方V6BlockPmu表逐项对应。下表为`事件cycle/总cycle×该block Duration`得到的活跃等价时间，不是不能重叠的耗时账本，也不是等待时长。

| AIV1 指标 | 原版 μs | 常驻 μs | 常驻+交替 μs |
| --- | ---: | ---: | ---: |
| Vector（0x501） | 58.45 | 58.64 | 58.43 |
| Scalar（0x1） | 405.29 | 403.75 | 413.53 |
| MTE2（0x202） | 166.30 | 160.88 | 164.53 |
| MTE3（0x203） | 156.22 | 160.90 | 147.36 |

交替版MTE3活跃等价时间减少约8.87μs，同时Scalar增加约8.23μs；不能据此相减计算收益。AIC Cube（0x301）529.95→529.70μs，AIV0 Vector404.73→404.76μs，基本不变。当前结果不支持“只要降低三任务切换成本就能追齐DS”。

## 为什么常驻没有带来预想收益

- 原代码不是每次恢复完整任务数据：EP8的`ReloadDispatchCumsumRange`仅恢复该专家所需prefix，首专家约32B，其余约64B；48专家合计约3040B/核。Tensor视图重新绑定本身也不等于重搬数据。常驻没有删掉主要的SwiGLU/通信数据流。
- 交替调度改善了早期Combine服务，但占用了原来用于提前发送后续输入的机会；本次完整关键路径未缩短。
- **常驻不等于跨任务异步。** 本次仍保留`FinishDispatchOwnedRows→DrainDispatchCopyBuffers`、`RunTokenCombineSlice→DrainCombineRowBuffers`、激活输出的`MTE3_V/MTE3_S`完成边界。共享事件0..5在返回前释放，不能因为UB隔开就直接删掉等待。
- DS还通过独立任务事件/持久游标让流水携带在途工作，SwiGLU每次最多32行yield，GMM1经FIX直达消费者UB；本次没有迁移这些协议或CV通路。因此本实验不能证明AIV1完全无关，只能否定“单独常驻/交替就有明显收益”的假设。

后续优先定位**暴露**的完成等待，再考虑独立事件域/跨任务保留在途状态；先画事件归属及wait-for依赖，不能直接删drain。GMM1输出路径仍需GM双credit与CV双credit对照；历史单槽CV无净收益，不重复单槽方案。每项都要求端到端收益，而非仅计数器/局部条带变短。

## 其他 shape：只作单次筛选与分段回归

| BS/卡 | 原版 μs | 常驻 μs | 常驻+交替 μs |
| --- | ---: | ---: | ---: |
| 96 | 655.059 | 656.879 | 655.489 |
| 1024 | 1146.946 | 1150.149 | 1147.721 |
| 4096 | 4130.604 | 4128.907 | 4126.498 |

全部8卡有限值及对原版逐位输出对照通过，覆盖小M、128行和256行激活内部分段。未运行独立Golden；单次筛选不能证明这些shape有稳定收益。未验证其他EP、偏斜路由、共享专家，不作为生产合入验收。

CPU执行生产UB规划函数及真实调度循环，覆盖容量、隔离、延迟就绪/fallback、replay和空任务；这不替代NPU同步验证。初次编译的host/device限定错误及第二次限定错误均保留；成功二进制来自第三次独立目录，非旧SO回退。

## 本地证据与源码阅读位置

本地根目录：`C:/Project/Infra/ops-transformer-hs/outputs/aiv1-resident-1020/final`。

- `summary.json`：正式两项ABBA、单shape筛选、分核原始counter与阶段/时序汇总。
- `control-trace/timer_trace.json`、`resident-trace/timer_trace.json`、`fair-trace/timer_trace.json`：完整被动流水。按group数字排序，等待不画实心条；`timer_all_events.json`保留原始完整事件供核对。
- 各自目录内`setup/rank_0/bs_512/profiler/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv`及`setup/rank_0/bs_512/profiler/timer_on/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv`：分别是off/on，不能混用；8个rank的原目录层级均保留。
- `pmu_*_PipeUtilization/profiler/PROF_*/device_0`：原生分核PMU、采集配置和官方`metric_summary.db`。
- `manifest.json`：1253文件的大小/SHA，包含184份原始kernel CSV，本地全部核验。
- `identity.json`、`resident-only/identity.json`、`compile-command.json`：源码逐文件SHA、实际SO SHA和编译命令；`harness/`保存实际测量/差分脚本。

主要修改位置（均在`experiments/megamoe/csrc/kernels/mega_moe`）：

1. `common/mega_moe_aiv1_ub_layout.hpp`：容量与三区域地址规划；`mega_moe.cpp`启动前校验。
2. `mega_moe_arch35.hpp::DispatchBuffInit`：常驻视图绑定，激活私有工作块64行。
3. `stage/mega_moe_gmm1_activation.hpp::Gmm1Aiv1EpilogueTileA8W4`：内部分段与原tile完成通知。
4. `stage/mega_moe_gmm2_combine.hpp::RunTokenCombineSlice`：复用常驻配置，保留原生槽位排空。
5. `mega_moe_wave_a8w4.hpp::ProcessAiv1PriorityStages`：交替推进；`DispatchNextExpertSlice`移除prefix恢复。

实际SO SHA：原版`aaf25491…`，常驻`d25a5d6d…`，交替`b01905f8…`。网页只新增实验记录，不替换冻结七shape三方法曲线。
