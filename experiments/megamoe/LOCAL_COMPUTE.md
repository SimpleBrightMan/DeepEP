# 零通信本地计算验证

当前 Wave 路径已将 GMM2 的 FIX 完成等待移到配对 AIV1，协议与验证边界见
[GMM2_AIV1_WAIT.md](GMM2_AIV1_WAIT.md)。下文为历史本地计算实验记录，
其中逐 slice 析构/排空的描述不代表当前持久 Wave 路径；这些历史精度和性能结果也不替代本次移植验收。

计算优化基线 `codex/megamoe-compute-opt` 基于 `codex/megamoe-local-compute`，
后者基于 `codex/megamoe-compute-replay`。保持计算调度和数学运算，优化 GMM2 完成边界。
原分支需要SHMEM初始化和一次完整forward作为priming；本分支新增local-only Plan，
在host准备专家连续FP8输入、E8M0 scales及每AIC一份block-major专家计数。
不初始化SHMEM/HCCL，不调用torch.distributed，不建立跨卡数据通路。
现有扩展仍链接SHMEM库（构建依赖不等于运行时建立通信），保留原版API。

服务器使用`/root/miniconda3/envs/zyx`，从hkq克隆，CANN入口：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate zyx
source /home/pkg/b070/cann-9.2.0/set_env.sh
bash experiments/megamoe/scripts/build.sh
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
# 单设备先跑，负载仍是EP8的每卡48专家、BS4096、每专家512token
python experiments/megamoe/scripts/local_compute.py --devices 0
# 8张卡独立运行相同负载；不是8卡集体通信测量
python experiments/megamoe/scripts/local_compute.py
```

默认3次重复，结果在`benchmark-runs/local_compute_<时间>/`。
检查内部GMM2结果finite、nonzero及重复一致性，不读取无效的最终MoE输出。
这不是完整数值Golden校验，不宣称精度通过。
脚本默认是功能smoke；默认不打timer，不输出kernel时延。性能采样使用下文的 `--profile`。
`host_wall_seconds`只用于排查卡住，包含host开销且可能受其他任务影响。
即使没有通信，仍会争用计算和HBM带宽，不能把共跑结果作为性能优化依据。
新环境/构建只作用于zyx及独立源码目录，不改hkq或其他用户任务。

本地地址兼容区由普通NPU tensor持有，不注册对称堆。local-only Plan在准备完成前拒绝forward，
避免错误地进入通信路径；数据/计数/容量/设备必须匹配。输入计数冻结，测试与真实路由输入不同。

## 2026-10-10 实机验证

- Ascend950DT_9582，CANN B070，Torch 2.12.0+cpu，torch_npu 2.12.0.post2。
- 原始基线`c4790e6`；原生扩展及设备库已成功构建，计算实现验证提交`7da0228`。
- BS4096，H7168，I3072，TopK6，无共享专家；EP8形状每卡48专家，每专家512 token。
- 单卡0和八卡0–7各执行3轮；全部输出finite/nonzero，重复结果逐元素相同。
- GMM2输出先填NaN，确保未覆盖输出不能误判通过；不等价于完整数值Golden。
- 每卡Plan workspace为7,393,402,880字节；这不含权重、输入、本地地址兼容区和框架缓存。
- 服务器八卡日志及汇总：`/home/zyx/smoke-local-compute-ep8/`。
- 默认不启用timer；本次只验证功能，没有发布可比较的kernel性能结论。

## GMM2 优化及复现实验（2026-10-10）

### 运行方法与测量口径

完成上面的环境加载和构建后，在仓库根目录执行：

```bash
# 八卡分别跑 EP8 形状的本地计算，不建立通信。正式性能比较需独占设备。
python experiments/megamoe/scripts/local_compute.py --bs 4096 --profile
# 单卡：性能阶段 timer 关闭，结束后另跑一次诊断流水，不混入性能样本。
python experiments/megamoe/scripts/local_compute.py --devices 0 --bs 4096 --profile --trace
# 与已保存的同设备、同 BS、同 seed 基线校验完整 GMM2 输出
python experiments/megamoe/scripts/local_compute.py --bs 4096 --profile \
  --reference-dir /home/zyx/opt-baseline-ep8
```

默认 warmup 20；三轮各采 50 次 MegaMoE kernel Duration，统计每轮后 25 次。
每轮先排入独立的小计算以减少 host 提交不及时，其耗时不纳入 MegaMoE 样本。
这不是完整 MoE E2E：没有真实 Dispatch、Combine 和跨卡同步。
`device_N.json` 保存所有 kernel 样本、统计值、输出 SHA256 和已安装设备库的 SHA256。
`source_commit` 是 runner 源码版本，不保证等于已安装 wheel 版本；切换 wheel 对比时应看
`kernel_binary_sha256`。`summary.json` 汇总八卡；`npu_before/after.txt` 保存设备快照。
设备快照不能证明整个测量窗口独占，因此 `performance_valid` 保守保留为 false。

`--trace` 额外生成 `device_N_timer.json`，不保存 wait 条，等待仍体现为原时间轴空白。
GMM2 标为 `AIC_GMM2_ISSUE`：表示发射区间而非 FIX 完成区间；不能将其累计值冒充计算总时长。
新增 timer 槽位需要重新构建 wheel，旧 raw timer buffer 不应套用新协议。

### 实测结果

H7168、I3072、384 专家、TopK6、无共享专家，BS4096，EP8 形状每卡48专家、每专家512 token。
同一台 Ascend950DT_9582 / CANN B070，timer 关闭，数据来自 profiler kernel Duration。

| 实验 | kernel 时间（μs） | 处理 |
| --- | ---: | --- |
| 原始计算基线，单卡三轮均值约 | 4196 | 对照 |
| 移除 TokenCombine 逐 tile FIX 排空，单卡 | 4113 | 保留 |
| 在上项基础上扩大 A 的 K 搬运窗口到512 | 4135 | 负收益，撤回 |
| 在 FIX 版本基础上改成四槽 L1 B | 4129 | 负收益，撤回 |
| 基线回测，八卡×三轮均值 | 4196.252 | 对照 |
| 最终版，八卡×三轮均值 | 4114.466 | 降低81.786μs，约1.949% |

最终版还去除了 slice-ready 发布前重复的 FIX 等待；未单独证明这一处有可分辨收益。
两项缓冲试验保留在 git 历史供查阅，但不在分支最终代码中。它们只证明这两种具体实现无收益，
不证明短 K 供数、缓冲或同步已经最优。

服务器结果：`/home/zyx/opt-baseline-ep8/`、`/home/zyx/opt-final-ep8/`；
负收益试验：`/home/zyx/opt-a512-1/`、`/home/zyx/opt-b4-1/`。
最终设备代码提交 `543525e`，wheel 为 `dist/build-jp70l016/ascend_deepep-0.1.0.dev5-cp312-cp312-linux_aarch64.whl`。
最终设备库 SHA256：`ed071caeaf6023529fa7113ff8a7f22fcc1179e0332200bcd83078b75488750b`。
基线使用未改计算实现的 local-only wheel；八卡回测时 runner 已更新，因此不能仅用其 source_commit 识别内核。

### 为什么可以删除等待

代码入口：`csrc/kernels/mega_moe/stage/mega_moe_gmm2_combine.hpp` 的
`Gmm2AicMmadA8W4`、`NotifyTokenCombineSliceReady`，以及 `mega_moe_arch35.hpp` 的调用点。

- TokenCombine 在 slice 完成后才发布就绪计数，原先每 tile 的 Scalar FIX 排空不是消费者需要的粒度，
  却阻止了后继发射与 FIX 输出重叠。现在只删除这个分支的逐 tile 排空。
- 逐 tile 消费路径 `NotifyCombineTileReady` 仍保留 FIX 等待；L0C unit flag 保护不变。
- `RunGmm2A8W4` 返回前，局部 `CatlassGmm2` 析构经 pipeline 的 `Cube::End` 完成 FIX_S，
  然后调用方才执行 `NotifyTokenCombineSliceReady` 发布计数。该 helper 不能移到析构之前使用。
- 未删除 activation 就绪依赖、资源回收排空或输出发布顺序。缓冲布局、MMAD tile 和 K 分块最终均保持原样。

### 验证边界与后续方向

BS4096 八卡全部与基线完整 GMM2 输出 hash 一致；单卡 BS96、BS4104（每专家513 token，含尾块）也一致。
BS4104 单独 timer pass 输出一致、协议解析成功，JSON 约3.27 MB，带 group_id/core_type/sub_id。
这些是回归等价验证，不是独立数值 Golden；真实通信模式尚未做端到端回归。
主机单测78项通过。该轮没有测量 timer-on 相对 timer-off 的开销。

基线测试前曾观察到他人的 NPU 进程，测试后快照为空；结果重复稳定，但不能保证整个窗口独占。
应在预约独占设备后复测再作为正式性能结论；未停止或重置他人任务。
DS 未在本次同机同输入条件下重新测试，因此不能把历史可见区间差285μs直接当成可回收E2E收益。

下一步优先研究跨 tile/slice 保持计算 pipeline 状态，减少反复初始化、资源排空和 flag 握手；
先统计其次数与时间，再做保持依赖关系的单变量实验。DS 参考实现的缓冲更持久，
而单纯在当前生命周期中增加槽位还会增加初始化和切换成本。
之后再检查短 K 的 L1→L0 等待、MMAD 连续发射及编译器生成代码。
目前证据支持“过早完成等待贡献约2%”，不支持“剩余差距全是 FIX 或全是矩阵乘指令本身”。

## 生命周期定位实验：codex/megamoe-lifecycle-study

本轮按“既有 trace → 必要的批次边界 → 依据证据决定是否复用状态”的顺序进行。
未拆独立 GMM2，未修改缓冲数量、tile、调度或原生同步。目标是定位，而非把未知收益作为优化交付。

### 如何复现

新协议追加生命周期事件，必须重新构建设备库和 Python 包；旧 raw buffer 与新协议不兼容。
在已激活 zyx、已 source CANN、已设置 SHMEM 库搜索路径的仓库根目录运行：

```bash
bash experiments/megamoe/scripts/build.sh
# 性能阶段关闭timer，结束后单独保存包含坐标的原始事件及精简流水。
python experiments/megamoe/scripts/local_compute.py --devices 0 --bs 4096 --profile --trace
# 单独的开销实验：每个预热/测量调用都打开timer，不用最后一次trace稀释开销。
python experiments/megamoe/scripts/local_compute.py --devices 0 --bs 4096 --profile --profile-timer
# 将上面输出目录替换为实际目录；对DS的rank_N_npu_time.json也可运行。
python experiments/megamoe/scripts/analyze_gmm_progress.py \
  benchmark-runs/<本次目录>/device_0_events.json --output benchmark-runs/<本次目录>/progress.json
```

`device_N_events.json` 保存已有的 expert/row/wave 坐标；`device_N_timer.json` 仍是无WAIT条的精简概览。
分析器分别统计每核 GMM 区间、相邻区间间隔、start-to-start 节奏和原生生命周期观测。
坐标缺失时输出 unknown，绝不将序号猜成专家编号。DS 的 `_part0/_part1` 保持分段，
片段计数与任务计数分开；片段间隔不拼成持续计算区间。不同核的累计时间不能相加当E2E。

### 第一轮事实与测量边界

- 未加新打点前，我方本地计算 timer-off 三轮：4114.268、4115.250、4114.655μs。
- DS同机完整MegaMoE三轮：4032.162、4013.463、4007.378μs，指标为每轮最慢rank取值后的均值。
- DS采用本地参考提交 `253b474`，原始上游 `8491bbb`；依赖按锁定的 DeepJIT `65f5139`、
  magic_enum `b233b96` 准备，安装在zyx，不改hkq。DS源码目录 `/home/zyx/DeepGEMM-lifecycle-study`。
- 形状相同，DS采用balanced routing；我方是每专家512行的本地预置输入，数值seed/通信口径不同。
  因此这些数字只能用于定位，不能作为同输入、同执行范围的正式加速比。
- DS首轮timer-on/off差异在噪声范围内，不能宣称负开销。服务器快照未见占用不能证明整段独占。

新增Begin/End版（`ad859bc`）完整GMM2输出与旧基线相同，BS96和BS4104尾块回归也通过。
该版timer-off三轮4117.929、4118.010、4117.994μs；每次调用启用timer后4152.261、4152.060、4152.383μs，
均值开销0.832%。打点关闭时比未加观测版约多3.3μs，诊断代码本身也可能改变代码生成，不能忽略。

group 0的一次观测如下；这些是区间统计，不是纯硬件忙碌时间：

| 原生边界 | 次数 | 平均μs | 累计μs |
| --- | ---: | ---: | ---: |
| GMM1 Begin | 36 | 0.121 | 4.351 |
| GMM1 End | 36 | 0.107 | 3.859 |
| GMM2 Begin | 48 | 0.103 | 4.927 |
| GMM2 End | 48 | 4.967 | 238.404 |

32核的GMM2 End累计在236.453–242.638μs之间。说明成本集中在GMM2收尾，而非普遍的Begin初始化。
同任务坐标内GMM2间隔均值0.112μs，换专家后约6.061μs，GMM2→GMM1约5.765μs。
GMM1区间36条累计2660.115μs；DS72条累计2508.397μs，但我方是原生完成边界、DS是ISSUE，不能据此断言MMAD差多少。
我方GMM2 ISSUE累计1151.848μs，DS84任务（140片段）累计1225.536μs；我方大量收尾在ISSUE区间外。
因此旧账本的“GMM2条带累计差”不能沿用到新的ISSUE版本。

原始证据位于服务器 `/home/zyx/lifecycle-existing-bs4096/`、`lifecycle-boundary-off/`、
`lifecycle-boundary-on/`、`lifecycle-boundary-bs96/`、`lifecycle-boundary-bs4104/`；
DS三轮在 `/home/zyx/ds-lifecycle-ep8/`、`ds-lifecycle-ep8-r2/`、`ds-lifecycle-ep8-r3/`。

### 安全性与实验三的约束

新增 `BeginBoundary/EndBoundary` 只读取时钟、写独占timer区，不调用PIPE_ALL或增加flag等待，
也不改当前tile的活动状态。`AIC_GMM2_FIX_DRAIN` 嵌套在End内部，两者不能相加。
事件后缀分别是每阶段生命周期调用序号，Begin/End按调用顺序对应，并保留expert/row/wave坐标。
Begin记录入口slice坐标，End/FIX_DRAIN可能沿用最后一个tile坐标；以同核、同阶段、调用序号配对，
不要要求两者row_count相同，也不要把End坐标当成整个批次范围。

`Cube::End` 必须在恢复MM布局之前等待FIX；`NotifyTokenCombineSliceReady` 必须在输出完成后发布。
只复用C++对象不等于能够删除该等待。真实通信路径也必须正确，不能利用compute-only没有消费者而删除同步，
再把这个诊断版的提速当成可迁移收益。
若要跨同类任务延续credit，需要同时处理旧slice的延迟完成发布、L0C复用和AIV0缓冲索引，
并在GMM1/GMM2资源切换前保留最终排空。未满足这些约束前不删除End。

### FIX细分观测与当前结论

最终设备代码 `0d1554e`，安装wheel `dist/build-kiyur6a9/ascend_deepep-0.1.0.dev5-cp312-cp312-linux_aarch64.whl`，
设备库SHA256为 `e7ba6ff7cd4c4f3a61222334e261f3572720f62485c503cb613e77ec161cc0c2`。
准备性能采样时发现其他进程占用八卡，主动中断了自己的采样；`/home/zyx/lifecycle-fix-off/`
是不完整且共跑的结果，不得引用为性能数据，后续timer-on命令没有执行。
占用快照在 `/home/zyx/lifecycle-fix-shared-device.txt`。未停止他人任务。

按已获授权仅执行共跑功能/诊断，结果在 `/home/zyx/lifecycle-fix-functional/`。
BS4096完整GMM2输出与基线hash一致，timer解析及32核边界分析成功。
group 0的48次GMM2 End共240.949μs，其中FIX_DRAIN为233.747μs；32核FIX_DRAIN/End比例均值约96.88%。
这支持“End中的原生FIX完成等待占主导”，但共跑时长不能作为优化收益估计。
FIX_S等待还可能覆盖尚未完成的前序MMAD，不等价于纯FIX执行或HBM写出耗时。
最终增加FIX_DRAIN后尚未验收timer开销，不能套用前版0.832%的结论。
CPU定向测试87项通过，协议校验通过；最终协议22类动态事件，追加观测没有新增同步调用。

实验三尚未实施：简单复用Begin状态不针对主要成本；直接删除End违反输出就绪和资源复用协议。
后续最小有意义的候选是同一GMM2段内允许后续计算提前推进，同时延迟发布旧slice就绪，
在阶段切换前完成排空。需要成对维护AIC/AIV0的credit、输出完成及AIV1消费者顺序。
必须在设备可用时完成输出回归、真实通信模式回归以及timer-off A/B/A测试后才保留。
本轮找到可证实的主要等待位置，但未完成因果消除实验、未交付新的性能收益、未宣称追平DS。

## 实验三：GMM2 wave 内生命周期复用

分支 `codex/megamoe-gmm2-wave-lifetime`，设备实现提交 `14a7a3e`，基于上面的生命周期诊断版。
这次只改变非量化 routed GMM2 的生命周期及其完成发布粒度，不改 wave 大小、任务顺序、
tile、K 分块、缓冲槽位、权重布局或数学运算。不是仅在 compute-only 下删除同步。

### 改动与依赖

- `mega_moe_wave_a8w4.hpp::RunGmm2CombineForWaveRange` 持有一个 GMM2 上下文，连续处理同一 wave 的专家 slice。
- `stage/mega_moe_gmm2_combine.hpp` 沿用相同 tile scheduler，向持久 `CatlassGmm2` 更新地址/形状。
  第一次实际有任务时 Begin，之后保留 AIC/AIV0 的 credit、L1/L0/scale 游标。
- `catlass_gmm/mega_moe_grouped_matmul.hpp` 的原生 L0C unit flag 继续保护输出复用。
  在 wave 尾部 End，仍先等待 FIX_S，再恢复 MM layout、回收事件，才允许 GMM1 复用这些资源。
- `mega_moe_arch35.hpp` 在持久上下文分支不提前发布 slice 就绪。
  wave End 后重放轻量分核计算，仅为实际有任务的 AIC 发布一次；不建立随专家数增长的 pending 数组。
  空专家/空核不通知，多个 tile 仍只计一个生产核。共享专家、量化路径保持局部对象生命周期。

代价是本 wave 的 Combine 最早只能在 wave 收尾后消费，而不是逐专家 slice 提前消费。
这需要真实通信模式回归及性能验证；本地计算提速不能直接宣称为完整 MegaMoE 收益。

### 同卡 A/B/A 结果

2026-10-10，B070 / zyx / Ascend950DT_9582，EP8 形状、BS4096、H7168、I3072、E384、TopK6、S0，
本地预置输入、每专家512行，固定 device 1/seed 20261011。
0号卡有其他用户进程，因此不运行整机 EP8 性能；device 1 测量快照未见其他进程。
没有资源预约锁，不能保证整个采样窗口独占，仍建议独占环境复测。
所有性能值都是 timer-off profiler kernel Duration，20次预热、三轮各50次、每轮取后25次均值。

| 同卡版本 | 三轮均值（μs） | 三轮平均（μs） |
| --- | --- | ---: |
| A：生命周期诊断基线 | 4117.942、4118.024、4118.576 | 4118.181 |
| B：GMM2 wave 生命周期复用 | 4026.579、4026.675、4026.775 | 4026.676 |
| A：回装同一基线 wheel | 4117.848、4117.771、4118.110 | 4117.909 |

以两次 A 共六轮平均4118.045μs为参照，B减少 **91.369μs / 2.219%**。
BS4096完整输出hash均为 `fc7018250c83458e8bb0ee60ea213fcee7a7d13561efc7ba8b8dbf9ade034444`。
基线设备库SHA256 `e7ba6ff7cd4c4f3a61222334e261f3572720f62485c503cb613e77ec161cc0c2`；
新库 `389aad5887ab778fa50a7b459f81e56689cefc567f2c25cc0b9cf60377fc9446`，
wheel为服务器 `dist/build-9yhoysf4/ascend_deepep-0.1.0.dev5-cp312-cp312-linux_aarch64.whl`。
runner source_commit不是安装库版本，回装A时需按设备库SHA辨认。

同一新库每次调用打开timer后为4047.082、4046.201、4046.598μs，平均4046.627μs，
相对 B timer-off 增加19.951μs / **0.495%**。未增加测量用同步；最终单独trace不混入off性能。

### 因果证据及不能相加的时间

同一rank/group 0的单独trace：

| 项目 | A | B |
| --- | ---: | ---: |
| GMM2逻辑任务数 | 84 | 84 |
| GMM2 Begin/End次数 | 48 | 16 |
| GMM2 End累计（μs） | 241.420 | 78.457 |
| 其中FIX_DRAIN累计（μs） | 233.853 | 75.419 |
| GMM2换专家间隙均值（μs） | 6.122 | 0.989 |
| GMM2 ISSUE累计（μs） | 1152.057 | 1212.914 |
| GMM1原生边界累计（μs） | 2656.779 | 2654.170 |

End减少与off性能改善同时出现，支持“逐slice排空限制重叠”的判断。
但ISSUE增加约61μs，说明部分原来位于边界的等待/工作转移到连续发射内；
不能把减少的163μs End直接当作E2E收益，也不能把嵌套FIX_DRAIN再加一次。
GMM1基本未变，与本次仅修改GMM2生命周期的范围一致。

原始数据在服务器 `/home/zyx/wave-lifetime-baseline/`、`wave-lifetime-v1/`、
`wave-lifetime-baseline-after/`、`wave-lifetime-v1-on/`，其中前两个含 `progress.json` 和原始timer事件。
BS8、BS96、BS4104分别先运行新库，再回装基线用 `--reference-dir` 对照，完整输出hash一致；
数据位于 `wave-lifetime-v1-bs{8,96,4104}/` 和 `wave-lifetime-baseline-bs{8,96,4104}/`。
这些覆盖很小slice、旋转分核及513行/专家跨wave尾块，但仍属于输出等价回归，不是独立Golden。
87项CPU定向测试与timer协议校验通过，B070设备编译通过。完整通信及共享/量化模式未验收。
随后在空闲device 1–7同时运行BS4096，每卡三个重复调用，七卡完整输出均与此前各卡基线hash一致；
记录在 `/home/zyx/wave-lifetime-v1-seven-devices/`。这是七个独立本地计算进程的功能回归，
不是EP7，也不是完整EP8通信测试。未使用被其他用户占用的device 0。
服务器zyx最终保留新库 `build-9yhoysf4`；没有修改hkq或停止他人进程。

### 已测优化账本与下一步

1. 已保留的逐tile FIX等待删除：此前八卡本地计算4196.252→4114.466μs，约81.786μs / 1.949%。
   原理是按消费者所需粒度保留slice完成边界，不逐tile强制Scalar排空。
2. 本轮GMM2 wave复用：在诊断基线之上约91.369μs / 2.219%。两轮测量范围及诊断代码不同，
   不直接相加百分比或宣称严格总加速比。
3. A窗口K512：此前比FIX-only慢约21μs；B四槽：此前慢约16μs，均已撤回。
   不能因DS有更多缓冲就认为在本后端一定有效；其资源生命周期、输出依赖也不同。

补测同卡原始未优化wheel（`dist/build-nd4hz8_d/`，设备库SHA256
`6f2a3169bc886b711353492eb2b63aab42d3e635233531dfff37d558ef94f959`），
三轮4195.380、4196.423、4196.103μs，平均4195.969μs，完整输出也与上述固定输入基线一致。
因此本次同卡直接测得的组合效果是 **4195.969→4026.676μs，减少169.292μs / 4.035%**，
不是将两次实验的百分比相加；原始版数据位于 `/home/zyx/wave-lifetime-original/`。

当前约4027μs尚不构成“已追平DS”的证据：DS同机4007–4032μs是完整通信及不同输入数值口径。
接下来先做同输入完整通信对照，确认延迟发布没有抵消计算收益；若有损失，优先研究按完成事件
异步发布slice，而非继续扩大wave。计算侧下一重点是GMM1逐逻辑tile的FIX_S→GM ready发布，
与DS的FIX→AIV1原生完成事件、L0C→UB消费路径对比。应成对修改生产者/消费者，
不能直接删除GMM1的FIX_S，否则SwiGLU可能读取未完成的数据。

## GMM1 异步完成通知：codex/megamoe-gmm1-async-completion

基于 `9eda05f`（上述 GMM2 wave 版），设备实现 `f0eb5c6`。
本实验保持 GM 输出布局和 SwiGLU 数学实现，不同时迁移 DS 的 L0C→UB 输出路径。
仅非量化、非 TopkWeightsPrefetch 的 routed A8W4 Wave 启用；共享 GMM1 和其他路径保持原协议。

### 生产、消费及资源回收

阅读顺序：`common/gmm1_async_completion.hpp` →
`stage/mega_moe_gmm1_activation.hpp::Gmm1AicMmadA8W4` →
`mega_moe_wave_a8w4.hpp::TryRunReadyActivation` / `ProcessMoeExpertStages`。
这些路径均相对 `csrc/kernels/mega_moe/`。

1. AIV1 初始化两个 empty credit。AIC 在 FIX 队列上等待对应 empty，然后提交 gate/up 的计算和写出。
2. AIC 把 full 通知排在两个投影的 FIX 输出之后，同时通过原 GM 序号发布 **issued**，不再逐 tile 等 FIX_S。
   issued 只保证已提交；不能据此直接读输出。
3. AIV1 先探测 issued，成功后在 **MTE2** 队列等待该槽的 full，随后排入 GM→UB 读取和 SwiGLU。
   FIX 真正完成输出后 full 才释放，消费者不会读取未完成数据。
4. SwiGLU 沿用原读入、计算、量化、写出依赖，完成后通过 MTE3 归还 empty。
   两槽 credit 保护的是完成事件复用；输出 GM 仍按专家/tile 独立寻址，不是新加两槽 GM 环形缓存。
5. AIC/AIV0 在同一 GMM1 wave 内延续缓冲游标及 credit，不逐专家 slice 重启。
   wave End 仍排空 FIX，再恢复 MM layout，才让 GMM2 复用 L1/L0 等资源；不跨阶段冒险保留状态。
6. routed 阶段最后消费两槽最终 empty，包括未使用槽，避免事件遗留到下一次 kernel 调用。

完成 full 使用 AIV1 flag 12/13，empty 使用 14/15；生产者通知/等待用 AIV1 selector `16|flag`，
消费者等待使用本地低4位，与 AIV0 权重 flag 6/7、8/9 分离。
保留原 L0C unit flag；empty 不替代 MMAD/FIX 的内部资源依赖。
由于 AIV1 还负责 Dispatch，必须先确认 issued 再排入消费等待，不能预先等尚未提交任务的 FIX。

### 同卡 A/B/A 性能

2026-10-10，环境、输入、seed、测量选样与上一节相同，device 1、EP8 形状、BS4096。
计时为零通信本地计算 kernel Duration，20 次预热，三轮各 50 次取后 25 次；timer 关闭。

| 版本 | 三轮均值（μs） | 平均（μs） |
| --- | --- | ---: |
| A：已有 GMM2 wave 版 | 4027.154、4026.821、4027.035 | 4027.004 |
| B：增加 GMM1 异步完成 + wave 复用 | 3923.770、3923.602、3923.318 | 3923.563 |
| A：回装同一旧 wheel | 4026.794、4026.964、4027.234 | 4026.997 |

两次 A 共六轮平均 4027.000μs，B 减少 **103.437μs / 2.569%**。
这是通知协议和必要的 wave 生命周期延长的组合收益，未独立归因到某一条指令。
相对上一节同卡原始未优化库 4195.969μs，总减少约 **272.405μs / 6.492%**；
原始值是较早的同卡补测，不是本轮重新回装原始库的 A/B/A。

timer-on 每次预热/测量调用均打点，三轮 3933.178、3933.585、3933.434μs，平均 3933.399μs；
相对 B 增加 **9.836μs / 0.251%**，满足本 case 约1%的目标。
性能 off 与单独 trace 分开。新事件追加为 `AIC_GMM1_ISSUE`，没有把旧完成区间悄悄改义。
GMM1 ISSUE 结束不是 FIX 完成；SwiGLU 区间可能包含 MTE2 上的 FIX 依赖，不能解读为纯向量忙碌时间。
没有为了时间戳增加 Scalar 完成等待。

group 0 诊断中 GMM1 Begin/End 从36次降到16次；新 GMM1 ISSUE 累计2481.630μs，
wave End 累计77.739μs，GMM2 ISSUE累计1210.599μs。
旧 GMM1 是完成区间、新的是发射区间，不直接相减为加速量；上述数据也不跨核累加为E2E。

### 回归、版本和复现

BS8、96、4096、4104（513行/专家，尾块）均完整 GMM2 输出 hash 与原基线一致，
每 case 三次调用，并检查单独 timer pass 输出不变。属于输出等价回归，不是独立数学 Golden。
BS4096 输出仍为 `fc7018250c83458e8bb0ee60ea213fcee7a7d13561efc7ba8b8dbf9ade034444`。
新设备库 SHA256：`30a117c2fc9728d070b64c69b04b54ff98ea944c126f41b0bfae34d39ad5e73f`，
wheel：`dist/build-k9sovwtj/ascend_deepep-0.1.0.dev5-cp312-cp312-linux_aarch64.whl`。
A 的 wheel、SHA 与上一节 GMM2 wave 版相同。

服务器证据：`/home/zyx/gmm1-async-baseline/`、`gmm1-async-baseline-after/`、
`gmm1-async-v1/`、`gmm1-async-v1-on/`、`gmm1-async-v1-bs{8,96,4104}/`。
新 trace JSON 同时兼容 group/core/sub_id 分组；重新构建后再解析新23类动态事件协议。
device 1–7 各三次 BS4096 调用，完整输出均与此前各卡原基线一致，记录在
`/home/zyx/gmm1-async-v1-seven-devices/`；七卡独立本地计算，不是 EP7/EP8 通信测试。
zyx 最终恢复并保留新库 `build-k9sovwtj`，未改变 hkq，也未停止或重置他人任务。
CPU 定向测试89项通过，包含实际完成通道头文件的编译/执行，覆盖0/1/2/3/36/113任务的
credit复用及尾部全回收；该模型不替代NPU的真实内存可见性和时序验证。timer C++/Python协议一致性通过。
服务器 Linux/zyx 全套测试 **226项通过**，日志 `/home/zyx/gmm1-async-pytest.log`。
Windows全套曾因缺少bash和Linux打包平台检查失败10项；未修改这些无关测试，改在目标Linux环境完成验收。

```bash
# 已激活 zyx、source CANN，位于本分支仓库根目录
bash experiments/megamoe/scripts/build.sh
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
python experiments/megamoe/scripts/local_compute.py --devices 1 --bs 4096 --profile --trace
# 单独测量 timer 开销
python experiments/megamoe/scripts/local_compute.py --devices 1 --bs 4096 --profile --profile-timer
```

0号卡有他人进程，未用于测试；device 1 采样前后没有其他进程，但没有资源预约锁。
目前未验收完整 EP8 通信、共享/量化路径、不均衡或空专家路由的设备行为，不作为生产合并结论。
不能将本地计算3924μs与 DS 的完整通信4007–4032μs直接比较、宣称追平。
下一步优先在独占机器上对照完整通信正确性和 E2E，尤其检查 AIV1 Dispatch/SwiGLU/Combine 的互相影响；
随后再单独研究 L0C→UB 输出，不能把本轮收益解释成已消除 GM 中转成本。
