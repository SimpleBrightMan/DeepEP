# DS特性迁移后的统一CV/GM候选

2026-10-20，分支`codex/megamoe-cv-adaptive-1020`，设备提交`dd5b2d7`，父版为CVⅠ`f48a49f`；性能对照为异步父版`26e1051`，不是宣称相对master取得同样收益。

结论：一个SO、一个资源选择规则，保留小M的CV收益，消除前两版大BS的明显回退。BS512同窗口ABBA降低1.13%；尚未追平布局优化DS，偏斜路由仍有风险，不合入master。

## 改了什么，为什么

已迁移DS的双128行CV stage、32行VF推进、跨任务异步credit与常驻通信UB。FIX结果直接进入AIV1 UB，绕过GMM1 BF16结果的GM写回/读取；保持原scale=64、BF16舍入、激活/量化数学操作、后GMM2路由加权、expert-slice通信和Wave调度。不是复制整个DS后端。

前三项实验见`codex/megamoe-ds-cv-stage-1020`下`analysis/ds-cv-stage-1020/README.md`。它们证明全CV并非普遍有利：当逻辑tile为256行，两槽各128行只能容纳一个完整tile，而旧GM两个credit可保护两个完整tile；更小的激活yield还会重复插入较大的通信slice。更少GM流量无法自动弥补这些反压。

本版在CVⅠ上加一个Host/Device共用选择函数`UseGmm1CvForLaunch`：

```
仅EP8 && paired权重可用 && BS × TopK ≤ 每rank专家数 × 128 → CV
否则 → GM
```

当前E384、TopK6、EP8，每rank48专家，故BS≤1024选CV，BS2048/4096选GM。规则来自stage容量，不是逐shape挑成绩。它使用均衡期望M，不是实际最大专家M；偏斜路由性能没有最优保证。其他EP保留GM，本轮未原生验证。

布局与事件选择保持一致：CV使用144KiB激活区、2pull/4push，GM保留64KiB激活区、6pull/2push。Combine metadata按384行分批循环，不截断长slice。GM工作区仍保留，避免改变workspace/tiling ABI。未增加环境开关或第二套用户接口。

## 同一SO的七shape结果

柜05 CPU7 / Ascend950DT / CANN B080 9.2.0 / 驱动25.6.rc1.b010；EP8，H7168/I3072/E384/TopK6/shared0，FP8/FP4，balanced，seed20260922。完整通信融合kernel Duration，timer-off，warm10，采20取后10，逐轮rank-min求均值。不是主机端到端，也不是无通信compute replay。逐轮rank-max及全部rank样本同时保留。

|BS/卡|路径|同窗口异步父版 μs|统一候选 μs|降耗|
|---|---|---:|---:|---:|
|32|CV|623.256|618.034|0.84%|
|96|CV|657.121|649.156|1.21%|
|256|CV|725.280|710.626|2.02%|
|512|CV|808.308|799.172|1.13%|
|1024|CV|1150.983|1146.082|0.43%|
|2048|GM|2135.428|2136.666|-0.06%|
|4096|GM|4127.283|4126.810|0.01%|

BS512按A/B/B/A，各两次独立启动：父版809.068、807.547；候选798.386、799.958μs。除512外每方案仅一次启动，均为筛选，不自动构成稳定收益或统计显著性证明。BS2048/4096视为基本持平，不宣称0.01%的优化。**没有用旧CVⅠ795.806μs替代本版实际成绩。**

布局优化DS本轮没有重测；离线网站保留其冻结参考。历史同机BS512约700.09μs，说明本次十余μs改善仍不足以填平差距，但不同窗口不能作为精确差额归因。

## 正确性范围、路由和timer

- 七shape全部8卡有限值/逐位差分通过；BS512对冻结基线输出，其他shape对同窗口父版输出。不是独立Golden，未做共享专家、A8W8、其他EP的本轮验收。
- random/skewed BS96对已冻结父版输出均8卡逐位差分通过。候选分别650.176、911.431μs；这两项没有同窗口性能control，只作输出与执行回归。前一窗口父版skewed约869μs，提示不能推广此期望M选择规则为所有路由的最佳策略。
- 完整被动timer仅rank0、全部32组AIC/AIV0/AIV1，共96轨，不额外添加排空。独立off/on实测开销**1.2894%**；等待/SYNC/DRAIN在派生图留白，原始事件仍保留。性能表取正式timer-off，不混用timer-on成绩。
- 未新增本版task-block PMU。前三项的BS512 PMU可用于理解迁移机制，不能冒充本版或大M的计数器。

## 代码阅读入口

路径均相对仓库根，搜索函数名即可定位，避免后续注释移动导致行号失效：

|模块|文件 / 入口|重点|
|---|---|---|
|共享资源规则|`experiments/megamoe/csrc/kernels/mega_moe/common/mega_moe_aiv1_ub_layout.hpp` / `UseGmm1CvForLaunch`|CV/GM判定、UB精确预算|
|Host配置|`experiments/megamoe/csrc/forward.cpp`|pull槽数必须与内核特化一致|
|内核选择|`experiments/megamoe/csrc/kernels/mega_moe/mega_moe.cpp`|UseCv与FuseGateUp分离；禁止未配对权重误入CV|
|资源常驻|`experiments/megamoe/csrc/kernels/mega_moe/mega_moe_arch35.hpp` / `DispatchBuffInit`|CV与GM使用各自布局和事件槽|
|AIC输出|`experiments/megamoe/csrc/kernels/mega_moe/catlass_gmm/block_mmad_a8w4.hpp`|提交、真实FIX full、EMPTY credit，128行stage|
|CV复制|`experiments/megamoe/csrc/kernels/mega_moe/catlass_gmm/copy_scaled_cv_output.hpp`|scale、BF16、L0C步长与子核映射不变|
|AIV1调度|`experiments/megamoe/csrc/kernels/mega_moe/mega_moe_wave_a8w4.hpp` / `TryRunCvActivation`|持久frame、32行推进、真实完成后ready与释放|
|输入消费|`experiments/megamoe/csrc/kernels/mega_moe/epilogue/block_epilogue_activation_mx_quant.hpp`|CV输入与量化scratch互不覆盖|
|flag协议|`experiments/megamoe/csrc/kernels/mega_moe/common/gmm1_async_completion.hpp`|V等FIX full；MTE3归还EMPTY；issued不等于完成|

## 验证与证据

26项CPU测试、设备kernel和host扩展构建通过；实际载入kernel SHA256：

`db9151c3b70b35083875f06b3beeee32f124f3a3e2da107239b5dd1865595c90`

host_C SHA256：`7330e171674917f7fcfed5d33d4d6e2f00bbf5ae2fcb97d86a0dc75c81066bf5`。

本地根目录`C:/Project/Infra/ops-transformer-hs/outputs/cv-adaptive-1020/final/`保存1350文件、160份kernel_details.csv，全部大小/SHA核验；`summary.json`含每次实际启动与有效配置，`manifest.json`是清单，`identity.json`记录源码/依赖/两SO。68个原生源码、15个Python文件及两SO与本分支逐字节核验。

- `adaptive-trace/timer_trace.json`：本版完整96轨派生流水。
- `adaptive-trace/timer_all_events.json`：包括等待的原始事件。
- 配对CSV在同一`adaptive-trace/`下的off/on profiling目录；不要拿正式timer-off表和timer-on图的绝对耗时混算。
- `native-{control,adaptive}/`保存两侧实际设备kernel与host扩展。
- `controller.log/cpu-tests.log/compile.log/campaign.log`、`native-build/compile_commands.json`、每rank配置和资源快照保留，可离线核对。

`prepare.py/build.sh/campaign.sh/collect.py/check_identity.py`是本次私有环境入口，不含凭据。解包校验可复用CVⅡ分支的`verify_archive.py`。只在空闲卡运行，本任务锁不保证其他用户不会抢占，不终止他人任务。

## 反思与下一步

此次可确认的是：移除GM结果中转并协调CV容量/通信资源后，小M能取得有限收益；不能证明GM流量是剩余约100μs差距的唯一来源。较大改动必须同时考虑有效缓冲深度、VF量子、通信slice与MMAD/FIX重叠，而非机械复制DS槽数。下一步应围绕这组协议做能区分原因的消融，先解决偏斜场景与独立正确性，再考虑推广。
