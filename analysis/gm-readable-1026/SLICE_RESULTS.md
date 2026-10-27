# 可读性GM分支合入Slice通知：柜5单轮结果

2026-10-27。已合入`codex/megamoe-gm-readable-1026`，设备源码冻结提交`8118a7647f43db5e1fe86116a1d473514bcb6134`；main未替换。

后续进展：本文记录的是最初单轮性能筛选；之后已补齐EP8/EP32全量独立Golden，见[Golden结果与main提升依据](golden/RESULTS.md)。不要将后补精度的控制面改动混入本性能窗口。

## 改了什么

迁入`codex/megamoe-slice-combine-1026`的`44a39b2`和`5f3e191`机制，不恢复已删除的CV代码。

- AIC在本核实际拥有输出的每个expert slice末，将FULL事件排在该slice全部FIX后并发布issued；不等本核整个Wave完成才通知。
- AIV1按同一个slice/owner顺序等待真实FULL，发布该slice的ready并归还EMPTY。没有输出tile的核不发布/消费不存在的事件；仍使用单槽credit，不逐slice增加AIC Scalar排空。
- 进入GMM2 Scalar完成等待前，先安排消费同FIX队列中更早的GMM1，防止堵住GMM1 EMPTY归还；lag条件复用可读性分支已有`UseGmm2WaveLag`，不重复配置512阈值。
- 不改变Wave预算、配对、GM地址、发送owner、原数学与最终跨卡同步；不是token-prefix提前归约。

重点阅读：`mega_moe_wave_a8w4.hpp`的`NextGmm2CompletionSlice`、`TryReceiveGmm2Completion`、`ProcessAiv1PriorityStages`、`RunGmm2CombineForWaveRange`，以及`common/gmm2_aiv_completion.hpp`。

## 新测试流程

按用户要求：提前部署柜5八节点，身份/空闲/锁检查后只启动一轮候选B，连续七BS，不默认ABBA、不再额外启动独立probe。设备/SHMEM初始化、正常预热和门禁保留。约定见`experiments/megamoe/TESTING.md`；项目状态MD及离线网站已同步。

EP8在CPU7，EP32在CPU1/2/3/4；两组不共用NPU，部分时间同时运行，EP8设备通信在单节点内。没有整柜网络独占保证。H7168/I3072/E384/top-k6、balanced、seed20260922、无共享专家、capacity4096、完整SHMEM/MTE通信；warm10/iters20取后10，timer-off完整kernel Duration，逐轮rank-min均值，另保留rank-max。

下表的A来自柜5已保存记录，**与本轮B不同窗口**：EP8是整理后未加slice的`0f5ed46`两次均值（同CPU7）；EP32是旧main `7de059f`的完整control-a一轮（同有序CPU1/2/3/4，原生代码`095b4c7`）。EP32同时跨过可读性整理，因此不能把全部差异唯一归因slice通知。B均为一次启动；正值是相对历史A降耗，仅作筛选，不是新增同窗因果收益。

| BS/卡 | EP8历史A μs | EP8合并B μs | 历史对比 | EP32历史A μs | EP32合并B μs | 历史对比 |
|---:|---:|---:|---:|---:|---:|---:|
|32|596.23|596.61|−0.06%|205.90|200.79|+2.48%|
|96|615.02|616.74|−0.28%|235.85|216.84|+8.06%|
|256|659.72|655.68|+0.61%|376.92|343.75|+8.80%|
|512|745.45|740.44|+0.67%|677.41|647.94|+4.35%|
|1024|1082.01|1070.94|+1.02%|1168.89|1155.68|+1.13%|
|2048|2067.97|2045.19|+1.10%|2162.49|2163.88|−0.06%|
|4096|4034.91|4018.44|+0.41%|4183.24|4175.70|+0.18%|

本轮未见明显新回退；EP32 BS96/256/512的较大改善与此前独立slice实验方向相容。其余小差异不当成稳定收益。没有本窗口DS对照，不能称全域最优；如需正式合并main验收或确认约1%的差异，再按需补同窗对照。

## 验证和本地证据

- 合并后本地11项协议/调度/UB检查通过；远端针对性CPU检查75项与CANN B080原生构建通过。快速流程4项CPU测试通过，验证单轮计划、状态标记和缺测/启动数失败路径。
- 本轮所有40rank均正常退出；560份原始CSV和14份rank0/32组完整被动流水已下载。报告逐rank核对库身份、20轮kernel及后10轮聚合、输出有限、timer输出一致、trace覆盖与周期合法。
- `compare_history.py`额外逐rank核对历史A与本轮B的完整输入/输出指纹一致；这些不等于独立Golden，按用户要求未跑Golden，也未覆盖偏斜路由或EP64/128。
- 独立timer-on pass中，每次测量调用均打点；EP8增幅0.05%～1.59%，EP32约−0.11%～0.65%。负值按测量波动解读，不表示打点加速；主表全部timer-off。
- Python解释器退出时仍有既有`SymmBuffer.__del__`的NoneType警告；旧基线也存在，保留日志，不在本轮顺手改变退出清理。
- 冻结包SHA256：`32dbbb93e008f6fa327298760fb87cfe4fcca1fc00de6475d0c7d0b7591097ac`；设备库SHA256：`5d42b7e1d66a6de8e15be8a51ca260ee5149646e6a1ea1faaec2cdb713acdb24`。

本地根目录：`C:\Project\Infra\ops-transformer-hs\outputs\gm-readable-1026`。

- EP8：`slice-ep8-r05-7-a`。
- EP32：`slice-ep32-r05-1234-a`。
- 两目录的`summary.json`为单轮原始审计，`historical-comparison.json`为明确标注跨窗口的历史比较；`manifest.json`保存收集证据哈希。每rank的`profiler`是timer-off CSV，`timer_profiler`是timer-on CSV，不能混用。

EP8 BS512流水（Insight单行反斜杠路径）：

```text
C:\Project\Infra\ops-transformer-hs\outputs\gm-readable-1026\slice-ep8-r05-7-a\node0\runs\bs512-combined-a\rank_0\timer\timer_trace.json
```

EP32 BS512流水：

```text
C:\Project\Infra\ops-transformer-hs\outputs\gm-readable-1026\slice-ep32-r05-1234-a\node0\runs\bs512-combined-a\rank_0\timer\timer_trace.json
```
