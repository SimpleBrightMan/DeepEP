# 纯计算版本的历史 profiler 对比

2026-10-10，仅读取已有结果，没有重新运行NPU程序。文件实际名为 `kernel_details.csv`。

## 结论与范围

原始纯计算4195.969μs → 最佳纯计算3923.563μs，减少272.405μs / 6.492%。
但这些CSV没有Cube/MTE利用率或带宽计数器，不能据此判断当前compute-bound、memory-bound或达到峰值算力的比例。
3924μs版与前一GMM2 wave版的A/B/A验收见 [LOCAL_COMPUTE.md](LOCAL_COMPUTE.md)；
这里为保留三版统一口径，分别读取各自三轮历史样本，不混入回装A后的三轮。
这些数字不是完整EP8通信E2E，也不是旧main与新main的五BS测试结果。

**后续补充：** 刚完成的完整EP8测试确实导出了MAC/MTE/FIX、指令cache miss与Cube utilization，
已从30份rank0 `kernel_details.csv` 提取，见 [完整EP8硬件指标](EP8_PROFILER_METRICS.md)。
本页“没有硬件指标”仅限历史纯计算三版，不能概括后来的Level1采样。

## 采样核对

- Ascend950DT_9582 / CANN B070 / zyx，device 1，EP8形状，BS4096。
- H7168、I3072、384专家、TopK6、无共享专家；每卡48专家、每专家512行。
- seed20261011，本地预置相同输入；没有Dispatch、Combine、跨卡通信。
- 每版三轮，每轮CSV恰好50条MegaMoE调用；选择后25条，共75个样本。
- 先排入的Exp不是MegaMoE样本，不纳入下表。duration均值与原 `device_1.json` 逐轮核对相同。
- 性能采样timer-off，单独timer调用没有混入；原始CSV未修改。
- 三版完整GMM2输出SHA一致：`fc7018250c83458e8bb0ee60ea213fcee7a7d13561efc7ba8b8dbf9ade034444`。
- 设备没有预约锁，仅有采样前后快照，不能证明整个历史窗口绝对独占。

## CSV的实际变化

| 指标 | 原始纯计算 | GMM2 wave版 | GMM1异步版 |
| --- | ---: | ---: | ---: |
| Duration均值（μs） | 4195.969 | 4027.004 | 3923.563 |
| Duration中位数（μs） | 4195.658 | 4026.941 | 3923.446 |
| Duration最小（μs） | 4191.085 | 4024.835 | 3921.939 |
| Duration最大（μs） | 4202.723 | 4030.733 | 3927.212 |
| Wait Time均值（μs） | 0.068720 | 0.072453 | 0.066987 |
| Block Num | 32 | 32 | 32 |
| 选中样本数 | 75 | 75 | 75 |
| Cube/MTE/FIX利用率 | 未采集 | 未采集 | 未采集 |
| HBM/L2带宽、cache命中率 | 未采集 | 未采集 | 未采集 |

每轮Duration均值（μs）：

| 版本 | round0 | round1 | round2 |
| --- | ---: | ---: | ---: |
| 原始 | 4195.380 | 4196.423 | 4196.103 |
| GMM2 wave | 4027.154 | 4026.821 | 4027.035 |
| GMM1异步 | 3923.770 | 3923.602 | 3923.318 |

### 这些字段能说明什么

1. **收益体现在kernel Duration内**。Wait Time仍在约0.07μs量级，没有与272μs相应的变化。
   当前CSV不支持把收益归因于host调用变快；内部完成边界/流水改动的因果证据来自版本控制和A/B/A。
2. **Wait Time不是内部flag等待总和**。官方将它定义为算子执行等待时间；这里是CANN task级字段，
   不能拿它表示FIX_S、L1供数等待，或当作内部无气泡证明。核内同步仍会影响Duration。
3. **Block Num不是利用率**。32只表明报告的任务切分数量；不代表32个Cube全程满载，
   也不能据此计算AIV忙碌率。Type/Accelerator Core在这批数据中为N/A，并不意味着没有执行计算。
4. **Name相同不等于内核相同**。三版kernel符号相同，必须用设备库SHA区分实现，不能只看函数名。

字段解释与附加硬件指标机制参考[官方PyTorch Profiler说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_1149.html)。
本次以实际CANN B070导出文件为准，不把其他版本文档中的可选字段假定为已采集。

## 为什么现有文件无法判断bound

九份 `profiler_info.json` 均记录：

```json
{"_profiler_level":"Level0","_aic_metrics":"ACL_AICORE_NONE","_l2_cache":false}
```

九份 `kernel_details.csv` 都只有：

```text
Device_id, Name, Type, Accelerator Core, Start Time(us), Duration(us), Wait Time(us), Block Num
```

也检查了底层 `op_summary_*.csv`，只有任务、stream、时间、block、输入输出描述等字段，没有隐藏的硬件利用率列。
因此不是JSON精简把指标删掉，也不是换一个CSV解析器就能恢复；这一轮采样没有采到这些计数器。
已有timer可以解释阶段边界，但新GMM1为ISSUE，不能把ISSUE条带累计换算成MMAD利用率。

## 后续怎样结合profiler优化

| 要回答的问题 | 后续需要的证据 | 不能直接作出的结论 |
| --- | --- | --- |
| Cube是否有效工作不足 | 有效数学工作量、kernel时长、目标精度的硬件能力，以及支持的Cube流水计数 | Cube时间占比高不自动等于有效TFLOPS高 |
| 是否供数不足 | 支持的MTE1/MTE2与Cube流水指标，配合tile推进/预取时机的单变量对照 | MTE占比高不自动等于HBM饱和，可能来自重复搬运或依赖 |
| HBM/L2是否瓶颈 | 对应层级的读写字节/带宽、cache计数，匹配本机可持续带宽参考 | 仅凭总Duration无法判断HBM或L2 bound |
| FIX/下游是否反压 | FIX相关观测、生产/消费依赖、End分布，以及异步协议A/B/A | FIX_S等待可能覆盖前序MMAD，不能等同纯FIX写出时间 |
| wave切换是否损失 | Begin/End次数、同核任务推进、完整通信尾部 | 不能跨核累加区间作E2E，也不能把ISSUE端点当完成 |

当前main的完整通信 `performance.py::collect_profile` 本来就使用 `Level1 + PipeUtilization`；
后续旧/新main五BS测试已完成，核实B070实际导出了49列及有效的流水数值，见上述补充报告。
该集合有指令cache miss，但没有HBM/L2带宽及L2数据cache命中率。不要把完整通信数据与历史Level0纯计算数据拼成同口径对照。
若要对4196→3924的纯计算后端做硬件bound归因，需要另外同口径采两版；本次遵守“不重跑程序”，只列为TODO。

## 原始证据定位

服务器根目录 `/home/zyx/`，每行目录各含round0/1/2三份CSV、配置和原报告：

| 目录 | 版本 | 设备库SHA256 |
| --- | --- | --- |
| `wave-lifetime-original/` | 原始纯计算 | `6f2a3169bc886b711353492eb2b63aab42d3e635233531dfff37d558ef94f959` |
| `gmm1-async-baseline/` | GMM2 wave | `389aad5887ab778fa50a7b459f81e56689cefc567f2c25cc0b9cf60377fc9446` |
| `gmm1-async-v1/` | GMM1异步 | `30a117c2fc9728d070b64c69b04b54ff98ea944c126f41b0bfae34d39ad5e73f` |

```bash
find /home/zyx/wave-lifetime-original /home/zyx/gmm1-async-baseline /home/zyx/gmm1-async-v1 \
  -name kernel_details.csv -o -name profiler_info.json -o -name 'op_summary*.csv'
```

本次原始文件快照：服务器 `/home/zyx/megamoe-existing-profiler-comparison.tar.gz`，
桌面 `C:/Project/Infra/ops-transformer-hs/tmp/megamoe-existing-profiler-comparison.tar.gz`。
桌面解包及逐文件SHA/统计记录在 `tmp/megamoe-existing-profiler/comparison.json`；
CSV只读导入，重新统计75个样本并与原报告交叉验证，不修改已有profiler文件。
