# 完整 EP8：rank0 硬件流水指标补充

2026-10-10。只读已有 `rank_0/bs_<B>/profiler/*_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv`，没有重跑NPU，也没有修改原始CSV。

## 两批数据必须分开

| 数据集 | 实际采样配置 | 可用指标 |
| --- | --- | --- |
| 历史纯计算4196→4027→3924 μs三版 | Level0 / ACL_AICORE_NONE | 仅任务时间、block等基础字段，见 [历史报告](PROFILER_COMPARISON.md) |
| 刚完成的旧main/优化main完整EP8 | Level1 / ACL_AICORE_PIPE_UTILIZATION | MAC、Scalar、MTE、FIX流水时间/占比、指令cache miss、cube_utilization |

“三版没有Cube/MTE指标”只适用于第一行，不能概括第二行。本报告补充第二行已有数据，不能把它标成历史纯计算三版的计数器对比。

两版身份、环境、精度和完整E2E见 [EP8验收记录](EP8_MAIN_VALIDATION.md)。本次核对6个性能run × 5个BS，共30份rank0 CSV：

- 每份49列，20条MegaMoE调用；按原规则选择后10条，排除前缀Exp等其他算子。
- 每版每BS三组，共30条rank0样本。CSV每条Duration与原 `profiling.json.per_rank_rounds_us[0]` 对齐，误差小于 `1e-6 μs`。
- 30份 `profiler_info_0.json` 均为Level1 / PIPE_UTILIZATION，`_l2_cache=false`。所列硬件指标均为实际数字，不是仅有表头的N/A列。
- 各列对选中样本做算术平均；`*_ratio` 从原始0–1转为百分比，`cube_utilization(%)` 原本已是百分比，不重复乘100。
- 下面Duration是 **rank0均值**，不是上一份E2E表的rank-min-mean，也不是rank-max-mean。

## BS4096：完整指标对比

| 指标 | 旧main | 优化main | 变化/说明 |
| --- | ---: | ---: | --- |
| rank0 kernel Duration（μs） | 4457.092 | 4255.712 | 减少201.381 μs |
| Wait Time（μs） | 1.591 | 1.592 | 任务间等待，不是核内flag等待 |
| Block Num / Mix Block Num | 32 / 64 | 32 / 64 | 相同 |
| aicore_time（μs） | 4456.749 | 4255.358 | 工具汇总时间 |
| aic_total_cycles | 224831404.3 | 211423278.5 | 跨block的执行cycle总和 |
| AIC MAC时间（μs） | 3942.715 | 4015.252 | 没有减少 |
| AIC MAC占比 | 88.473% | 94.363% | +5.890个百分点 |
| AIC Scalar时间（μs） | 886.535 | 957.159 | 不是所有同步等待的总和 |
| AIC Scalar占比 | 19.903% | 22.500% | +2.597个百分点 |
| AIC MTE1时间（μs） | 1375.438 | 1399.124 | L1→L0供数 |
| AIC MTE1占比 | 30.890% | 32.900% | +2.010个百分点 |
| AIC MTE2时间（μs） | 2526.738 | 2571.623 | 外部存储→核内搬运 |
| AIC MTE2占比 | 56.687% | 60.437% | +3.750个百分点 |
| AIC MTE3时间（μs）/占比 | 0.001 / 0.000% | 0.001 / 0.000% | 不代表FIX或AIV没有写出 |
| AIC FIX时间（μs） | 816.520 | 823.724 | 不是Scalar等待FIX的时长 |
| AIC FIX占比 | 18.300% | 19.363% | +1.063个百分点 |
| AIC指令cache miss率 | 0.100% | 0.100% | 原CSV粒度有限，不是L2数据cache |
| aiv_time（μs） | 4452.422 | 4250.758 | 工具汇总时间 |
| aiv_total_cycles | 470175664.0 | 448879950.1 | 跨block的执行cycle总和 |
| AIV Vector时间（μs） | 949.662 | 949.982 | 基本不变 |
| AIV Vector占比 | 21.310% | 22.353% | +1.043个百分点 |
| AIV Scalar时间（μs） | 1565.592 | 1297.559 | 减少268.033 μs，不能直接等同收益来源 |
| AIV Scalar占比 | 35.167% | 30.520% | −4.647个百分点 |
| AIV MTE2时间（μs） | 1924.866 | 1911.522 | 减少13.344 μs |
| AIV MTE2占比 | 43.243% | 44.973% | 分母缩短，时间下降也可使占比上升 |
| AIV MTE3时间（μs） | 1121.784 | 1116.678 | 减少5.107 μs |
| AIV MTE3占比 | 25.197% | 26.277% | +1.080个百分点 |
| AIV指令cache miss率 | 0.100% | 0.100% | 不是L2数据cache |
| cube_utilization | 95.537% | 94.091% | −1.446个百分点，不能与MAC占比混同 |
| HBM/L2读写带宽、字节数 | 未采集 | 未采集 | 不能用MTE时间推算GB/s |
| L2数据cache命中率 | 未采集 | 未采集 | `_l2_cache=false` |

## 五个 BS 的关键趋势

每格「旧 → 新」，均为rank0三组均值。这里的Cube utilization是CSV字段原值，不是有效TFLOPS/硬件峰值。

| BS | Duration μs | AIC MAC占比 | AIC MTE1占比 | AIC MTE2占比 | AIC FIX占比 | Cube utilization |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 878.339 → 843.142 | 32.59% → 33.86% | 43.63% → 45.31% | 17.29% → 17.82% | 10.87% → 11.94% | 97.95% → 98.21% |
| 96 | 912.975 → 876.596 | 31.35% → 32.66% | 42.06% → 43.79% | 20.24% → 20.69% | 11.11% → 12.16% | 97.98% → 97.96% |
| 256 | 968.661 → 930.243 | 38.52% → 40.22% | 42.96% → 44.80% | 31.54% → 32.63% | 11.10% → 12.31% | 97.05% → 96.89% |
| 1024 | 1287.027 → 1256.554 | 78.52% → 84.29% | 44.31% → 47.19% | 63.78% → 71.20% | 17.25% → 18.26% | 93.78% → 90.27% |
| 4096 | 4457.092 → 4255.712 | 88.47% → 94.36% | 30.89% → 32.90% | 56.69% → 60.44% | 18.30% → 19.36% | 95.54% → 94.09% |

## 怎样解释，而不误判bound

1. **不同流水会重叠，不能将MAC/MTE/FIX/Scalar时间相加当E2E，也不能用100%减去占比之和得到气泡。**
   BS4096总Duration下降，但MAC、MTE1、FIX时间未下降。这更符合减少串行边界、改善重叠的假设，而非某一硬件流水自身执行时间大幅缩短；CSV本身不能把收益分配到各次Begin/End。
2. **MAC占比升高与AIV Scalar时间降低是有用的相关证据。** AIV Vector时间约950 μs不变，Scalar从1565.6降到1297.6 μs。
   但Scalar字段是指令执行统计，不等于“所有wait”，不能宣称268 μs全是删除FIX等待的收益。AIC Scalar时间反而上升，也必须保留这一事实。
3. **Cube utilization不是MAC占比，更不是量化精度下有效FLOPs的峰值达成率。** 官方给出的计算式基于 `total_cycles/(freq × core_num × task_duration)`，其中total_cycles是各block执行cycle之和，不能代替有效矩阵乘工作量。
   BS32的新版本该字段98.21%，但MAC占比只有33.86%；BS4096变快时该字段反而下降，说明不能拿它单独决定优化好坏。
4. **当前不能确定HBM-bound或L2-bound。** MTE2占比没有字节数和带宽信息，融合算子内还存在AIV供数、通信、同步以及不同阶段重叠。
   也不能因为MAC占比94%就宣布只剩6%优化空间；整体计数未分开GMM1/GMM2，且有效FLOPs、尾块填充、精度吞吐与各阶段关键路径都没有由这张表单独确定。
5. **小BS与大BS应分开研究。** BS32/96的MAC时间约286 μs，MAC占比仅约1/3；结合timer优先检查tile填充、任务管理和供数衔接。
   大BS则继续检查GMM2供数/FIX重叠和完整通信尾部。这里是下一步定位方向，不是已证明某个单一bound。

字段定义参考[昇腾官方op_summary说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0067.html)。
计数器按工具口径报告，不把它们当精确单核墙钟区间；频率变化等也会影响换算。以上没有对缺失带宽/cache指标填0。

## 证据与可复核性

使用 [EP8验收记录](EP8_MAIN_VALIDATION.md) 中六个性能run，仅取rank0。每个源CSV的完整路径、SHA256、原始表头、配置、选样数量与逐列统计保存在桌面 `tmp/main-ep8-profiler/analysis.json`。
只读导入使用表格技能的CSV读取流程，并独立核对原始性能样本。统计对象为完整融合kernel，不能分解为独立GMM1/GMM2的硬件计数器。

服务器原始文件归档：`/home/zyx/main-ep8-rank0-profiler-columns.tar.gz`，含30份CSV及对应profiling/profiler_info JSON。
SHA256：`6de7bb3a2c6c8b9a560c4fc7c691d69225fc7f503c98ca9518890bc222dd65fa`。
本地副本：`C:/Project/Infra/ops-transformer-hs/tmp/main-ep8-rank0-profiler-columns.tar.gz`。

BS4096第1组源路径示例：

```text
旧：/home/zyx/DeepEP-main-baseline/benchmark-runs/ep8_20261010_162847_04f18aa8/bs_4096/rank_0/bs_4096/profiler/rack07os7_403877_20261010163152580_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv
```

历史纯计算三版仍需另采同口径硬件指标才能补完整，不能拿这份新旧main对照替换；本次没有额外运行采样。
