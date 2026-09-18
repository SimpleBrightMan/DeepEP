# 性能测量与复现

当前没有 benchmark 实现或目标 NPU 测量数据。本文件定义后续记录口径；[结果模板](../benchmarks/results/template.csv)只有表头，没有实测数据。

## 测量范围

| 类型 | 分别报告 | 基线要求 |
| --- | --- | --- |
| 核心 EP | dispatch 延迟、combine 延迟、明确口径的有效带宽 | 相同路由、精度、布局、规模与完成边界 |
| MegaMoE Beta | 完整 dispatch–GMM–激活–GMM–combine 延迟 | 数学语义、权重处理、精度与负载一致的组合流程 |

EP 通信带宽与 MegaMoE 端到端加速分别记录。异步 API 的 Host 返回耗时不代表 Device 完成延迟。测量必须说明事件、同步点及消费结果的边界。

## 必需记录

保存源码和基线提交、软件配套、硬件和拓扑、节点/rank 数、每 rank token 数、hidden、top-k、expert 数、dtype、实际变体、策略和 AIV 配置。记录预热次数、重复次数、随机种子、计时方法、rank 聚合方式与复现命令。

有效带宽按 `effective_bytes / elapsed_seconds` 计算，使用 GB/s 时以 `10^9` 换算。`effective_bytes` 必须明确是否计入路由重复、scale、元数据和跨链路重复传输；未定义字节口径时只报告延迟。

加速比采用匹配基线的 `baseline_latency / candidate_latency`，需保持统计方式和计时边界一致。记录 p50、p95 等统计值时明确采样和跨 rank 聚合方式。

## 执行过程

1. 固定代码、配套、拓扑、负载、策略与随机种子。
2. 先验证结果正确性、异步完成和 handle 条件。
3. 明确预热与测量区间，在完成边界上计时。
4. 保存各 rank 原始结果与聚合规则，再生成摘要。
5. 将复现命令、配置和日志位置与结果一同归档。

本地测量输出可放入根目录 `benchmark-runs/`（已忽略）。可公开的基线数据经核对后存入 `benchmarks/results/`；Beta 专属测量与对应版本和配置关联。
