# 性能入口

按明确的完成边界测量核心 EP；Beta 端到端性能由实验入口维护。

## 规划内容与职责

- `bench_ep.py`：V1/V2 通信延迟与带宽。
- `_utils.py`：计时、预热、rank 聚合与结果写入。
- `tune.py`：存在合法候选空间和稳定测量流程后再实现。
- `results/`：可复现结果记录及 CSV 模板；目前没有实测结果。

[完整文件规划](../docs/repository-layout.md) · [架构说明](../docs/architecture.md)
