# 性能测量与复现

已提供 [benchmark 入口](../benchmarks/README.md)、无设备规划与测量协议引擎。原生适配器尚未实现，当前**没有设备实测数据**；[结果模板](../benchmarks/results/template.csv)只有表头。完整配置、时序、能力门禁和后端接入条件见[详细设计](benchmark-design.md)。

## 测量范围

| 模式 | 计时工作 | 准备工作（不计时） |
| --- | --- | --- |
| dispatch | 调用至输出可安全消费；fresh 含其通知/布局操作 | 输入与路由；cached 模式额外准备可复用 handle |
| combine | 调用至 BF16 合并输出可安全消费 | 用真实 dispatch 准备匹配 handle 和 BF16 合并输入 |
| roundtrip | 直接测量 dispatch → 同一 handle 的 combine，无专家 GEMM | 输入、路由及 cached handle；FP8 接收数据恢复 BF16 在计时内 |

roundtrip 不能由两份单项延迟相加得出。普通布局按目标 rank 去重，展开布局保留有效槽位重数。带权重时值通道独立传递，不额外把 gate 乘到 hidden payload；这不是完整 MoE 业务延迟。

计时采用同步 Host 墙钟 `perf_counter_ns`。每轮先 barrier、同步，再开始计时；调用后等待完成依赖并同步，最后停止计时。API 内分配计入，预先生成输入、正确性检查、结果释放和样本收集不计入。该口径包含 Host 调度与完成等待，不能与纯 kernel event 时间混用。

## 统计与字节定义

保存全 rank、逐 iteration 原始微秒样本。对齐同一轮后取 rank MAX，再计算 mean/min/max、nearest-rank p50/p95；不能先算各 rank 的分位数再取 MAX。

`effective_bytes` 是全 rank 的**有效 hidden payload 逻辑字节数**：

- 普通布局：每个源 token 的有效目标 rank 数 × H × 元素字节数。
- 展开布局：每个源 token 的有效路由槽位数 × H × 元素字节数，重复 expert 槽位分别计数。
- BF16 dispatch 每元素 2 字节，FP8 dispatch 每元素 1 字节；combine 总是 BF16、每元素 2 字节；roundtrip 两方向相加。
- 包括本 rank 路由；不包含 padding、FP8 scale、索引、权重、协议开销或链路重传。

报告 `bandwidth_gbps = effective_bytes / (mean_rank_max_us × 1000)`，使用十进制 GB/s。这是约定口径的有效带宽，**不是物理链路带宽、单 rank 带宽或互联利用率**。路由全无效时可出现有效字节数为零，但原始延迟仍必须为正数。

## 正确性与记录要求

测量前后均验证相同负载的路由、布局、payload/scale、combine 数值和值通道，按[精度标准](precision.md)使用独立 Golden 和上游同 dtype 标杆。任一 rank 失败、样本缺失/非有限值、能力不足或资源退出失败，都不能发布成功结果。

结果保存完整配置、配置指纹、源码 SHA、公开软件/设备/拓扑标识、计时/字节口径及原始样本，见[结果字段](../benchmarks/results/README.md)。版本、负载、精度、拓扑、缓存策略和完成边界一致才可比较。没有匹配的实测基线，不填写基线延迟或加速比；未测量不填写 0。

输出放入被 Git 忽略的 `benchmark-runs/` 新目录。发布前逐项检查元数据，只保留可公开、可复现的信息；入口不会自动采集机器名、地址、环境变量或本地路径。dry-run 的规划文件不能作为性能证据。

MegaMoE Beta 后续独立记录完整 dispatch–GMM–激活–GMM–combine 的延迟及语义匹配基线，不能与此处 EP 通信有效带宽混用。
