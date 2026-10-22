# URMA trace dashboard

双击 `urma_trace_dashboard.html` 即可使用；无需服务器、网络、Python 或 NPU。拖入一个或两个 `allrank.json` / 单 rank `urma_trace.json`，两份文件独立调参、左右对比。每项指标以及各 rank 的算术均值均可点击，查看代入数字的计算过程；按 Esc、点击外部或关闭按钮退出。

首页采用两行摘要：上排 Dispatch/Combine 组内与跨组带宽，下排 MFU（带空泡）/参考掩盖率。allrank 默认展示算术均值，逐 rank 明细和其它指标默认折叠；可分别展开。调整参数后保留展开状态，CSV 始终包含全部指标和 rank。

支持当前 URMA trace schema 1，以及 GMM 合并显示的 JSON 合成事件 33/34 和上下文批次 35–38。批次条内的 Wait、条间空隙不计入原始非等待累计核时；按 `source_records` 恢复各 tile 的形状和时长，AIV0 反量化不计入 AIC 忙时。不是 aggregate `timer.json` 或 `summary.json` 解析器。缺失 Quiet、溢出、记录数不匹配等情况不会伪造完整带宽。两份文件最多同时存在，新的文件可替换现有数据。

## 指标口径

- **Dispatch/Combine、组内/跨组带宽**：按当前 rank、引擎和 peer，将首 PUT 到对应 Quiet 完成重建为一个通信窗口；该类全部窗口取时间并集。带宽为实际 PUT payload / 并集时间。并发核心不重复计时，不乘双向系数、不平均核时。点击可另看包含中间空隙的首尾跨度带宽。
- **组内**只计同组其它卡。本卡路径已经融合进 Pack 和 GMM2，无法分出准确的独立本卡带宽；不将本卡字节硬加入网络带宽。
- **无空泡时间**：分别对每个核心累加对应阶段的记录，再取最大值。包含通信 Quiet/Publish、GMM 完成 Drain；不包含独立 Wait 或条间空隙。条内部等待不能从 trace 中分离，因此不是硬件纯忙碌时间。GMM 合并条通过 `source_records` 恢复分条核时，保证合并前后统计一致。
- **融合时间**：各 rank 原始 `metadata.envelope_us`，即诊断调用跨度。优先使用该字段，以保留隐藏 Wait 后的原始时间边界。各 rank 时钟独立，绝不跨 rank 拼接时间轴。
- **AIC 时间**：同一 rank 的所有 AIC 上，首个 GMM 开始到最后 GMM/drain 结束的包络，含中间空泡，不包含首个 GMM 之前的输入等待。参考掩盖率按用户定义取此包络 / 融合时间，属于时间覆盖占比。
- **计算量**优先使用实际 AIC tiles：GMM1 `4×H×Σ(rows×columns)`（gate/up 两路），GMM2 `2×I×Σ(rows×columns)`。缺少几何信息才用均衡估计 `6×BS×TopK×H×I`；BS 为每 rank 的输入 token 数，不再乘 EP 或专家数。只计有效矩阵乘，未计 padding、激活、量化与 Reduce。
- **MFU（带空泡）**：`F/(T_AIC×Peak) = (F/Peak)/T_AIC`。两种写法等价，页面和 CSV 仅保留这一项，弹窗展示两种计算式。峰值默认 765 TFLOPS/card，需对应实际计算精度/稠密峰值。
- **理论通信**固定计入 Dispatch + Combine：`D=(H×dispatchBytes+512)×TopK×BS`，`C=H×combineBytes×TopK×BS`，`T_comm=T_D+T_C=(D+C)/EP×跨组卡数/(链路数×单向GB/s×协议效率)`。默认四根线、每方向 56 GB/s、效率 0.8。Dispatch 和 Combine 默认分别为 1B、2B/元素，修改两者都会更新理论通信耗时和理论掩盖率。Combine 不额外虚构 512B 开销。模型假设对各 peer 均匀分配，未建模本组链路瓶颈。
- **理论掩盖率**为 `(T_comm+T_compute−T_fused)/min(T_comm,T_compute)`；允许负值和超过 100%。理想模型中的时间是性能下限，实际执行还含软件开销、同步、Pack/Reduce 等。此比值不等于硬件测得的真实重叠率。

实际 Dispatch scale 为 `H/32` 字节/路由行（当前 case 为112B）；理论附加512B与它不是同一个量。默认带宽读取 JSON 的 `args.bytes`；改为“模型”模式后才能用元素字节数重新估算带宽，页面会加估算标记。峰值、BS、H、I、TopK、专家数、EP（单 rank）、拓扑和链路模型均可配置。专家数只影响路由分布，不作为总 FLOPs 的额外倍数。

参数尽量从 Pack 行数和 AIC tile 维度推断；BS 推断使用默认 TopK=8，若不是该 case 请人工核对。两侧文件参数独立，可复制左侧参数到右侧。文件和计算数据仅留在浏览器内存；不上传、不持久保存。

## 开发验证

修改指标 JS 后，将其重新内嵌到可独立分发的 HTML：

```bash
python3 experiments/megamoe/tools/build_urma_dashboard.py
node --test experiments/megamoe/tests/test_urma_dashboard_metrics.cjs
```

发布时只需复制 HTML 一份文件，其余文件是开发源代码与检查脚本。
