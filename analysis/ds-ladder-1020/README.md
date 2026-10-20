# DS逐级迁移/消融实验报告

完成11个有效累计节点。当前本轮BS512候选中，l6相对同窗口L0降低2.32%，仍比同窗口布局优化DS慢11.49%。**尚未追平DS，未合入master。**

不是预设迁移必然有效：有效但变慢的节点继续保留在累计链；构建失败和修复重试另记，不计为优化实验。数学舍入、scale算法和route weight位置未悄悄改成DS，不把少做数值工作当成等价收益。

## EP8 BS512累计结果

完整通信融合kernel Duration，timer-off；H7168/I3072/E384/TopK6/shared0、FP8/FP4、balanced、seed20260922。预热10、采20取后10，逐轮rank-min再对两次独立启动求均值；rank-max及逐rank样本保留。顺序L0→父版→本版→DS→DS→本版→父版→L0（L1父版即L0，不重复启动）。

|节点 / 本步迁移|L0 μs|直接父版 μs|本版 μs|布局DS μs|较父版降耗|较L0降耗|
|---|---:|---:|---:|---:|---:|---:|
|L1 Combine Scalar读metadata|799.94|799.94|796.94|701.05|+0.38%|+0.38%|
|L2 N半块提前FIX|799.30|797.76|799.92|698.42|-0.27%|-0.08%|
|L3 GMM2同向分核|797.77|800.37|800.78|701.12|-0.05%|-0.38%|
|L4 DS lookahead顺序＋旧完成发布|800.74|801.02|822.49|699.34|-2.68%|-2.72%|
|L5 共享两层异步epilogue协议|798.87|821.13|787.43|698.25|+4.10%|+1.43%|
|L6 CV原地压缩＋整stage写出|800.24|785.94|781.65|701.09|+0.55%|+2.32%|
|L7 保留数学合同的融合VF|799.80|783.60|787.21|702.21|-0.46%|+1.57%|
|L8 Dispatch Scalar写metadata|799.10|788.94|787.97|702.38|+0.12%|+1.39%|
|L9 Dispatch Scalar读索引|799.50|787.64|787.62|700.51|+0.00%|+1.49%|
|L10 Dispatch strided owner|798.18|788.22|785.89|700.15|+0.30%|+1.54%|
|L11 Dispatch源行互素打散|799.08|786.58|787.65|698.51|-0.14%|+1.43%|

各行是不同时间窗口，收益以本行对照为准；不能直接相加单项百分比。小于波动量的差值不宣称稳定收益。本轮最低值也不等于已通过七shape验收的最佳统一版本。

## 目前最重要的结论

- L4单独迁移DS任务顺序反而变慢；L5配套共享epilogue序列/真正FIX full与empty生命周期后改善，说明任务顺序与完成协议必须作为闭合系统审查，不能把某个flag或排列视为独立“神优化”。L5相对较慢L4的约4%不等于相对初始版的约4%。
- L6在新协议上整stage写出有小幅额外收益；旧协议上同名策略曾无收益，体现交互作用。
- 融合VF、Scalar metadata/索引以及通信owner/源行打散的收益必须看表中真实结果。机制更接近DS，并不保证本工程剩余数据布局、完成边界和生成指令也已经相同。
- 大差距仍未闭合。下一批优先迁移完整AIC compute leaf及GMM2 SF布局/供数协议；不是再盲目增加buffer。具体差异、代码入口与风险见[差异账本](DIFFERENCES.md)。

## 中间版本与阅读入口

初始版：`50d3ebe`，设备`dd5b2d7`。中间版本均独立分支；总工作分支`codex/megamoe-ds-ladder-1020`。

|节点|设备提交|独立分支|完整被动timer开销|
|---|---|---|---:|
|l1|`be8a290`|`codex/megamoe-ds-ladder-l1-1020`|1.89%|
|l2-r1|`7fc7fad`|`codex/megamoe-ds-ladder-l2-r1-1020`|1.48%|
|l3|`95a3de5`|`codex/megamoe-ds-ladder-l3-1020`|2.10%|
|l4-r2|`d07f258`|`codex/megamoe-ds-ladder-l4-r2-1020`|1.34%|
|l5-r1|`31794c3`|`codex/megamoe-ds-ladder-l5-r1-1020`|1.13%|
|l6|`1f44a59`|`codex/megamoe-ds-ladder-l6-1020`|1.36%|
|l7|`cf804e2`|`codex/megamoe-ds-ladder-l7-1020`|0.89%|
|l8|`c086a22`|`codex/megamoe-ds-ladder-l8-1020`|1.02%|
|l9|`db59c66`|`codex/megamoe-ds-ladder-l9-1020`|1.57%|
|l10|`0fc3df0`|`codex/megamoe-ds-ladder-l10-1020`|1.39%|
|l11|`1fd5708`|`codex/megamoe-ds-ladder-l11-1020`|0.92%|

每个节点：`candidate-a/timer_trace.json`为完整32组×AIC/AIV0/AIV1的96轨派生图；`timer_all_events.json`保留原始等待事件，派生图仅留白，不重排时间。`candidate-a/setup/rank_i/.../kernel_details.csv`为正式off CSV，`.../profiler/timer_on/...`为配对on CSV。图来自单次rank0观察，不是两次正式均值；ISSUE不是硬件完成，不能与PMU活跃时间混算。

本地证据：`C:\Project\Infra\ops-transformer-hs\outputs\ds-ladder-1020\checkpoint-4`，大小/SHA清单`manifest.json`已核验。保存各节点源码归档、实际kernel与host扩展、构建命令/日志、有效配置、逐rank输出差分和失败记录。统计摘要含每个节点实际kernel SHA及独立启动样本。

## 验证与限制

- 有效节点均通过8卡有限值/逐位差分；DS与其冻结DS输出差分，未声称我们与DS逐位相等。按用户要求未跑独立Golden；这不是正式数学精度验收，不能据此直接合入master。
- CPU测试执行生产调度、行映射、credit或编排代码；不能证明硬件可见性。共享epilogue后32项、strided owner后33项CPU测试，设备kernel和host扩展均需原生构建成功。
- 本轮未新增task-block PMU，不复用历史PMU冒充本版计数器。正式性能、调试流水、PMU三种用途分开。
- 小M闭合任务流仅在实际各专家M≤128、无共享专家且N2 tile数≤核数等条件成立时启用；其他场景保留原Wave。Host的CV选择仍使用均衡期望M，不保证偏斜性能。
- 同一SO跨shape/路由筛选另列；不同分支的最低成绩不能拼接为一个不存在的统一实现。

## 未完成/失败构建记录

- `l2`：build=1，campaign=None。原样保留，不计为有效优化。
- `l4`：build=1，campaign=None。原样保留，不计为有效优化。
- `l4-r1`：build=1，campaign=None。原样保留，不计为有效优化。
- `l5`：build=1，campaign=None。原样保留，不计为有效优化。

执行顺序见[PLAN](PLAN.md)；调度、同步、UB边界的详细说明见[差异账本](DIFFERENCES.md)。
