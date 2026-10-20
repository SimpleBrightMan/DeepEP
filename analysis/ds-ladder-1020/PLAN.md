# DS累计迁移实验链

基线：`50d3ebe`（设备`dd5b2d7`），冻结参考为已测布局优化DS，不跟随上游HEAD变化。

目标是用逐步替换找出重要差异，而非预设某项必然有效。累计链保留有效但回退的节点；最佳链另记，错误/死锁不作为有效父版本。每项保存源码提交、实际host/kernel SHA、命令、有效配置、性能、输出检查，关键节点保存完整被动timer与配套CSV。不是把不同分支最低值拼成候选。

## 差异与执行优先级

|顺序|差异/迁移|为什么这样排|仍需保持的约束|
|---|---|---|---|
|L0|当前统一CV/GM候选|同窗口起点|真实通信、同输入、同数学实现|
|L1|Combine metadata由GM直接Scalar读取，替代小批DMA→UB→Scalar|独立叶子边界，先隔离短任务控制成本|保留既有GMM2 ready；目的rank/token/topk不变；先不改metadata ABI|
|L2|小M最后K的N半块完成后直接发FIX|验证MMAD/FIX接续，而非再删外层等待|每N半块unit flag，真正FIX full；不能覆盖未释放UB|
|L3|GMM2与GMM1同向分核|DS采用同向全局task分配；先独立验证反向分核差异|执行、空任务与完成通知必须同方向；其他兼容路径维持原默认|
|L4|统一计算task流/DS lookahead顺序|先对齐实际每专家M≤128、无共享专家的闭合协议|AIC/AIV0同序，GMM1分核不变供AIV1重放；保留GMM2 phase级FIX完成发布，明确尚未等于DS异步GMM2 epilogue|
|L5|两层共享epilogue task序列与FIX full/empty；AIV1确认GMM2完成|在L4上单独移除GMM2 phase生产端排空，闭合DS式生命周期|两层共用issued/credit，GMM1由V/MTE3消费释放，GMM2由Scalar消费释放；不新增GM空间|
|L6|CV输入槽原地压缩＋stage末尾累计写出|先隔离32行量化scratch credit与多次DMA，不混入VF数学变化|保留BF16舍入和原有多VF；已读前缀才可覆盖；EMPTY保护MTE3写出|
|L7|SwiGLU+MX寄存器驻留融合VF|迁移DS的融合结构，不混入数值合同变化|显式保留两次BF16 RN、最大指数scale与后GMM2加权；其他激活模式和FP8格式沿用原计算|
|L8|Dispatch metadata由Scalar写GM，去掉逐槽S/MTE3元数据事件|迁移DS轻量控制面；先不改256B/row ABI|payload仍受双槽MTE credit保护；发布ready前DDR barrier覆盖Scalar metadata|
|L9|Dispatch route index直接Scalar读取|隔离小批DMA索引与MTE2_S成本|仍按真实prefix/ordinal取数，不改变源/目的排列|
|L10|DS strided通信owner|先隔离物理core→目标行分配，不混入源行置换|仅actual M≤128的small-linear；按真实行数贡献ready，第0个M256槽；M128仍一次最多4行，未宣称完整DS两行yield|
|L11|DS源行互素置换、紧凑metadata及readiness协议|牵涉布局和三角色依赖，作为闭合协议迁移|源/目的映射与数据可见性；不能仅换flag编号|
|终点对照|完整冻结DS|用于检查测量链和差距能否闭合，不冒充单项收益|若舍入/加权等合同不同，单独列为语义变更，不能称等价优化|

已接近一致、不重复计功：L1 A3/B4、小M K512/N128和L0A跨N复用、同MTE1队列上B ready覆盖此前A/scale、双128行CV槽、32行VF推进、2pull/4push与常驻通信UB。

## 测量

先EP8 BS512、H7168/I3072/E384/TopK6/shared0、FP8/FP4、balanced、seed20260922。完整通信融合kernel Duration，timer-off，warm10/采20取后10，逐轮rank-min；同时保留rank-max和逐rank样本。各累计节点与初始版、DS在同一机器交替，关键转折追加ABBA。重复启动不计作新增优化。

保留组合再筛七BS：32/96/256/512/1024/2048/4096，并检查random/skewed。按用户此前要求不运行昂贵独立Golden；输出有限值/逐位差分仅用于等价节点，语义改变需另列，不能用跳过精度掩盖行为改变。

默认完整被动timer：rank0、全部32组AIC/AIV0/AIV1，无额外排空，独立off/on核验开销。PMU按需task-block；利用率与ISSUE不相加为耗时账本。

只用空闲卡，私有目录/包/cache，每项独立输出，失败原样保留，不终止他人任务。若中途设备不可用，明确保存已完成与待完成，不编造后续收益。
