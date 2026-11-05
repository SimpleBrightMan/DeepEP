# 实验B：B四槽与独立反量化UB环

基于unitflag-shared；与实验A相互独立，没有提前下发下一A窗口。
参考Yancy/DeepEP main c5d68f0的B供数加深和输入/输出credit分离，保留本分支原FP4/NZ数学与K/N顺序。

| L1用途 | 分配 |
|---|---|
| A | 2×64KiB，[0,128KiB) |
| B | 4×64KiB，[128,384KiB) |
| Scale A | 2×32KiB，[384,448KiB) |
| Scale B | 2×32KiB，[448,512KiB) |

A窗口容量上限收缩到64KiB。AIC/AIV0同时采用修改后的K计划，不允许K512×M256溢出A槽。
B full使用跨核flag4..7，empty使用8..11；上下文收尾flag由10移到12，避免覆盖B credit。
scale仍使用独立两槽与原4096-K尺度，不跟随四槽B一起复用。

AIV0 UB输入4×16KiB、输出3×32KiB，总160KiB。输入等待V最后读取，输出等待MTE3最后读取；
两套游标独立轮转并跨tile/投影/专家保留。输出按256B向量交错，VF步长和UB→L1 stride同步更新。
没有直接采用Yancy的224KiB UB布局，因此保留当前192KiB最低UB要求。

默认生效，不增加运行参数；完整重编wheel后使用原benchmark命令。
重点看GMM2 B等待、AIV0供数和整核性能；B供数加深可能增加并发访存压力。
提交时未执行NPU性能测试；编译/精度状态由总实验报告记录。
