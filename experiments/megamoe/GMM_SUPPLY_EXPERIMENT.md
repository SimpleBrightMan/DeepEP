# 实验A：下一A窗口提前供数

基于unitflag-shared；与实验B相互独立，不合并B资源调整。
参考Yancy/DeepEP main c5d68f0的A窗口预取顺序，适配本分支的非连续输入Gather。

首个A窗口显式加载。后续窗口在当前窗口MTE2→MTE1完成通知之后、当前MTE1/MMAD消费之前，
提交到下一L1 A槽。下一槽先取得原MTE1→MTE2 empty credit；最后一个K窗口不越界预取。
KA可跨多个KB窗口时，只在KA边界预取；原KA/KB选择、L1/L0资源、tile和同步归还保持。
共享GMM的连续输入和路由GMM的分段Gather均使用这套供数逻辑。

默认生效，不增加运行参数；使用原urma_benchmark命令。完整重编wheel后运行。
重点比较AIC包络、GMM1/GMM2阶段空隙及无timer整核耗时；A prefetch仍可能与通信争用HBM/L2。
没有以预取提交时间替代完成时间，原Unit Flag和FIX完成发布均保留。

提交时未执行NPU性能测试；编译与精度验证状态由总实验报告记录，不据此文件声称收益。
