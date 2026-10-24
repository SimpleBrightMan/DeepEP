# 目标分支EP32补测（10月24日）

用户指定`codex/megamoe-r31-two-round-wave-1023`。已核对远端HEAD `9f356345afd891d202d3edc3dfe3717b0cc3a7b7`，算法仍为`904131cfa5c22e27eb8e278ec9019ab4dbb0392a`。
研究工具新增`campaign.py --method-only control`，每BS只测该冻结版本两次，保留原ABBA默认模式。选用历史R1包的**control**，不是其combined候选；未修改生产分支、main或master。

## 有效结果

05柜CPU1/2/3/4：EP32、每卡BS4096，两次独立启动4181.9907/4182.6072μs，rank-min均值**4182.29895μs**，rank-max均值**4189.83095μs**。

真实通信完整融合kernel Duration；H7168/I3072/E384/topk6/共享0、FP8/FP4、balanced seed20260922。timer-off，warm10、采20取后10，逐轮rank-min/rank-max分别求均值，再对两个启动求均值。不能称rank-min为全局最慢rank延迟，也不是主机调用端到端。

64份原始CSV逐rank重算、每次20个目标kernel；全部worker正常退出。host/device身份以及两次每rank输入/输出哈希一致，输出有限。按要求未运行独立Golden。本轮没有timer或PMU。

本地证据根`outputs/r31-ep32-matrix-1024`，有效窗口`r05-1234-e`，含`summary.json`、`RESULTS.md`、`experiment.json`、`events.jsonl`及node0–node3的原始CSV。公开页只发布摘要，不公开机器路径和原始附件。

## 尚未完成

BS32/96/256/512/1024/2048没有本轮有效性能。五次窗口尝试均保留：首次05柜BS32停在输入准备，尚未进入预热；只取消本任务并确认worker退出。两次01柜和另一次05柜在预检门禁检测到外部任务。最后窗口完成4096两次后，下一BS门禁再次检测到四节点外部NPU任务，全组取消。

最后05柜复扫八节点均有外部任务，01柜也不足四节点。没有终止他人程序、重置设备或后台继续占卡。缺测不继承历史成绩，不与不同节点窗口拼均值；本轮没有旧版本或DS同窗对照，不能推断相对性能收益。

## 工具验证

四项CPU测试通过：单版本双启动报告、旧ABBA兼容、缺rank拒绝、输出漂移拒绝；未用CPU测试冒充NPU正确性。网站新增单版本EP32七BS表，缺测明确显示，旧数值不变。
