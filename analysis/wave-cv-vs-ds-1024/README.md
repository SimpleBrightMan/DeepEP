# Wave＋CV、BS512 lag统一候选与布局DS

分支`codex/megamoe-wave-cv-lag512-1024`，父提交`b08d26a`。

删除Linear专用入口、AIC/AIV1重放函数及无消费者的调度头；统一Wave，保留原CV/GM派发、wave预算、gate/up配对及同步依赖。所有EP按每卡实际BS≥512开启GMM2滞后一波。不同于`codex/megamoe-r31-wave-lag1-all-1024`的全BS lag，不能继承它的性能表。

CPU测试执行实际角色入口（CV/GM、AIC/AIV0/AIV1及replay），不提供Linear桩；同时执行Wave顺序、阈值、行范围、owner及资源协议测试。CPU通过不能替代设备验证。

计划固定模型H7168/I3072/E384/topk6/shared0、FP8/FP4、balanced seed20260922，EP8/32覆盖BS32/96/256/512/1024/2048/4096。两方案同节点同窗ABBA，性能timer-off，warm10/采20取后10；每种方法两次独立会话连续七BS，capacity4096。EP8单节点、EP32四节点，启动前复查全组空闲，不终止他人任务。

参考为历史布局优化DS，必须校验真实头文件及host扩展身份，物理权重预排匹配实际每专家M。两方案原始输入逐rank一致；数学规则不同，不要求跨方案输出逐位一致。独立Golden按用户要求跳过；timer单独采样并保存与kernel_details.csv配对。新结果不与不同机器的历史4184/4505μs拼均值。

证据根`outputs/wave-cv-vs-ds-1024`。本文件初建时尚未完成本候选原生构建或新性能验收，最终以审计摘要为准。
