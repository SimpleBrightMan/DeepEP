# 统一 Wave+GM 对照

父版本为推荐统一Wave基线`4a0de857`，不包含提前归约或扩大rail的实验。分支`codex/megamoe-wave-gm-1025`。

- 单一修改：`UseGmm1CvForLaunch`统一返回false，host预算、device UB绑定及kernel派发共同切换至既有GM路径。不改数学计算、gate/up配对、两轮Wave、BS≥512的lag或原Dispatch/Combine通知。
- 基线原本仅EP8且期望每专家M≤128时使用CV：本模型EP8 BS32/96/256/512/1024变更，BS2048/4096不变；EP32七个BS原本就全部GM，是负对照，不得把其波动归因于CV切换。
- GM路径不等于只增加一次GM往返：它同时采用既有GM完成/消费粒度、64行激活量子和6个Dispatch/2个Combine UB槽；CV路径是原生32行量子和2/4槽。实验检验现成GM后端的组合，不声称隔离单次GM流量。
- 默认计时器仍是完整被动版：主性能timer-off；另采rank0全部32组AIC/AIV0/AIV1流水，成对保存CSV，不为timer增加同步。
- EP8/EP32各七BS，同节点有序列表、连续多BS、同窗ABBA。核对原生身份及输出差分，不跑独立Golden；未完成结果不补造。
- 上一轮连续TCPStore测试有主机协调故障。本轮只在隔离测试控制层使用CPU Gloo（预检先确认可用），设备Dispatch/Combine仍为SHMEM/MTE，不引入HCCL或将CPU通信计入kernel。两方案使用相同控制层，记录该差异，不与旧窗口绝对时间混算。

构建/测试结果待实测填写；不预设GM优于CV或相反。
