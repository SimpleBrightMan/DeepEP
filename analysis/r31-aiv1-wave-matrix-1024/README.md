# R31两轮wave＋AIV1完成等待统测

- 算法合并：`904131c`，父提交为两轮wave `aa2328d` 和R31异步完成 `f2f1197`。保留配对与host两轮预算，不叠加Combine配额。
- 控制包：`7512c6c`对应的已冻结两轮wave设备/host包；`aa2328d`只追加说明，生产代码一致。候选重新构建host及device，包身份逐rank校验。
- EP8、EP32；每卡BS32/96/256/512/1024/2048/4096。H7168、I3072、384专家、topk6、共享0、FP8/FP4、balanced、seed20260922。
- 每shape独立启动ABBA：control-a → combined-a → combined-b → control-b。真实通信完整kernel Duration，warmup10、采20取后10，先逐轮rank-min再均值；同时保存rank-max。不与历史不同机器窗口拼均值。
- a轮在主性能后另做rank0完整32group被动timer-on全20轮。JSON、原生稀疏数据与同rank off/on CSV配对；b轮只有timer-off。
- 用户要求性能优先：`accuracy=not_run_user_requested`；检查有限值、off/on与版本间指纹等价，不宣称独立Golden通过。

## 执行

`build.py`在隔离目录构建，不安装/修改共享Conda；`--resume-package`只允许完整原生构建后的打包导入恢复，先验证冻结源码。

`campaign.py`显式接收inventory、host-key、依赖、包、EP和节点；不保存登录信息。EP8需要1节点、EP32需要同柜4节点。每case先持共享锁并检查全部8卡无进程，全组READY后才GO；任一失败立即停止后续case。不得为了测试终止他人的任务。

`report.py <campaign-output>`从所有rank原始CSV重算统计，审计正常退出和版本身份；完整矩阵才记audited_complete。`--partial`只发布已完成ABBA的shape，其余保持pending。

## 合并语义

完成事件仍按wave：AIC在FIX队列等待EMPTY并发布FULL，Scalar只写issued序号；AIV1观察issued后等待真正FULL，再重放相同wave规划逐slice累加ready，最后归还EMPTY。issued不能当完成，不能删除最终credit回收。小任务CV路径维持原协议。

## 当前验证边界

CPU预算/完成事件/调度/工作区5项测试通过，CANN B080原生编译及隔离导入通过；NPU性能与回归状态以本轮experiment.json、summary.json为准，不能继承上一轮EP32/BS4096的11.15%。

首次打包导入缺少libshmem.so搜索路径，补齐检查子进程环境后从完整构建恢复；保留首次日志，没有修改冻结生产源码或为了打包重新跑NPU。
