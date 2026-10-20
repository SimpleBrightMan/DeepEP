# DS终点与成组二分诊断

本轮从累计L12（设备8293d6d）延续；L6（1f44a59）冻结为性能对照，不改master。
这是一条归因实验线，不把原样调用DS当成我们的优化。

## 顺序

1. D0：把冻结布局优化DS的完整device/布局/调度/通信/Python包装导入本项目的诊断目录，复用同SHA的host扩展，在新目录重新JIT。与原冻结DS、L6和L12同窗口对照EP8 BS512/1024。
2. C：闭合迁移计算数据面，包括完整AIC compute leaf、必须配套的AIV0交接和scale生产/消费；无额外运行时转置的边界才可拆开。源/目的ABI不闭合时合并为一个包。
3. R：闭合迁移头部metadata、AIV1推进、Dispatch/Combine ready和尾部归约。比较OO/DO/OD/DD；不能把死锁或多余适配拷贝当成有效对照。
4. 对重要的包反向恢复原实现并二分；两半单独无效但组合有效时记录交互，不丢弃组合。
5. 有效统一候选复查七BS和random/skewed，再考虑其他EP。

数值舍入、SF算法和路由加权位置另记语义节点；保持旧合同的变体对旧版逐位差分，完整DS对齐版对冻结DS逐位差分，不能混用判据。未跑独立Golden，不能据此合入master。

## 实验合同

EP8，H7168/I3072/E384/topk6/shared0，FP8/FP4，balanced，seed20260922。真实完整通信kernel Duration；预热10、采20取后10，逐轮rank-min均值并同时报告rank-max。筛选后对关键项同窗口交替复测，不拼不同分支最低值。

正式性能timer-off，完整被动timer另测rank0全部32组/96轨；不加同步。保存off/on配套CSV、实际kernel与host扩展、源码及JIT身份。PMU需要时单独task-block采集。

只用空闲卡，每次启动前重新检查；私有目录、benchmark.lock、全局超时，不终止他人进程。所有失败和回退保留。

## 冻结参考身份

- DS header SHA256：f274c707897a0e2b716019817324588abe0fb136c495cc0a7b3c80a5830d76da。
- DS SIMD SHA256：c8fcd758cf864c68982af8679b14c0f71eddf95b6719f5439c6d083e92729ec3。
- DS host扩展SHA256：896ff49f1c71ab6cc5eb403699074469dbf7c314cd33c41a4f29807644f0aab4。
- 完整DS包归档SHA256：6597c1b56e817f73bb22814834e2f37895ce88b3c32785bd10ce9316f5e9eada。

`reference/deep_gemm`是冻结包的源码快照（不提交.so）；各变体以Git提交区分，不引入正式运行时多后端开关。DS文件保留MIT许可，完整归档及host扩展另存本地证据。

## 实际节点

- D0：完整冻结DS在本项目诊断目录重新JIT；Git统一LF换行，原始字节归档另存。不是已经迁移到我们的默认kernel。
- C1：在L12上替换M≤128、N256、K整除512的完整AIC供数/MMAD leaf。使用DS静态物理M128、独立A/SF完成事件及C API L1/L0地址；保留原AIV0 L1B协议、原GM ABI、FIX scale=64与旧数学规则。其他形状仍原实现。这个节点不是完整计算包已全部闭合的声明。
- 新核对：原工程权重SF的ColumnMajor MX布局实际是每N行连续K-SF；DS权重SF是pair-major。后续布局节点须同时修改离线权重准备和AIC读取，不仅是SwiGLU的激活scale布局。
- D1/D2/D3：在完整DS终点中，分别恢复原工程权重SF的N-major、激活SF的row-major、两者组合。三者都保持DS数学、任务顺序和完成协议，用DS冻结输出逐位差分；这先闭合混合包的数据ABI，不把它称为已完成完整OO/DO/OD/DD控制矩阵。
