# N512 分阶段实验

分支基于 megamoe-urma 的 58a4ebc，不覆盖原分支。条件保持 `BS×EP×TopK < Experts×128`；当前扫描只有 EP4/BS1024。其它情况走原 M256 kernel。

## 第一笔：仅 GMM1 N512

- GMM1计算组M128，L1/L0C物理N512，内部MMAD保持N256/K128。K外层、两个N子块内层，同一L0A先后用于gate256、up256。
- 每组5个GMM1任务，SwiGLU每任务处理256列；GMM2保持原N256 block、14个任务和批次drain。为保证热点路由不超出GMM1 L0C，计算组统一按M128划分，所以热点专家的GMM2组数量可能改变，GMM2物理tile仍不变。
- L1 B双槽各128KiB（0/128KiB），A双槽各64KiB（256/320KiB），单份SA16KiB、SB64KiB（384/400KiB），共464KiB。Scale在本tile所有L0读取结束后才释放。固定K3584/1280均由4096-K scale窗口覆盖。
- L0A双槽各16KiB，L0B双槽各32KiB，L0C单个256KiB完整结果。普通NZ、显式M_FIX/FIX_M，无UnitFlag。A槽在两个N子块的MMAD都消费后才释放。
- B的L1 K仍为256。UB分4段K64×N512反量化，每段输出32KiB，原192KiB UB不扩大。优化目标是A重复读取和任务调度，权重总字节量不减少。
- 初始化权重及scale使用 `transform_weights_for_mega_moe(..., gate_up_block=256)`。benchmark自动设置；直接API调用需显式设置。运行时没有权重转换，旧gate128权重用于小路径会报错。
- 小路径独立入口 `mega_moe_urma_m128_kernel`，profiler已支持。GMM1 trace的tileN/columns为逻辑激活坐标（宽256），物理输出为512列。中间诊断按gate256/up256解析。

按要求未编译、未运行测试。拉取后需重新编译安装，benchmark运行参数不变。
