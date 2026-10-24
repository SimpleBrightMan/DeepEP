<!-- Copyright (c) 2026, Lu Lu; Modified by SimpleBright_Man 2026 -->
# MegaMoE A8W4 CATLASS GMM

此目录管理物理缓冲、供数、MMAD/FIX 和事件；专家地址、slice 与任务顺序由上层 stage/编排提供。
不要从函数返回推断异步写回已经完成。

## 阅读顺序

1. `a8w4_k_plan.hpp`：唯一资源图与 AIC/AIV0 共用的 K/N 供数计划。
2. `mega_moe_grouped_matmul.hpp`：GMM1 gate/up、GMM2 down 投影及资源生命周期适配。
3. `block_prologue_a8w4.hpp`：AIV0 FP4→FP8，独立管理 UB 输入/输出槽，向 L1 提供权重。
4. `block_mmad_a8w4.hpp`：AIC 读取 A/scale，消费 L1 B，执行 MMAD、FIX 写出。
5. `../common/gmm1_async_completion.hpp`：AIC/AIV1 的 issued/full/empty 协议。

`resource.hpp` 只是存储视图，不创建 TPipe，不重置其他阶段的事件。
`mega_moe_mx_matmul.hpp` 与 `block_mmad_mx.hpp` 保留通用 A8W8/A4W4 路径。

## 所有权与完成

- `UpdateProblem` 更新形状、地址和权重 layout，不重新分配存储。
- `Begin` 幂等启动；`End` 排空并归还事件，且恢复相关 MM layout。
- 异步 routed GMM1/GMM2 用 `HandoffTo` 转移缓冲游标与 credit，任一时刻只有一个对象 active。
- GMM2 Wave 结束先 `DrainOutput` 等 FIX 完成，才能发布各 slice 的 Combine ready；随后 handoff 回 GMM1。
- routed 阶段最终由当前 owner End；共享专家、量化与其他兼容路径保留既有局部生命周期。
- GMM1 的 GM 序号只表示 issued；AIV1 的 MTE2 等 FIX full 后读取，消费后通过 MTE3 返回 empty。
  不能用 Scalar 发出指令的时间替代真实完成，也不能为 timer 额外排空。
- AIV1 不参与权重 Prologue；其持久计算 context 休眠，不操作 AIV0 的缓冲/事件。

## 资源图

常量由 `A8W4BufferPlan` 统一定义，不在调用方重复硬编码：

| L1 区间 | 用途 |
| --- | --- |
| [0,192KiB) | A：3×64KiB |
| [192,448KiB) | B：4×64KiB |
| [448,480KiB) | scaleA：2×16KiB |
| [480,512KiB) | scaleB：2×16KiB |

AIV0 UB 输入为 [0,96KiB)，3×32KiB packed 槽；输出为 [96,224KiB)，2×64KiB FP8 槽。
输出按256字节向量交错。输入只等待 V 最后读取，输出只等待 MTE3 最后读取，不绑定两者的复用游标。
AIC/AIV0 共用 `MakeA8W4KPlan`：M≤128 时 KA/KB=512、B列块128，否则 KA/KB=256、B列块256。
不能只改一端的 K/N 循环顺序或槽位数量。

权重通道：AIV0 发布 BFull4～7，AIC 返回 BEmpty8～11。
GMM1 完成通道：AIC/AIV1 使用 full12/13、empty14/15 与 AIV1 目标编码，不与权重通道混用。

## 布局和上层接口

`CatlassGmm1` 管 gate/up 投影及输出偏移；`CatlassGmm2` 管 down 投影及行主序输出。
routed 权重的配对布局由 Python 离线准备，Host/JIT 和消费者必须选择同一 `FuseGateUp` 特化。
当前配对阈值是 `W4A8_PAIRED_GATE_UP_MAX_TOKENS=4096`，不是旧版的256；
它与共享专家的小 BS tile 策略阈值256是不同概念。
共享专家保持自己的 planar 布局与完成通知，不借用 routed ready 序号。

GMM1 stage 入口直接选择计算或兼容向量后处理，不再经过 `Gmm1ExecA8W4` 包装；
计算侧只接收 scheduler，不再携带无用的 Tensor WorkSet。
底层物理流水和同步顺序沿用性能基线，不因整理接口改变。

CPU 模型不能证明设备可见性。验收需覆盖尾 K/N/M、空/偏斜专家、跨 Wave/阶段 handoff、重复 launch、
共享专家和兼容路径，并分开报告输出等价、独立 Golden、性能与 timer 开销。

## 当前统一Wave调度

Linear专用task流、入口和重放代码已删除；AIC/AIV0统一走`ProcessGmmWaves`，AIV1走
`ProcessAiv1PriorityStages`。所有EP按每卡实际BS≥512开启GMM2滞后一波。
CV只决定GMM1结果通路，不再决定另一套调度；FIX完成、epilogue credit、输入ready及资源归还协议保留。
CV输出仍在已消费输入前缀内压缩，stage末尾统一写出，scale槽独立。
旧Linear消融代码及实测结果保存在Git历史和各独立实验分支中，不作为当前版本的新实测。
