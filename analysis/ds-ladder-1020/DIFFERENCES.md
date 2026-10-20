# 与冻结布局优化DS的差异账本

范围：EP8，H7168/I3072/E384/TopK6/shared0，FP8/FP4。DS身份固定为header `f274c707...`、SIMD `c8fcd758...`，不是随上游HEAD变化。累计节点见PLAN和结果摘要；以下“相同”只指已经核对的机制，不表示指令或所有数据布局相同。

## 这一轮按顺序迁移

1. Combine控制面：metadata由GM直接Scalar读。保留原记录ABI。
2. AIC输出：最后K时每个N半块完成MMAD后立刻发FIX；full仍由真实FIX完成产生。
3. GMM2 task方向：与GMM1同向；执行和完成通知同步修改。
4. 全局计算task顺序：DS lookahead替代small-linear场景的Wave交替，先保留原GMM2 phase排空。
5. 完成协议：两层共享epilogue序号、两份full/empty credit；AIV1等待GMM2真实FIX，AIC不再按phase排空。
6. AIV1存储：CV输入槽原地压缩，32行VF推进、stage末统一写出；不再每32行复用单份量化scratch。
7. AIV1计算：迁移DS寄存器驻留融合VF结构，但显式保留本工程的BF16舍入和指数scale合同。
8. Dispatch metadata：Scalar写GM替代UB写metadata及逐槽S/MTE3事件；ready前DDR barrier保证可见性。
9. Dispatch索引：Scalar直读已到达GM索引，替代小DMA和MTE2_S观察。
10. Dispatch owner：small-linear中物理core c处理c,c+cores,...，不再连续区间加Wave轮转；单M-group按实际行数发布ready。
11. Dispatch源行：使用DS互素步长，将拼接的source-rank行打散；metadata跟随真实源行，目标行仍按strided owner。

L4/L5必须配套理解：L4只对齐顺序，却保留旧完成发布，不能代表“完整DS调度”。有效但变慢的累计节点保留；编译失败/修复重试不计作新的优化。

## 尚未等同的部分：下一批优先级

|优先级|真实差异|为何需要闭合迁移，而非只抄一行|验证方式|
|---|---|---|---|
|P0|AIC计算后端：DS编译期MAD形状/C API指针布局；我们仍为CATLASS/TLA动态Tensor视图与适配层|MMAD算术指令之外的地址、scale装载和控制指令可能不同；“同样K/N”不等于同样发射成本。此前单项常量化无收益，不能把整个差距直接归因Scalar|同任务/同布局替换完整compute leaf；比timer-off和ISSUE外区间，再看task-block分核PMU，不能只比源代码行数|
|P0|小M的物理布局/槽间距：以BS512的M64为例，DS两N半块的L0C基址相隔64KiB；我们按实际M构造父layout，相隔32KiB|DS固定MAD_M128槽，再以实际M发MMAD；我们也按实际M定位第二半。工作量可以相同，物理位置并不相同；目前没有证据证明哪一个必然更好|单独对齐L0C基址/stride，保持实际M和MAC次数不变；验证FIX读取、unit flag、尾M以及复用依赖，不能为了对齐布局顺便多算M128|
|P0|GMM2输入scale：DS按M-block内SF pair-major存放；我们激活输出仍为row-major compact|必须同时迁移SwiGLU输出地址、GMM2 GM→L1读取和尾M布局；单改生产者会读错scale|逐位差分、奇数尾M/多M块/偏斜路由；对比MTE2字节量、活跃时间与实际端到端|
|P1|SF与A供数的提交顺序和完成边界|DS先预取weight-SF，A与SF用独立完成事件，在真正消费前等；我们在compute内装载SF/A并在K窗统一观察，copy leaf也把data/SF绑定。更换事件不一定消除关键路径|闭合替换copy/credit协议；确保每个buffer最后读取后才释放，比较首窗与连续窗，不为timer补同步|
|P1|通信游标：DS每次最多2行pull/push；我们部分入口仍按完整expert slice服务，M128可能每core4行|BS512每core2行时外观已经接近；BS1024/小BS余数分配可能仍不同。strided owner也可能使小M长期集中在低core|同一SO七BS＋random/skewed；不要只看BS512，更不能逐shape选分支最低值|
|P1|紧凑metadata ABI和头部计数/ready证书|DS源metadata为32bit packed；我们仍保留256B/row记录及既有prefix/count表。Scalar化没有消除记录stride或所有初始化|同步替换写方、读方、容量校验和分配；核对rank/token/topk编码边界。用timer看首次GMM1，而非拿小DMA次数预测收益|
|P2|Combine ready-prefix、尾部归约重叠与workspace/ring复用|DS的发送完成前缀和归约窗口属于同一协议；我们保留原跨卡收尾及weighted unpermute。必须检查真正的token readiness|端到端、尾部关键路径和真实通信输出；不能通过提前宣告完成制造“收益”|
|单列数值合同|DS的FP32激活amax/power-of-two scale、激活前后路由权重位置；我们保留BF16激活舍入、指数scale和最终加权|这些不是纯调度替换。即使都通过各自原有精度标准，也不保证逐位相等；不能把不再做的数值工作冒充零成本等价优化|建立独立的容差/数值基准后单列实验；当前性能-only＋逐位差分不能证明这类迁移正确|

上表是下一批实验优先级，不是已经验证每项都能更快的结论。改完必须继续保存有效负结果。只有移植完整实现才能达到“与DS相同代码”的端点；那个端点不能替代逐项归因，也不能预先保证在共享硬件上测出完全相同的μs。

## 阅读入口

- 调度/三角色协同：`experiments/megamoe/csrc/kernels/mega_moe/mega_moe_wave_a8w4.hpp`，`ProcessMoeExpertStages`、`ProcessSmallLinearTasks`、`ProcessSmallLinearAiv1`。
- lookahead与行映射：`common/small_linear_task_schedule.hpp`、`common/strided_row_owner.hpp`。
- AIC/权重Prologue：`catlass_gmm/block_mmad_a8w4.hpp`、`block_prologue_a8w4.hpp`；`mega_moe_grouped_matmul.hpp`负责handoff，未变成DS C API后端。
- 消费者：`epilogue/block_epilogue_activation_mx_quant.hpp::RunCvStep`、`activation/swiglu_mx_fused.hpp`；融合结构相似，数学合同并未改为DS。
- 通信：`stage/mega_moe_token_dispatch.hpp`、`mega_moe_gmm2_combine.hpp`；L10/L11仅用于actual M≤128的闭合路径。
- DS对应：冻结包`deep_gemm/include/deep_gemm/mega_moe.hpp`中的`aic_gemm_pipeline`、`aiv1_epilogue_comm`，SIMD文件中的`vf_swiglu_quant`；以保存的SHA核对，不用同名其他分支替代。

## 不混淆测量含义

正式表是完整通信融合kernel Duration、timer-off、warm10/采20取后10、逐轮rank-min均值；同时保存rank-max和逐rank样本。trace是单次rank0完整被动观察，不是8卡最慢rank完成时间；ISSUE不是硬件忙碌时间，角色的KERNEL_TIMING边界也未必相同。输入为FP8，不能把旧BF16输入场景的“额外40μs quant”直接套到本轮。

相邻行的累计耗时来自不同窗口；收益以本行直接父版或L0为准。无法把所有单项百分比直接相加，正收益也可能包含交互作用，需要反向撤销重要项再确认因果。CPU协议模型＋8卡逐位差分不是独立Golden，尚不能据此合入master。
