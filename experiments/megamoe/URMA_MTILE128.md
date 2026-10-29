# M128 / GMM1 A3B3 / GMM2 A3B4

基于megamoe-urma的24a6f1c，新分支megamoe-urma-mtile128；不包含N512实验或此前A4实验。仅在 `BS×EP×TopK < Experts×128` 触发，当前网格为EP4/BS1024。保留gate128/up128融合及random-balanced。非小路径仍使用原M256 block。

## Tile和槽位

GMM1/GMM2物理L1 tile为128×256×256，L0 tile为128×256×128；相比基线只改M，N/K保持原值。保留原自适应A窗口，每槽保守64KiB。L0A双槽各16KiB，L0B双槽各32KiB，普通NZ L0C双槽各128KiB；不启用UnitFlag。

Scale双缓冲独立于数据槽，按实际完整K分配，不随A/B深度膨胀：

| 阶段 | A数据 | B数据 | A scale | B scale | L1合计 | 余量 |
|---|---:|---:|---:|---:|---:|---:|
| GMM1，K3584 | 3×64=192KiB | 3×64=192KiB | 2×14=28KiB | 2×28=56KiB | 468KiB | 44KiB |
| GMM2，K1280 | 3×64=192KiB | 4×64=256KiB | 2×5=10KiB | 2×10=20KiB | 478KiB | 34KiB |

L1 A放在0/64/128KiB，B从192KiB起每槽64KiB。Scale紧随B区；全部区域有静态容量约束。A预取三个窗口；L0A保留两个K128操作数的提前量。A/B槽位有独立事件。B跨核ready为6起、free为12起，上下文结束标志10保留。所有标志在上下文结束时收回。

计算组按M128划分，GMM1仍每组10个任务，GMM2每组14个；Combine/Reduce的依赖由相同planner生成。通信CV、PUT/quiet及Reduce流水不变。GMM1仍即时FIX完成后发布给激活，双L0C不代表GMM1跨tile排空已取消；GMM2沿用专家批次末尾drain。

独立小kernel入口mega_moe_urma_m128_kernel，profiler已适配。Python/native标记升级，需重新编译安装。运行参数不变，打印compute_pipeline=M128-G1A3B3-G2A3B4-C2。

仅源码检查；未编译、未运行测试。仍需精度与性能验证，尤其是双L0C散写及B多槽跨核同步。
