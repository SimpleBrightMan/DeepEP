<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# MegaMoE Prefill Beta

工程设计、文件/函数职责及同步维护约束见 [ARCHITECTURE.md](ARCHITECTURE.md)。当前仅保留
MTE + 非量化 Combine；W4A8 与 A8W8 都继续支持。URMA、量化 Combine、top-k weight prefetch
的不可达执行分支已清理，A8W8 MX 计算独立在 `stage/mega_moe_gmm{1,2}_a8w8.hpp`。
下方 10-16 整合数据属于 `18ada75b1` 及其受测设备代码，不能作为本次后续清理的验收结果。

## main：计算优化与完整 EP8 验收

main 已合入 GMM2 完成边界/wave 流水复用、GMM1 FIX 异步完成通知及 wave 流水复用。
`batch_launch.py` 与公开 API 默认均执行完整通信，不会自动进入 compute replay。
环境加载后可直接运行五个 BS，默认无共享专家并保存 timer：

```bash
python experiments/megamoe/scripts/batch_launch.py --ep-size 8 --bs-values 32,96,256,1024,4096
```

正式 profiler 性能样本仍为 timer-off，另采 timer 流水，避免改变旧main的比较口径。
旧main基线、重建要求、准确性与五个BS的对照步骤见 [OPTIMIZATION_TODO.md](OPTIMIZATION_TODO.md)。
历史3924μs是零通信本地计算，不是完整EP8 E2E；采样字段分析见 [PROFILER_COMPARISON.md](PROFILER_COMPARISON.md)。
独立的 `scripts/local_compute.py` 仍仅供零通信诊断；使用方法和历史验收见 [LOCAL_COMPUTE.md](LOCAL_COMPUTE.md)。

公开入口：`deep_ep_experimental.megamoe`。W4A8 共享专家按实际单卡 B 自动选择小 BS / Prefill 策略；路由专家保留 Prefill 调度。
构建和多机测试入口位于本目录的 `scripts/`；机器地址及环境配置由使用者提供。

## 0. 先读主流程，再读计算实现

本轮 main 候选基于性能分支 `codex/megamoe-passive-timer-1013@bef32bca3`，迁入
`codex/megamoe-readable@bf1a1b7d8` 的职责划分、命名和 Dispatch 规划，没有覆盖为旧 readable 的计算后端。
保留异步 FIX 完成、跨 GMM 资源 handoff、独立 UB 输入/输出槽、供数/缓存和被动 timer 策略；
不改变公开接口、tiling ABI、workspace 布局、flag 协议或数学契约，包含现成本机 EP8 启动入口。风格参考
[DeepGEMM-Ascend 的 MegaMoE](https://github.com/deepseek-ai/DeepGEMM-Ascend/blob/main/deep_gemm/include/deep_gemm/mega_moe.hpp)：
按角色组织流水、显式保留调度游标，在同步和复用边界说明约束；没有复制其实现或改变当前后端。

### GMM1 调用层次：哪些保留，哪些删除？

```text
ProcessGmmWaves                     决定 GMM1/GMM2 的 Wave 顺序
  → RunGmm1Wave / RunNextGmm1Slice   规划单专家 slice，推进游标
  → RunGmm1ActivationForExpert       绑定专家地址和同步区
  → RunGmm1A8W4                     派生一次配置和分核，直接按角色执行
      AIC/AIV0 → RunGmm1ComputeTiles → AIC tile循环 / AIV0权重转换循环
      兼容AIV1 → RunGmm1VectorEpilogueA8W4
```

原 `Gmm1ExecA8W4` 纯分派包装层已移除。AIC/AIV0 不再建立只为包装器准备的 A/B/scale/bias Tensor，
也不再各复制一份 persistent/local 分派。计算 helper 是资源生命周期边界，tile 循环是输入依赖及输出通知边界，
CATLASS 是物理缓冲/引擎边界，因此保留；不把所有工作塞成大函数，也不继续增加纯转发层。
现役非量化 AIV1 走独立优先级调度器，而非这里的兼容 epilogue。维护约定见本目录 `AGENTS.md`。

异步 routed GMM1/GMM2 用 `HandoffTo` 转移唯一资源 owner；GMM2 在发布 Combine ready 前
`DrainOutput` 等真正 FIX 完成，**不是每个 Wave 都销毁/重建流水**。最终 owner 在 routed 阶段结束时统一 End。

### 2026-10-16 整合验收

设备代码提交 `0f90f5333` 与 `bef32bca3` 在同一 Ascend950DT 八卡节点交替执行 A/B/B/A。
H7168/I3072/E384/TopK6、FP8/FP4、无共享专家、balanced；每组预热10次、正式20次，
统计后10轮的逐轮 rank-min 均值，另保留 rank-max。全部是完整通信 kernel Duration、timer-off。

| BS | 性能基线 μs | 整理后 μs | 延迟变化（正值变慢） |
| --- | --- | --- | --- |
| 32 | 631.848 | 632.901 | +0.17% |
| 96 | 660.025 | 661.108 | +0.16% |
| 256 | 722.746 | 721.634 | −0.15% |
| 1024 | 1164.709 | 1163.576 | −0.10% |
| 4096 | 4130.505 | 4129.038 | −0.04% |

结论是本轮重构基本保持该性能基线，不宣称新的加速或所有平台/EP上的“最快”。
四组全部 rank/shape 的完整 BF16 输出 SHA256 一致；随机路由 BS33/257/4104 和共享专家1个的
BS32/257 也完成基线/候选全输出等价回归。等价回归不是独立 Golden，未继承历史精度 PASS。
Linux CPU 完整套件312项通过，含实际生产函数的调度、角色、供数、资源交接及被动 timer 桩；
timer 协议校验通过。本次没有重测 timer-on/off 开销、EP32/128、A8W8 或量化 Combine 的设备用例。

复现入口、wheel/SDK/源码指纹、逐轮数据与原始 CSV 归档位于
[`analysis/main-readable-1016`](../../analysis/main-readable-1016/)。该目录的 `benchmark_entry.py`
只在 profiler 采集结束后复制输出，不在计时区间插入 D2H。最终归档提交只补文档、测试和证据，
设备生产代码与上述受测提交一致。

### 阅读顺序

以下路径均相对本目录；在 VSCode 搜索表内函数名可直接定位。

| 顺序 | 文件 / 函数 | 先回答的问题 |
| --- | --- | --- |
| 1 | [api.py](python/deep_ep_experimental/megamoe/api.py)：`SymmBuffer._prepare`、`_run` | 哪些资源长期持有，哪些工作每次 forward 执行？ |
| 2 | [forward.cpp](csrc/forward.cpp)：`MakeTiling`、`Plan::Plan`、`Plan::forward` | H/I/BS 如何变成 tiling、workspace 和 kernel 参数？ |
| 3 | [mega_moe.cpp](csrc/kernels/mega_moe/mega_moe.cpp)：`mega_moe_kernel_do` | 实际实例化了哪些模板？现役 W4 路径是非量化 Combine。 |
| 4 | [mega_moe_arch35.hpp](csrc/kernels/mega_moe/mega_moe_arch35.hpp)：`ProcessWave` | 输入、专家计算、输出的公共阶段怎样连接？ |
| 5 | [mega_moe_shared_policy.hpp](csrc/kernels/mega_moe/common/mega_moe_shared_policy.hpp)：`SelectInputPreparationMode` | 为什么选这个输入协议？不要把策略当成 ready flag。 |
| 6 | [mega_moe_wave_a8w4.hpp](csrc/kernels/mega_moe/mega_moe_wave_a8w4.hpp)：`ProcessMoeExpertStages`、`ProcessGmmWaves`、`ProcessAiv1PriorityStages` | AIC/AIV0 与 AIV1 分别怎样推进？ |
| 7 | [mega_moe_wave_schedule.hpp](csrc/kernels/mega_moe/common/mega_moe_wave_schedule.hpp)：`ExpertSlice`、`TakeNextExpertSliceInWave`、`GetExpertSliceInWave` | 怎样固定本次 slice、推进下一位置，再由 GMM2 重放同一范围？ |
| 8 | [mega_moe_gmm1_activation.hpp](csrc/kernels/mega_moe/stage/mega_moe_gmm1_activation.hpp)：`WaitForGmm1InputReady`、`RunGmm1A8W4` | Dispatch、Prologue、MMAD、SwiGLU 的数据依赖是什么？ |
| 9 | [block_prologue_a8w4.hpp](csrc/kernels/mega_moe/catlass_gmm/block_prologue_a8w4.hpp) 与 [mega_moe_gmm2_combine.hpp](csrc/kernels/mega_moe/stage/mega_moe_gmm2_combine.hpp) | 如何用分块生产/消费和完成计数保护数据？ |

### 公共入口与输入模式

`ProcessWave` 按以下顺序组织调用（不同角色可以并行，不能把它当成全卡串行时间线）：

```text
ProcessInputPreparationStage → WaitForLocalInput
→ 可选 ProcessSharedExpertGmm1 → ProcessInputRouteHandshake
→ 派生类 ProcessMoeExpertStages → 可选 ProcessSharedExpertTail
→ ProcessOutputStage
```

`inputMode_` 在 Init 时确定；原来的 `decodeInputReady_` 和 `routeAiv1Only` 不再保存为独立布尔值。
BS 指本卡实际输入行数，而非某专家接收行数或 buffer 容量。

| 输入模式 | 条件 | 路由与同步 |
| --- | --- | --- |
| `CountEpoch` | W4A8、BS≤256，不要求有共享专家 | 双 AIV 量化/打包后本卡同步，再发送路由；count/epoch 检查替代输入跨卡握手 |
| `Aiv1RouteOverlap` | W4A8、BS>256、有共享专家 | 双 AIV 准备输入；公共屏障后 AIV1 做路由/握手，与 AIC/AIV0 共享 GMM1 重叠 |
| `AllAivRankBarrier` | 其余路径 | 双 AIV 在输入准备中做路由，随后做输入跨卡握手 |

`SharedExpertPolicy` 只负责共享 GEMM 的分块。`useSmallBatchTiles` 表示共享专家使用 N192/N224；
与输入模式共用小 BS 条件，但不再夹带通信职责。Host 的路由分核与设备端使用同一个输入模式选择函数。

### Wave 与三种角色

M-group 是同一专家最多 256 行；尾片也占一个组，不与下一专家拼接。
Slice 不跨专家，Wave 可包含多个 slice。`ExpertWaveRange` 将起点、终点和最后非空专家放在一起，
用于当前或滞后一轮的 GMM2；它不表示计算已完成。

- AIC：`ProcessGmmWaves` 中执行 GMM1/GMM2。
- AIV0：与 AIC 同序执行权重 Prologue，以 L1 ready/free 事件协作。
- 非量化 Combine 的 AIV1：进入 `ProcessAiv1PriorityStages` 后返回，不执行下面的计算侧 Wave 循环。
  优先级保持为路由 SwiGLU、Dispatch、共享 SwiGLU、Combine；未就绪任务只探测，不阻塞整个调度器。
  一次搬运不能被中途抢占，每完成一个 tile/slice 才重新检查优先级。
- 量化 Combine 兼容路径：保留首 Wave 预取及下一 Wave Dispatch / 当前 Wave Activation 的原顺序，
  不应把这段代码误当成当前非量化 AIV1 路径。

BS<4096 时计算顺序为 `G1(W0), G2(W0), G1(W1), G2(W1)...`；BS≥4096 时为
`G1(W0), G1(W1), G2(W0), G1(W2), G2(W1)...`，末尾补齐最后一个 GMM2。
GMM1/GMM2 各自保留跨专家分核游标，GMM2 物理分核方向反转。AIV1 的进度不由这些调用顺序代替。

### 专家状态、当前 slice、下一位置

计算调用将完整专家元数据与本次行范围分开：

| 对象 | 含义 | 谁修改 |
| --- | --- | --- |
| `ExpertContext` | 当前专家编号、专家起始 GM 行、完整专家 M 以及 GMM1 的 N/K 约定 | `PrepareGmmExpertContext` 按专家顺序更新；`gmm1ExpertContext/gmm2ExpertContext` 分别保存 |
| `ExpertSlice` | 本次 `expertIdx / rowStartInExpert / rowCount / globalRowStart` | 规划后保持不变，传给 GMM1/GMM2 |
| `ExpertTokenPosition` | 下一次待规划的专家、专家内行和本卡紧凑 GM 行 | `TakeNextExpertSliceInWave` 内调用原推进算法更新 |
| `ExpertWaveRange` | 一整个 Wave 的起止位置与最后非空专家 | GMM1 记录，GMM2 通过 `GetExpertSliceInWave` 重放 |

`globalRowStart = expertContext.globalTokenStartIndex + rowStartInExpert`，是本卡紧凑 GM 行号，
不是源卡 token ID。切片可以让下一位置进入下一个专家，但本次 slice 和专家状态仍属于旧专家。
`ExpertContext.problemShape.M` 始终是专家总行数；执行时复制形状并设置 `M=slice.rowCount`。
GMM2 由相同 N/K 约定构造实际计算维度：`N=H`、`K=I`。count 表在阶段入口只等待一次，
不再把 `expertCountTableReady` 放进专家元数据；这不代替 Dispatch、激活和 GMM2 的数据 ready flag。
`gmm1Cursor` 表示下一待规划位置，`gmm1ExpertContext` 表示当前计算专家，`slice` 表示本次固定范围。

例如 EP64、每卡 BS4096、384 专家、TopK6、H7168/I3072、32 AIC、每 Wave 六个 M-group，
完全均衡时每个本地专家4096行。W2 的两个 slice 如下：

| slice | `expertIdx` | `rowStartInExpert` | `rowCount` | `globalRowStart` | 规划后的下一位置（专家/专家内行/GM行） |
| --- | --- | --- | --- | --- | --- |
| E0 尾段 | 0 | 3072 | 1024 | 3072 | 1 / 0 / 4096 |
| E1 首段 | 1 | 0 | 512 | 4096 | 1 / 512 / 4608 |

GMM1 用 `TakeNextExpertSliceInWave` 产生这两个 slice；滞后一 Wave 的 GMM2 用保存的 W2 边界
恢复同一范围。分核滚动、通信 owner、ready 通知和 timer 范围沿用原协议。

### 计算循环与支持边界

`ProcessGmmWaves` 只组织 GMM1/GMM2 Wave 的先后关系。`RunGmm1Wave` 推进一个 Wave，
`RunNextGmm1Slice` 推进其中一个专家 slice；是否还有任务由游标和 M-group 容量直接判断，
不再维护 `currentWaveNeedsGmm1/nextWaveNeedsDispatch` 的重复布尔状态。
是否还有待执行 GMM2 由 GMM1/GMM2 完成的 Wave 数推导，保留 `gmm2LagActive` 表达 BS 调度策略。

量化 Combine 的首 Wave 预取、交错激活和专家级量化回传入口已删除。
W4A8 的 AIV1 始终使用独立优先级调度器；输入量化与 SwiGLU 后的 MX 量化不受影响。
A8W8 的独立实现继续保留，不把它作为 W4 运行时开关。

### Dispatch：范围规划与实际搬运分开阅读

先读 [mega_moe_token_dispatch.hpp](csrc/kernels/mega_moe/stage/mega_moe_token_dispatch.hpp)，
再读 [mega_moe_peer_rows.hpp](csrc/kernels/mega_moe/common/mega_moe_peer_rows.hpp) 的配额算法。

```text
W4 单专家 slice → DispatchExpertSlice
通用跨专家范围 → DispatchTokenRange（整个范围只分配一次 owner，再投影到专家）
                  ↓
DispatchOwnedExpertSlice → rank 顺序布局 / peer 均衡布局
                  ↓
PeerTransfer → AppendPeerRouteSegment → ConsumeRouteBatch → IssuePeerTokenCopies
                  ↓
FinishDispatchOwnedRows：消费剩余 batch → DrainDispatchCopyBuffers → PublishGmm1TileReady
```

`PeerTransfer` 明确记录来源卡、源 compact route 起点、行数、目标 GM 行。
均衡布局的 `PeerTransferCursor.Next` 只规划这些段，不搬 token、不发布 flag；
保留规则配额快路径和 peer 耗尽时的余数分配顺序。
`AppendPeerRouteSegment` 批量读取 compact 索引，`ConsumeRouteBatch` 等索引可读后发起 token 搬运，
token 环形槽跨 peer 复用。两种布局共享同一个收尾边界：末行写完、相关 MTE 事件回收后才发布 owner 的行贡献。
没有增加屏障，也没有改成按 peer 发布 ready；默认策略、目标行排列和 GMM1 ready 计数不变。

### 最重要的数据依赖

| 生产者 → 消费者 | 数据 / 就绪条件 |
| --- | --- |
| 量化/打包 → 对端 Dispatch | 本地 token/scale 完成后才发布 count；接收方验证本轮 epoch |
| Dispatch → GMM1 | `flagDispatchToGmm1Ptr`：本 M-group 已落地的实际行数 |
| AIV0 Prologue → AIC | `AIV_SYNC_AIC_FLAG` 发布 L1 B ready；`AIC_SYNC_AIV_FLAG` 允许复用槽位 |
| GMM1 → routed SwiGLU | 异步路径 `flagGmmToEpiloguePtr` 是逐核 issued 序号；MTE2 必须再等真实 FIX full 后读取 |
| routed SwiGLU → GMM2 | `flagActivationToGmm2Ptr`：对应输入行块的激活完成计数 |
| GMM2 → Combine | 对应 expert/slice 的完成计数，必须覆盖实际生产者 |
| 共享 GMM1 → 共享激活 → 共享 GMM2 | 独立的 `flagSharedGmm1DonePtr` / `flagSharedActivationDonePtr`，不借用 routed 序号 |
| Combine → Unpermute | 保留输出跨卡握手，确保源卡读取前远端结果已返回 |

Prologue 转换的是权重，不是 Dispatch 输入；AIC 同时需要两者就绪。
Prologue 按 K 块将 FP4 转为 FP8 并送入 L1，不必等整个专家的全部权重转换完才启动 MMAD。
不要删除看似重复的 drain/barrier：Dispatch、SwiGLU、Combine 会顺序复用 UB，flag 发布必须晚于数据写完。

### 重构验证边界

CPU 测试直接编译生产代码中的策略和 Wave 辅助函数，比较旧版推进公式，覆盖空专家、255/256/257 尾片、
跨专家/跨 Wave、大于 32 位的 GM 行号和 2000 组确定性随机分布。
另一个测试提取当前 `ProcessWave` 及三个阶段函数的实际 C++ 函数体，使用记录型底层桩执行，
核对 108 种角色/BS/共享专家组合的阶段及屏障调用顺序：

```bash
python -m pytest experiments/megamoe/tests/test_device_schedule.py -q
python -m pytest experiments/megamoe/tests/test_dispatch.py -q
```

Dispatch 测试把当前生产规划与重构前的冻结实现逐段对照，覆盖 AIC/AIV、两种布局、
快路径开关、EP2～128、空/不均衡专家、容量裁剪与单专家直接入口。
另以 DMA 桩执行实际环形流水及收尾函数，检查 1～6 槽、跨 peer/batch 复用、末行补写、排空后发布 ready。
这些是 CPU 行为检查，不替代 NPU 编译、真实异步 DMA、精度和性能验证。
需要 C++17 编译器；可用 `CXX` 指定，没有编译器时测试会明确 skip。
这不是 NPU 仿真，不能验证 DMA 可见性、死锁或性能。设备验收请在原版与重构版分别重新构建，
保持同输入、路由、计时方式，至少检查：

- W4A8、共享专家 0/1，BS=96/255/256/257/4095/4096；EP8 后再扩到 EP64。
- balanced/random/skewed 的精度与专家计数，重复调用及 `256→257→256` 的 epoch/模式切换。
- timer 关闭的性能、独立采集的阶段顺序和等待；不以 CPU 测试通过声称性能不变。
- A8W8 公共阶段兼容性；公开 benchmark 固定 W4A8，A8W8 需独立设备用例。

## 1. 输入、输出与公开接口

定义：`B` 为每卡 token 数，`H` 为隐藏维度，`I` 为 SwiGLU 后宽度，`K` 为 top-k，
`E` 为全局路由专家数，`L=E/EP` 为每卡路由专家数，`S` 为每卡共享专家数。

```python
# 公开调用签名；原位写入 y，返回 None。
def fp8_fp4_mega_moe(
    y, l1_weights, l2_weights, sym_buffer,
    shared_l1_weights=None, shared_l2_weights=None,
    cumulative_local_expert_recv_stats=None,
    recipe=(1, 1, 32), activation="swiglu",
    activation_clamp=None, fast_math=True,
    *, x=None, timer=None,
) -> None: ...
```

| 参数/输入 | 类型、形状与用途 |
| --- | --- |
| `y` | 调用方预分配的 BF16 `[B,H]` 输出，不能与输入重叠 |
| `sym_buffer.x / x_sf` | 默认输入：FP8 E4M3 `[B,H]` + E8M0 `[B,H/32]`，调用前填入 buffer 的前 B 行 |
| `x` | 可选 BF16 `[B,H]`，覆盖 buffer 中的 FP8 输入；内部量化为 FP8，**不是 BF16 GEMM** |
| `sym_buffer.topk_idx` | int32 `[B,K]`，全局专家 ID，同 token 不重复且不能为 -1 |
| `sym_buffer.topk_weights` | FP32 `[B,K]` 路由权重；BF16/FP8 输入均从 buffer 读取路由 |
| `l1_weights / l2_weights` | 两对 `(weight, scale)`，当前测试使用 FP4 E2M1 权重 + E8M0 scale |
| `shared_l1_weights / shared_l2_weights` | 共享专家的两对权重，必须同时传入，数量与 buffer 的 S 一致；默认不使用 |
| `cumulative_local_expert_recv_stats` | 可选 int32 `[L]`，在原值上累加专家接收 token 数 |
| `timer` | 可选 `buffer.allocate_timer()`；仅在单独诊断调用中传入 |

`recipe/activation/activation_clamp/fast_math` 当前仅支持签名中的默认值。所有张量在 buffer 所在 NPU 上。
FP8 输入仍需打包到通信区，因此流水中的 `AIV_QUANT` 此时表示打包，不是重新量化。

权重在测试/推理前转换一次，转换过程不计入性能测量：

| 原始 ND 权重 | packed FP4（uint8，两个值/字节） | E8M0 scale |
| --- | --- | --- |
| GMM1 | `[L,2I,H/2]`，N 方向先 gate 后 up | `[L,2I,H/64,2]` |
| GMM2 | `[L,H,I/2]` | `[L,H,I/64,2]` |

调用 `w1, w2 = transform_weights_for_mega_moe(raw_w1, raw_w2)` 得到 Ascend NZ 权重。FP4 routed GMM1 一次准备两套 weight/scale：实际 B≤256 使用 `[gate128, up128]` 融合布局，B>256 使用 `[gate | up]` 配对布局。转换不进入计时区间，但额外保留一份 GMM1 权重和 scale；GMM2 不复制。
请完整保留返回的 `w1`，不要解包后重组、原地修改或自行搬运其张量；权重更新或设备变化时从原始 ND 重新转换。共享专家将首维 L 换成 S，并传入 `shared=True`，使用 planar 权重布局。
不能直接传入已经做过 CUDA 专用重排的权重。

buffer 的构造参数如下，模型参数含义与测试 JSON 相同：

```python
from deep_ep_experimental.megamoe import (
    StoreGroup, SymmBuffer, fp8_fp4_mega_moe, transform_weights_for_mega_moe,
)

# torchrun 启动后先 torch.npu.set_device(local_rank)。
group = StoreGroup.from_env()  # 主机协调用 TCPStore，算子通信用 SHMEM + MTE
buffer = SymmBuffer(
    group, num_experts=384, num_max_tokens_per_rank=4096, num_topk=6,
    hidden=7168, intermediate_hidden=3072,
    num_shared_experts=0, mma_type="fp8xfp4", activation="swiglu",
    ranks_per_node=8, timeout=120,
)
```

准备输入后依次调用 `buffer.validate_inputs(B)`（BF16 时额外传入 `x=x_bf16`）、`buffer.prepare(B)` 和算子。
执行成功后，所有 rank 按相同顺序调用 `buffer.destroy()`、`group.close()`；
异常处理及完整可运行调用示例见 [example.py](python/deep_ep_experimental/megamoe/example.py)。

约束：EP=2～128，E 可整除 EP 且每卡最多 128 个路由专家；S=0～8；B=1～65536；
H=1024～8192 且为64的倍数，I=256～4096 且为128的倍数，1≤K≤min(32,E)。
各 rank 必须同序、同 B 调用；仅推理，不支持反向、空/不等长 B 或图捕获。
`mma_type="fp8xfp8"` 是保留入口，不支持共享专家/timer；**下文 benchmark 固定测试 W4A8，不提供权重 dtype 开关**。

兼容入口 `mega_moe(x, ids, weights, w1, w2, buffer, ...)` 返回 `(y, per_call_counts)`。
接口签名参照 DeepGEMM `78b6900`。本实现使用 int32 路由 ID、E8M0 输入 scale，
输入 buffer 是普通 NPU 存储，需要打包进入通信区。路由权重在 GMM2 后施加，
与在中间量化前施加权重的实现不保证数值等价；CPU 参考遵循本实现的计算顺序。

### 共享专家策略

当 W4A8 的 S>0 时，由实际 B 选择策略，容量 `num_max_tokens_per_rank` 不参与判断：

| 项目 | B<=256 | B>256 |
| --- | --- | --- |
| 输入 | 双 AIV Quant → AIV 同步 → Route → reset → 共享输入准备 | 双 AIV Quant → reset → 共享输入准备 |
| 本卡全核屏障后 | AIC/AIV0 共享 GMM1，AIV1 count/prefix/Dispatch | AIC/AIV0 共享 GMM1，AIV1 Route/子组及跨卡同步/count/Dispatch |
| 共享 GMM1 | 完整 gate/up 输出 N192 独立任务 | N256 gate/up 配对任务 |
| 共享 GMM2 | N224 | N256 |

两条路径均使用 CATLASS 后端、M256 和 N256 共享 SwiGLU。路由专家在 B≤256 使用逻辑 N128/物理 N256 融合；B>256 使用 gate256/up256 配对任务。
Token Combine 按实际单卡 B 选择：B≤256 使用六槽，B>256 使用两槽，与是否开启共享专家无关。
小 BS 使用逐槽 epoch/count 就绪协议，每个物理 AIV 本地推进一次 epoch，
输出握手再推进一次；大 BS 使用输入/输出跨卡握手。共享 GMM1 后无需全核屏障。
共享 SwiGLU 在 AIV1 上优先于 Combine，共享 GMM2 按本核路由完成及共享激活就绪推进。
S=0 的 W4A8 小 BS 也使用上述量化完成/count 就绪协议；W8A8 使用输入/输出跨卡握手。
tiling ABI 为 224 字节。
CATLASS GMM 的布局和同步机制见 [GMM 说明](csrc/kernels/mega_moe/catlass_gmm/README.md)。

所有 rank 必须按相同顺序、相同 B 调用，切换 B 时在计时前集体调用 `prepare(B)`。
已有 `prepare` 一致性检查用于防止各 rank 选择不同握手策略。共享计算五类 timer 保留，
不添加 count 诊断事件；N192/N224 的 tile 元数据按实际 N 分块记录，JSON 格式不变。

验证边界建议覆盖 B=255/256/257/512、S=0/1/2，以及同一 buffer 下反复切换
`256 → 257 → 256` 和 epoch 回绕。CPU 协议测试不能替代 CANN 编译和 NPU 精度/性能验证。

## 2. 编译安装

W4A8 在 B≤256 时使用较小 wave、rail 路由中转和打包 count。
rail 按每8个 rank 一组，EP≤8 或完整8卡分组启用，其余 EP 回退直发路由。
B>256 保留 Prefill 输入协议，所有路径保留输出跨卡同步。

要求 Linux、Python ≥3.11、Ascend950、CANN 9.2.0 及匹配的 Torch/torch_npu；框架环境由使用者准备。

```bash
git clone https://gitcode.com/Ascend/DeepEP.git
cd DeepEP
source /path/to/cann/set_env.sh
bash experiments/megamoe/scripts/build.sh
python -m pip show ascend-deepep
```

在匹配的 Torch/torch_npu 环境中执行。实验构建入口默认开启 MegaMoE 和 timer，
自动初始化固定版本 SHMEM/CATLASS 子模块、校验或构建 SHMEM，再构建安装 wheel。
`PYTHON=/path/to/python` 可指定解释器；`DEEPEP_MEGAMOE_TIMER=OFF` 关闭设备打点。
已激活 Conda 时，入口会在 Python 启动前优先使用该环境的运行库。
已有 SDK 可用 `SHMEM_ROOT`、`CATLASS_ROOT` 指定；已有子模块须保持干净且版本与[依赖锁文件](../../dependencies.lock.json)一致。
需要重建仓内 SHMEM 时，停止使用该 SDK 的作业后传 `--rebuild-shmem`。
wheel 位于 `dist/build-*/`；运行前按构建输出设置 SHMEM 的 `LD_LIBRARY_PATH`。
基础包的 `pip wheel` 默认关闭 MegaMoE，详见[构建文档](../../docs/build.md)。

## 3. 单机与多机测试

### 本机 EP8（一键性能与流水）

在 Linux 八卡机器上激活匹配的 Torch/torch_npu 环境并加载 CANN，进入本仓库执行：

```bash
python experiments/megamoe/scripts/batch_launch.py --ep-size 8 --bs-values 96
```

默认使用 `deepseek_v4_pro`、FP8 输入/FP4 权重、balanced 路由、性能模式，
自动构建安装、运行并保存/解析 rank0～7 的 timer；无需 hosts 文件、SSH 或
`CONDA_BASE/CONDA_ENV/ASCEND_ENV`，直接继承当前 Python/CANN 环境。
需要预加载 Conda libstdc++ 的机器可在命令前加
`LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"`，该设置会传给执行阶段。
入口自动配置仓内 SHMEM 库路径、通信端口，并从 `/tmp` 启动，避免源码遮蔽已安装包。
需要八张可用 NPU；默认占用本机 29654/26733 端口，可分别用 `--master-port/--shmem-port` 更改。

```bash
# 只预览，不编译、不运行。
python experiments/megamoe/scripts/batch_launch.py --ep-size 8 --bs-values 96 --plan
# 已构建过：复用与已安装包身份匹配的唯一仓内 wheel，仍校验并安装。
python experiments/megamoe/scripts/batch_launch.py --ep-size 8 --bs-values 96 --skip-build
# 精度：自动先准备 CPU Golden。
python experiments/megamoe/scripts/batch_launch.py --ep-size 8 --bs-values 96 --mode accuracy
```

不指定 `--bs-values` 时使用配置的 96、4096；EP8 的 384 个路由专家为每卡48个，
建议先跑 BS96 确认内存容量。单机复用下面的结果格式：
`benchmark-runs/ep8_*/bs_96/bs_96/timer/` 下保存原始 bin、流水 JSON 和耗时报告，
性能采样关闭 timer，流水另行采集。默认流水隐藏 WAIT；需要查看时可按下方离线命令追加
`--include-wait` 重新解析。`--no-timer`、`--no-parse-timer` 和共享专家配置也适用于 EP8。
执行前保持已跟踪源码干净；更新算子或包内 Python 后不要用 `--skip-build` 复用旧包。

### 配置与启动

所有节点需要相同绝对路径的共享 checkout、匹配的 Conda/CANN 环境、SSH 访问和每机8张空闲 NPU。
在仓库外创建机器配置，通过 `--hosts-file` 显式传入。以下地址仅用于文档示例，需替换为已分配的节点：

```json
{
  "hosts": ["192.0.2.1", "192.0.2.2", "192.0.2.3", "192.0.2.4"],
  "master": "192.0.2.1"
}
```

EP32/64/128 分别需要4/8/16个不同节点。`master` 可省略，默认使用列表首节点。
主节点排在 rank0～7，其余节点保持列表顺序。仓库不提供实际机器池。
在一台控制机设置远端环境后启动：

```bash
export CONDA_BASE=/path/to/conda
export CONDA_ENV=your_env
export ASCEND_ENV=/path/to/cann/set_env.sh
export HOSTS_FILE=/path/outside/checkout/hosts.json
# REMOTE_USER 默认为控制机当前用户，可按需覆盖；默认使用 SSH 密钥。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --ep-size 32 --plan
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --ep-size 32
```

入口编译一次、逐节点安装同一 wheel，再运行测试及解析 timer；不会更新 Git 分支。
`CONDA_BASE`、`CONDA_ENV`、`ASCEND_ENV` 必须显式设置，路径对应每台远端机器。
密码登录可设置 `SSH_AUTH=password` 并安装 sshpass，密码通过交互输入或 `SSHPASS` 提供。
实际机器配置、凭据和输出报告不要加入版本控制。

### 常用参数

| 参数 | 默认 / 作用 |
| --- | --- |
| `--mode perf\|accuracy` | 默认 perf 执行性能测试；accuracy 检查全量输出和专家计数 |
| `--input fp8\|bf16` | 默认 fp8；只改变输入类型，权重仍为 FP4 |
| `--config NAME或路径` | 默认 `deepseek_v4_pro`；可传模型 JSON 路径 |
| `--bs-values 96,4096` | 覆盖 JSON 中 BS 列表；每个 BS 单独启动 |
| `--routing balanced\|random\|skewed` | 默认 balanced；random/skewed 用于精度压力测试 |
| `--no-timer` | 关闭打点和解析；默认自动保存、解析主节点 rank0～7 |
| `--no-parse-timer` | 跳过自动流水解析，保留原始 timer 和耗时拆分 |
| `--ep-size 8\|32\|64\|128` | 默认64；8为本机，其余分别为4/8/16台八卡机器 |
| `--hosts-file 路径 / --master-ip IP` | 必填机器列表 JSON / 覆盖列表内的主节点 |
| `--master-port / --shmem-port` | 默认29654 / 26733，必须不同且未占用 |
| `--skip-build / --wheel 路径` | 复用已安装身份匹配的唯一 wheel / 指定共享 wheel；不再编译，但仍向各机安装 |
| `--prepare-only / --plan` | 只部署并检查 / 只预览 |
| `--timeout 秒` | 每阶段、每 BS 默认7200秒 |

`balanced` 路由按全局 `[token, top-k]` 顺序轮转目标 rank，再轮转卡内 expert。
精度使用相同路由生成 CPU Golden。修改算子或包内 Python 后，需要重新构建安装，
首次运行时省略 `--skip-build`；修改文档或启动脚本无需重编算子。

以下命令使用8节点机器文件，执行 EP64：

```bash
# 全量精度测试，关闭设备打点。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --mode accuracy --bs-values 96,4096 --no-timer

# BF16 输入，性能 + 自动流水解析。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --input bf16 --bs-values 96

# EP128：将 HOSTS_FILE 改为包含16个节点的配置。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --ep-size 128
```

### 模型和共享专家配置

默认模型：H=7168、I=3072（gate+up 总宽6144）、全局384路由专家、TopK=6、S=0。
EP32/64/128 分别每卡12/6/3个路由专家；共享专家不计入384，`num_shared_experts` 表示**每卡**数量。

已提供两个配置，只有 `num_shared_experts` 不同，其余模型、BS、seed 和轮次一致：

- [`deepseek_v4_pro`](python/deep_ep_experimental/megamoe/configs/deepseek_v4_pro.json)：无共享专家，默认配置。
- [`deepseek_v4_pro_shared1`](python/deep_ep_experimental/megamoe/configs/deepseek_v4_pro_shared1.json)：每卡额外1个共享专家，不减少原有路由专家。

```bash
# 每卡1个共享专家，全量精度检查 BS96/4096。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --config deepseek_v4_pro_shared1 --mode accuracy --no-timer

# 使用相同配置执行性能测试，并自动采集、解析主节点8卡流水。
python experiments/megamoe/scripts/batch_launch.py --hosts-file "$HOSTS_FILE" --config deepseek_v4_pro_shared1
```

需要其他数量可复制 JSON、修改 `num_shared_experts` 后使用 `--config /path/to/case.json`。
脚本自动生成对应共享权重、传给公开接口，CPU Golden 也包含共享专家；
**没有 `--shared-experts` 开关**。warmup/iters/seed 同样在 JSON 中修改。

### 测试结果

精度测试以相同输入、权重和路由生成的 CPU Golden 为参考，生成输出和专家计数的校验报告。
CPU Golden 缓存位于 `benchmark-golden/`，相同配置可复用。
性能测试根据关闭 timer 时 profiler 采集的算子 kernel Duration，生成耗时统计报告。
流水分析根据独立采集的设备 timer 数据，生成流水 JSON 和阶段耗时报告，默认解析主节点 rank0～7。

终端输出各 BS 的 `[PERF]` 耗时或 `[ACCURACY SUMMARY]` 精度摘要，结果保存至：
`benchmark-runs/ep<EP>_<时间>_<编号>/`。

| 目录内文件 | 内容 |
| --- | --- |
| `summary.txt` | 各 BS 的性能/精度汇总 |
| `logs/node_<n>_<stage>*.log` | 各节点的构建、Golden、运行、解析日志 |
| `bs_<B>/rank_0/bs_<B>/accuracy.json` | 全部 rank 的精度报告 |
| `bs_<B>/rank_<r>/bs_<B>/profiling.json` | 性能汇总；同目录保留 `kernel_samples.json` 和 `profiler/` |
| `bs_<B>/bs_<B>/timer/` | 主节点8卡原始 `.bin`、`rank_<r>_npu_time.json`、CSV 和 `timer_breakdown.txt/json` |

原始 timer 数据也可离线解析：

```bash
python -m deep_ep_experimental.megamoe.timer /实际结果目录/bs_96/bs_96/timer --ranks 0-7
```

所有 BS 执行成功后，终端输出 `[DONE] all nodes completed`；执行或报告校验失败时返回非零退出码。

## 4. 测试与验证范围

```bash
python -m pip install -r experiments/megamoe/tests/requirements.txt
python -m pytest experiments/megamoe/tests -q
```

保留公开 API、数值参考、Golden 缓存、打包、主机协调与计时解析测试。
CPU 检查不启动设备 kernel，也不作为设备验证结果。Ascend950 设备验证记录按以下维度区分：

- MegaMoE 开启/关闭、timer 开启/关闭的构建安装验证。
- BF16/FP8 输入、不同路由、共享专家、容量边界和多次调用的精度验证。
- 多流生命周期、BS 切换、初始化/销毁和多机长时间运行验证。
- 各声明 EP 规模的设备测试及同输入、同软件版本、同计时范围的性能对比。
