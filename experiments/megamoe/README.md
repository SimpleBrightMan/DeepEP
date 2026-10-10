<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# MegaMoE Prefill Beta

## 本分支：零通信本地计算

`codex/megamoe-gmm2-wave-lifetime` 基于 `codex/megamoe-lifecycle-study`，实验性地延长同一 GMM2 wave 内的流水生命周期。
保持 tile、缓冲容量和任务顺序，延续 AIC/AIV0 credit 与游标；在 wave 尾部排空 FIX 后统一发布 slice 完成。
这是完成发布粒度的改变，可能影响真实 Combine 重叠；不能把本地计算回归当成完整通信验收。
Begin/End/FIX_DRAIN 仍是原生边界的无同步观测，非性能优化开关。
不建立SHMEM/HCCL、不执行通信预热；保留 GMM1 → SwiGLU → GMM2 调度和数学运算。
构建后运行 `python experiments/megamoe/scripts/local_compute.py --bs 4096`，默认八卡独立功能验证。
加 `--profile` 采集 timer-off kernel Duration；加 `--trace` 在性能采样结束后单独保存流水图。
环境、限制和验证记录见 [LOCAL_COMPUTE.md](LOCAL_COMPUTE.md)。
下面保留的 `batch_launch.py` 仍是原通信预热版，**不是**新的零通信入口。

## 原入口：计算重放诊断（compute replay）

基于 `codex/megamoe-local-ep8` 的 `a859c17`。本分支的 benchmark 默认只测计算重放，
不是完整 MoE，不作为生产或精度基线；下文原有 API 默认仍运行完整 MoE，只有显式调用
`buffer.enable_compute_replay()` 后才切换。

在已激活的 Ascend 环境、仓库根目录运行（首次不要加 `--skip-build`，需要重编译 kernel 和扩展）：

```bash
python experiments/megamoe/scripts/batch_launch.py --ep-size 8 --bs-values 4096
```

默认 DeepSeek V4 Pro：H=7168、I=3072、E=384、TopK=6、S=0；保留 timer 采集和自动解析。
正式性能轮次不传 timer，之后另跑 timer 批次。计时结果明确标为 `MegaMoe compute replay`，
JSON metric 为 `megamoe_compute_replay_duration_us`，不能与完整 E2E 混用。

每个 BS 先完整运行一次，设备同步后固定真实 Dispatch 输入、scale 和专家计数，再开始预热和测量：

| 阶段 | 重放计时内 |
| --- | --- |
| 输入量化/打包、Route/count、Dispatch、跨卡输入握手 | 跳过，仅 priming 执行 |
| 本地 flag reset、本卡同步 | 保留，保证连续调用安全 |
| GMM1 权重 Prologue、MMAD、SwiGLU + MX 量化 | 原后端、原分块与依赖 |
| GMM2 Prologue、MMAD、GM 写出 | 原后端与 Wave 顺序；保留原 FIX 完成边界和本地完成通知 |
| Combine 远端写、输出跨卡握手、Unpermute/加权聚合 | 跳过 |

EP8/BS4096 均衡路由下，每卡 48 个专家、每专家 512 token，共 24576 个路由 token；
不是只计算 4096 行，也没有改变专家形状。GMM2 仍按原有大 BS 策略滞后一 Wave。
`SendAndQuantBuffInit` 仅建立 UB 视图并初始化 scratch，不执行输入量化或网络传输。
`ResetSyncStatus` 只清 flag，不覆盖 priming 留下的 Dispatch 数据/scale/计数。
重放不再等待 Dispatch/count producer，但保留 GMM1→SwiGLU→GMM2 的 ready 协议。
AIV1 的优先级循环只消费 SwiGLU，不执行 Dispatch 或 Combine。

限制与读数：

- 仅支持 W4A8、S=0、perf；benchmark 拒绝 accuracy 和共享专家配置。
- SHMEM 初始化、priming、主机协调仍需要八卡；这些不在计算重放的 kernel Duration 中。
- 重放不会更新最终 `y`，它仍是 priming 的旧值；专家计数也是 priming 的快照，不能用来验证重放正确性。
- 同 BS 的输入/路由被固定。需要新输入时应重新创建 buffer；切换 BS 会建立新的计划并重新 priming。
- 重放时间包含本地读写、Prologue、激活、计算依赖及同步，不是纯 Cube MMAD 时间。
- “完整 E2E − 重放时间”不能直接称作通信耗时：通信与计算原本重叠，且通信去掉后资源竞争、缓存行为和 AIV1 调度均会变化。
- 若重放仍约 4200us，应重点查计算后端/本地数据流；若明显降低，说明原计时中的计算区间受到通信竞争或等待影响。仍需结合流水细分，不能仅凭差值定因。

timer 文件仍位于日志给出的运行目录下 `bs_4096/bs_4096/timer/`。
流水应没有 Dispatch、Combine、跨卡同步或 Unpermute；保留 GMM/Prologue/SwiGLU。
breakdown 的 `head/launch/dispatch` 为 N/A，新增 `setup_us` 表示首个可见事件到首次 GMM。
离线重新解析时显式声明重放口径，若要查看 wait，加 `--include-wait`：

```bash
python -m deep_ep_experimental.megamoe.timer <timer目录> --ranks 0-7 --compute-replay --include-wait
```

这里没有跨卡同步锚点，不支持 `--time-origin sync`；各 rank 的相对时间不能解释为全球对齐时间。
本地 CPU 测试不代表 CANN 编译或 NPU 正确性验证。上机需确认日志打印 `COMPUTE-REPLAY`，
timer 具有预期 GMM 数量且无通信事件，并观察多轮运行无挂起/非法访问后再比较性能。

---

公开入口：`deep_ep_experimental.megamoe`。W4A8 共享专家按实际单卡 B 自动选择小 BS / Prefill 策略；路由专家保留 Prefill 调度。
构建和多机测试入口位于本目录的 `scripts/`；机器地址及环境配置由使用者提供。

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
