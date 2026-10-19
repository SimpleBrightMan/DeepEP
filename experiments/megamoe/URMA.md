# MegaMoE URMA 固定 CV 实验

本实验是完整的 Prefill 算子：真实路由准备、本地展开、Dispatch、GMM1、SwiGLU/MXFP8 量化、GMM2、Combine PUT、最终同步和加权 Reduce。没有屏蔽计算，也没有使用预置 balanced 专用路由。

首版范围：单机 EP16、28 个 AIC / 56 个 AIV、每个 AIV 至少 256 KiB UB、H=3584、I=1280、384 个专家、TopK=8、无共享专家、FP8 输入/FP4 权重、BS=2048/4096/8192。任意合法路由均走同一流程；超过配置接收容量的偏斜路由会在启动计算前明确报错。

通信接口与固定 CV 调度参考 [ops-transformer PR11602](https://gitcode.com/cann/ops-transformer/pull/11602)（分析版本 `3ab6b8bba2cbdaf3751596c3e4b0488e1991df5f`）。计算复用本仓主线的 FP8/W4 CATLASS 和 SwiGLU 算术；返回方向改为 PUT。

## 数据与通信顺序

1. 每个输入 `[token, topk_slot]` 都是独立路由实例。按 `[目标rank][目标本地expert][原始token/topk位置]` 稳定排序，建立 `order` 和逆映射 `reduce_index[token,slot]=expanded_row`。
2. 本地将 FP8 token 和 E8M0 scale 展开到连续发送区。一个token命中多个expert时复制对应多行，本版不做去重。
3. 每个目标rank的连续行流按 `cv_rows` 截断，通过 URMA PUT 发送。token与scale使用各自连续平面。
4. 每个CV与expert区间取交集，产生 `source_rank、cv_id、expert、cv_row、expert_row、rows、source_expanded_row` 描述符。GMM1、SwiGLU、GMM2共用行身份；跨expert边界必须分别计算。
5. SwiGLU按已完成的gate/up tile推进；首版GMM2等待该CV/expert交集的全部SwiGLU tile完成，返回PUT等待整个CV的完整H列。GMM2直接写按来源rank排列的连续返回区。
6. 本rank的15个远端及self Dispatch全部完成后才提交Combine PUT，此处不插入全卡同步。Combine延后quiet，所有PUT完成后参加最终全卡同步，再按`reduce_index`读取各top-k结果、乘路由权重并求和。

15个通信AIV1各独占一个远端rank。通信worker `w` 的peer为 `w < rank ? w : w+1`；剩余13个AIV1执行SwiGLU。本卡通路使用本地搬运。两个阶段都只使用PUT，队列同一peer始终只有一个提交者。

接收数据/中间结果/返回结果不按CV覆盖复用。所有BS共享固定对称heap布局，epoch由`UrmaBuffer`统一递增，不因创建新Plan或切换BS重新归零。

通信域启用MTE和UDMA，每个对端独立QP0，不使用共享jetty。MTE只用于完成标志与本地搬运；远端token、scale和FFN结果均走UDMA PUT。每次调用在同一runtime stream串行执行，并在下一次写入前与各peer握手，保护上一轮Reduce尚未读完的结果。

## Python 接口与路由成本

```python
from deep_ep_experimental.megamoe.urma import UrmaBuffer

buffer = UrmaBuffer(group, num_max_tokens_per_rank=8192, cv_rows=256)
# x = (FP8 E4M3 tensor, E8M0 scale tensor)
# weights1/2 来自 transform_weights_for_mega_moe；不要拆掉GMM1准备对象。
y = buffer.forward(x, topk_idx, topk_weights, weights1, weights2)

# 仅用于显式复用不变路由，benchmark单独标成CACHED_FORWARD：
prepared = buffer.prepare_routes(topk_idx)
y = buffer.forward_prepared(x, topk_weights, weights1, weights2, prepared)
buffer.destroy()
```

`prepare_routes`每次在输入NPU上执行排序、计数、前缀和及逆映射。首版随后把384个counts复制到CPU，经Store交换完整计数表，native据此生成CV描述符并送回NPU。它包含设备同步和host控制开销，是后续需要优化的一部分。

`forward`始终包含这一步。`forward_prepared`只允许复用同一buffer的不变路由。修改prepared中的任何路由tensor后拒绝继续使用。默认`max_recv_rows=2*max_BS*topk`；非均衡路由可能需要在所有rank一致增大该参数，并承担额外HBM占用。

同一buffer的native调用统一入队到owner专用NPU stream。调用前等待caller stream的输入/描述符写入，返回时在caller stream插入输出就绪等待；不同caller stream也不会并发覆盖同一对称workspace。`destroy()`同步专用stream后再集体清理。

校验或启动异常会向Store发布abort，并禁止再次使用该buffer，避免各rank的epoch或调用序列分叉。Store abort能唤醒host协调等待，不能取消已经运行的NPU kernel；发生异常后应由启动器终止整个作业，不应捕获异常后继续调用或让单个rank进入清理屏障。正常cached forward不插入额外host集合通信。

## 编译和运行

使用本分支的完整MegaMoE wheel构建流程，URMA随`DEEPEP_BUILD_MEGAMOE=ON`编入，没有额外URMA编译开关。先激活Python环境并source匹配的CANN包，在仓库根目录执行：

```bash
export PYTHON="$(command -v python3)"
export DEEPEP_BUILD_MEGAMOE=ON
export DEEPEP_MEGAMOE_TIMER=ON
# 若使用已经安装的外部依赖，设成本机实际路径；否则保持未设置，
# build.sh会准备仓库固定版本的bundled依赖。
# export SHMEM_ROOT=/actual/shmem/install/shmem
# export CATLASS_ROOT=/actual/catlass
bash experiments/megamoe/scripts/build.sh
```

只能在已分配给本作业的全部16卡上运行。以下命令在仓库根目录执行，随后切到`/tmp`，避免源目录的`deep_ep`遮蔽已安装wheel：

```bash
REPO="$PWD"
OUT="$REPO/benchmark-runs/megamoe_urma/$(date +%Y%m%d_%H%M%S)"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:19091
export SHMEM_ROOT="${SHMEM_ROOT:-$REPO/third-party/shmem/install/shmem}"
export CATLASS_ROOT="${CATLASS_ROOT:-$REPO/third-party/catlass}"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib${CONDA_PREFIX:+:$CONDA_PREFIX/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
(
  cd /tmp || exit 1
  python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=16 \
    -m deep_ep_experimental.megamoe.urma_benchmark \
    --bs-values 2048,4096,8192 --routing balanced --cv-rows 256 \
    --warmup 2 --iters 10 --timer --output-dir "$OUT"
)
```

同一命令将三种BS顺序跑完，整个进程只创建、销毁一次通信域。原来的`DEEPEP_MEGAMOE_PREFILL_OPT`、4/24核配比、XOR顺序、MTE槽位数不控制这个独立URMA接口。

`cv_rows`首版接受128到4096之间128的整数倍。原始输入scale可以为`[BS,H/32]`或`[BS,H/64,2]`，Python在native调用前统一为后者，不改变scale内容。

默认只测性能，明确打印`VALIDATION skipped`，不将其写成精度通过。如需数值对照，先按照现有CPU Golden流程生成FP8、相同config/seed/routing/BS/check-tokens的缓存，再增加`--golden-dir <目录> --check-tokens <同一数值>`。CPU路由单元测试不等于NPU数值验证。

## 计时和带宽

每种BS输出三项相互独立的实测，正式统计取后半轮：

| 指标 | 计入范围 | 时钟 |
|---|---|---|
| `ROUTE_PREP` | NPU路由操作、D2H计数、Store交换、native描述符构建/拷贝 | Host wall time，完成后同步 |
| `CACHED_FORWARD` | 本地展开、两次GMM、SwiGLU、双向阶段PUT、最终同步和Reduce | NPU Event区间，不含预先准备路由 |
| `FULL_E2E` | 每次全新路由准备 + 完整forward + 完成同步 | Host wall time |

输入/权重生成、权重格式转换、通信域创建、计时前的host对齐屏障均不计入上述区间。输入契约是预量化FP8，BF16到FP8输入量化不在本接口内。`FULL_E2E`是单独执行的完整测量，不是将前两项相加。

`--timer`另跑一次诊断调用，保存每rank原始`urma_timer.bin`及schema。URMA格式为`int64[28,3,8,16]`，引擎顺序AIC/AIV0/AIV1，阶段为Pack/Dispatch/Gmm1/Activation/Gmm2/Combine/Barrier/Reduce。每记录前七个字段为`first_tick,last_tick,sum_ticks,payload_bytes,count,reserved,schema`。不要用旧Wave timer parser读取。

带宽分开报告Dispatch/Combine各自的INTRA/CROSS：

```text
单卡某阶段某通信类有效带宽 = 该类实际payload_bytes总和
                             / (该类最晚结束时刻 - 最早开始时刻)
```

只使用同一rank的时钟；不平均核时、不乘2、不跨rank相减。INTRA包含本卡copy。不同类别拥有不同时间窗口，不能把两类带宽直接相加。这个数值包含该阶段调度/等待间隙，是单向聚合有效带宽，不是链路瞬时峰值。

原始tick始终保留。换算默认沿用现有MegaMoE `timer/protocol.py`的1000 ticks/us；这不是硬件频率自动检测。必要时通过`--timer-clock-mhz`覆盖。

## 下一步优化方向

- 将计数交换、前缀和、CV描述符生成移到设备，减少同步和host工作；完整E2E应始终计入实际准备成本。
- 扫描CV行数和Mtile：当前均衡场景每来源每expert平均仅42.7/85.3/170.7行，大CV不会消除不同expert或来源的M尾块。
- 将GMM2就绪条件从整个CV/expert交集细化到完整I列的Mtile，缩短等待；本地展开可按原始token一次读入后复制多个目标行。
- 增加每peer队列的完成进度控制，同时保持独占提交；不能把15个peer owner直接改成多个核竞争同一队列。
- 在正确性与真实全算子收益确认后，才考虑受credit保护的接收/返回缓冲复用，以降低全量中间结果的HBM占用。

## 验证范围

2026-10-19 已通过 CANN 9.2.0 的设备核、Host扩展和完整wheel编译；MegaMoE及runtime的CPU回归为449项通过。CPU检查涵盖路由逆映射、CV/expert交集、28/13核任务覆盖、完成顺序、stream串行及通信域生命周期。尚未进行EP16 UBX上的数值与性能实测，不能将CPU检查或编译通过视为设备精度通过。
