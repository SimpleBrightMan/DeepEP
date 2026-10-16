# MegaMoE 四卡组分层去重原型

本分支实现 **跨组同号代理 GET → 组内唯一 token 缓存 → 原有专家计算 → 分层 Combine 归约**。
目标是先验证数据寻址、去重后的真实字节量、输出精度及整调用成本。
这是多 kernel、分阶段同步的可运行原型：尚未把通信重新融合进生产算子的逐 Wave 流水，
也没有实现“UB 读一次、直接向多个专家槽位复制”的优化。

## 编译和一键测试

在单机 16 张已分配卡、每卡 28 个 AICore 的 UBX 机器上运行，连续四个 rank 为一组。
先激活匹配的 Python/Torch 环境并 source CANN；SHMEM/CATLASS 依赖按仓库版本锁准备。
`SHMEM_ROOT` 要指向包含 include/lib/src 的完整 SDK；已有 SDK 时设置：

```bash
git fetch origin
git switch codex/megamoe-hierarchical-dedup
git pull --ff-only

# 改成当前机器上的实际 SDK 路径；构建入口也能准备仓库锁定的 SHMEM 子模块。
export SHMEM_ROOT=/actual/path/to/shmem/install/shmem
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:$ASCEND_HOME_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DEEPEP_BUILD_MEGAMOE=ON
export DEEPEP_MEGAMOE_TIMER=OFF
bash experiments/megamoe/scripts/build.sh

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:19091  # 使用当前机器的空闲端口。

# 只打印计划，不访问设备、不写输出。
python3 experiments/megamoe/scripts/hierarchical_bench.py --plan

# 检查当前 Python 的分层模块和原生接口，不初始化 NPU、不启动 rank。
python3 experiments/megamoe/scripts/hierarchical_bench.py --check-install

# 默认 H3584、I1280、E384、K8、S0、EP16、balanced，BS2048/4096/8192。
python3 experiments/megamoe/scripts/hierarchical_bench.py
```

已有 timer-enabled wheel 也可以使用；这个原型测试不传 per-core timer 指针。
切换分支后必须重新构建安装完整 wheel，不能复用另一个分支的扩展。
一键入口会清除扫描分支的两个通信核配比变量，每个 BS 单独启动一个 torchrun。
启动前在与 worker 相同的环境检查安装，并固定 worker 使用启动脚本的 Python，防止遗留 `PYTHON_EXEC` 切到另一环境。
安装信息保存在 `installation.json`；`PYTHONPATH` 保留，以兼容 CANN Python 依赖。
读取 `npu-smi info` 并保存快照，不会抢占或杀掉已有作业；16 张可见卡应全部分配给本次任务。

```bash
# 先验证一个 BS。
python3 experiments/megamoe/scripts/hierarchical_bench.py --bs-values 2048

# 随机路由包含多个专家落在同一卡的情形，且使用可能为负的路由权重。
python3 experiments/megamoe/scripts/hierarchical_bench.py --bs-values 2048 --routing random

# 高度倾斜路由检查去重、空任务和本地多专家归约。
python3 experiments/megamoe/scripts/hierarchical_bench.py --bs-values 2048 --routing skewed

# BF16 输入仍会量化为 FP8，权重保持 FP4。
python3 experiments/megamoe/scripts/hierarchical_bench.py --bs-values 2048 --input bf16
```

支持范围：W4A8、EP16、28 核、无共享专家；BS 为 `[384,8192]` 中 128 的倍数，
H 为 `[1024,8192]` 中 256 的倍数，I 为 `[256,4096]` 中 128 的倍数；
E 能被 16 整除且每卡专家数不超过 128，K≤min(32,E)。不支持 Decode。
因此 H=3712 目前不支持，默认 H=3584 支持。

### 找不到 hierarchy_benchmark 模块

该入口必须安装到运行 Python 的 `site-packages`；只编译 `.so` 或加载旧 wheel 不足以运行。
本分支的 wheel 包含 `hierarchy.py` 和 `hierarchy_benchmark.py`。
新版构建脚本会在安装后检查新入口及原生 `Plan.configure_hierarchy` / `hierarchy_buffer_size`。

若执行过 `build.sh` 仍报错，先用 `--check-install` 查看实际 Python、包路径与错误，
再核对 `PYTHON`、`PYTHONPATH` 和本次安装的 wheel。可以显式统一解释器：

```bash
PY=$(python3 -c 'import sys; print(sys.executable)')
PYTHON="$PY" bash experiments/megamoe/scripts/build.sh
"$PY" experiments/megamoe/scripts/hierarchical_bench.py --check-install
```

如果已有包含这两个新模块的正确 wheel，只需用该 Python 重新安装这个 wheel，无需重新编译。
不要只把仓库源码加入 `PYTHONPATH` 来绕过错误，否则可能把新 Python 接口与旧 native 扩展混用。

## 当前的数据流

设源 token 身份为 `(source_rank, token_id)`，全局 token key 为
`key = source_rank * B + token_id`；某个路由边的身份为
`edge = key * K + topk_slot`。同 token 不能重复选择同一专家，但可以选择同一卡上的多个专家。

每个目标组 `G` 的代理固定为：

```text
proxy(G, source_rank) = G * 4 + source_rank % 4
```

例如 rank11=G2_R3 的 token 被 G0 内任意专家消费，则 G0_R3=rank3 负责从 rank11 拉取一次。
rank3 的代理槽位为 `source_group * B + token_id`；这个槽位在整个调用期间稳定。
需要该 token 的 G0 内各 rank 再从 rank3 取一次，写入自己的唯一 token cache。
同组源 token 直接从源 rank 获取，省略跨组代理步骤。

### 控制面和寻址表

调用 `buffer.configure_hierarchy()` 时：

1. 把本卡真实 `topk_idx` 复制到 CPU；通过 TCPStore 分块交换全局 `[16,B,K]` 路由。
   每块不超过 32768 个整数，避免超过 StoreGroup 单条消息的限制。
2. 对每个 token 构建 16-bit 消费 rank 掩码，得到它需要到达的组和 rank。
3. 为每个 rank 构建 `token_keys` 和 `token_slots[key]`：一个源 token 在本 rank 只占一个 cache 槽位。
4. 构建 `edge_keys` 和 `edge_slots[edge]`：每条实际专家路由都有独立的 GMM2 输出槽位。
5. 生成跨组 GET、组内 GET、路由权重搬运，以及三层归约的源/目的地址表，上传到设备。

所有 rank 从同一份全局路由独立生成确定的目录，因此 v1 不再另发一轮“代理槽位目录广播”。
`token_slots`、`edge_slots` 与代理固定槽位一起解决“本卡要从代理第几个位置取”“结果属于源 token 的第几条路由边”。
目录里的缺失项为 -1；实际任务只为有效消费者创建。

**当前按整次调用建立缓存和去重任务，未按 Wave 回收缓存。**
原 MegaMoE 的专家 Wave、GMM 行布局、tile-ready 依赖继续使用；
进入原 Dispatch 时根据 key 把输入地址改为本地唯一 cache，专家需要多份时从本地 cache 再搬运。
因此跨卡重复流量已经去掉，而本地 GM→UB 的重复读取仍存在。

### Forward 的八个阶段

| 阶段 | 当前执行内容 | 结束条件 |
|---|---|---|
| Prepare | 原输入打包/量化和路由准备，把当前路由权重写入对称区 | Stream 同步 + 全 rank 屏障 |
| Metadata | 拉取当前调用的路由权重到本地表 | Stream 同步 |
| Dispatch CROSS | 同号代理从三个远端组 GET 去重 token | Stream 同步 + 全 rank 屏障 |
| Dispatch INTRA | 每个 rank 拉取本组源/代理的数据，形成唯一 token cache | Stream 同步 |
| Compute | 原 GMM1/SwiGLU/GMM2 Wave；GMM2 的每条 BF16 输出写本地 edge 槽位 | Stream 同步 + 全 rank 屏障 |
| Local reduce | 同一卡上同 token 的专家输出转 FP32、乘相应路由权重并求和 | Stream 同步 + 全 rank 屏障 |
| Group reduce | 同号代理收集组内各消费 rank 的 FP32 partial，得到组内和 | Stream 同步 + 全 rank 屏障 |
| Source reduce | 源 rank 取各目标组代理的 FP32 partial，求和后转换为 BF16 输出 | Stream 同步 + 全 rank 屏障 |

权重只在 Local reduce 乘一次。组内和跨组后续阶段只求和；共享专家路径没有接入。
当前用阶段屏障保证数据完成与缓存可见性，不存在未实现 ready 标记就让消费者提前读取的问题。
其代价是多次 kernel 启动、Host 同步及通信/计算重叠减少；测量会把这些成本保留下来。

## Combine 为什么这样返回

一条输出从 `edge_slots[(source_rank * B + token_id) * K + topk_slot]` 找回对应专家结果，
并从全局权重表读取同一条 edge 的路由权重。然后按 token key 在本卡、目标组代理、源 rank 三级归约。
源 rank 的最终输出行始终是原来的 `token_id`，不用依赖 token 在接收 buffer 中的到达顺序。

Local reduce 输入保留原 GMM2 的 BF16 输出，累加与跨层 partial 使用 FP32，最后才转 BF16。
归约次序与原直接路径不同，不能承诺逐 bit 一致。默认验收要求全输出有限、专家计数完全相同、
relative RMSE≤0.01 且最大绝对误差≤0.05；`--relative-rmse`、`--max-abs` 可显式设置。

## 默认 balanced case 的通信量

对 H=3584、K=8、EP16、连续四卡一组，仓库 balanced 路由每个 token 选择八个不同 rank，
覆盖两个完整组。下表是全部 rank 汇总后，平均到每个源 token 的有效网络 payload，
不包含本卡读取、metadata、协议控制和重复本地 cache 搬运：

| 通信类别 | 原 flat 路径 | 当前分层原型 | 分层 / flat |
|---|---:|---:|---:|
| Dispatch 跨组 | 6 × 3712 B | 1.5 × 3712 B | **25%** |
| Dispatch 组内 | 1.5 × 3712 B | 6 × 3712 B | 400% |
| Combine 跨组 | 6 × 7168 B，BF16 | 1.5 × 14336 B，FP32 | **50%** |
| Combine 组内 | 1.5 × 7168 B，BF16 | 6 × 14336 B，FP32 | 800% |

跨组“向量个数”都减少到 25%，但 Combine 传 FP32，单个向量字节翻倍，所以跨组字节只减少到 50%。
组内流量增加是把聚合工作搬到组内代理的成本；不能把全网络字节总量也称为原来的 25%。
随机/倾斜路由的比值不同，运行时按真实路由生成独立 Python oracle 并校验原生计数。

## 如何读带宽和总耗时

一键脚本最终产生 `summary.json` 和 `summary.csv`，并打印：

- 原 flat 和 hierarchy 的完整 ACL event span：每轮取 16 个 rank 最大值，再取均值。
- hierarchy 的 Host wall 时间，包含 Python/API 检查与同步路径。
- 路由收集、CPU 建表和任务上传的 plan wall 时间，按 rank 单独保存，并报告最大值。
- Dispatch CROSS/INTRA、Combine INTRA/CROSS 四个阶段的有效 payload GB/s。
- 如果路由每次变化，`rebuild_every_call_estimate_us` 给出“最大 plan 时间 + 完整 forward wall”的估算。

**ACL event span 包含全部 kernel、Python collective preflight 等待及阶段间 Host 间隙，不是单个 fused kernel 的 Duration。**
各次计时调用开始前的外层对齐 barrier 不在测量内；调用内部的同步和阶段间屏障在测量内。
**阶段带宽 = 对应 GET payload / 同步阶段 wall 时间**，包括该阶段的同步、归约和等待，
不等价于 MTE/物理链路忙时带宽。不要直接与独立 Dispatch benchmark 的 class envelope 当成同一种计量。
INTRA 有效网络字节排除同卡读取，但当前阶段也会执行同卡拷贝/归约，耗时保留这些工作。

正确性使用同进程、同输入、同权重的原直接路径作为全量回归基准：
检查全部 `[B,H]` 输出和每专家计数，并在重复调用后再次检查。
额外变更输入和路由权重（包含负权重）后再与直接路径比较，检查 cache 是否按新输入刷新、权重是否正确更新。
这属于原路径回归，不能代替独立 CPU Golden 的数值正确性证明。

## 缓存容量和接口

构造时使用 `SymmBuffer(..., hierarchical=True)`，提前按最大 B 预留：
原通信区、4B 个代理输入槽、16B 个本 rank FP32 partial 槽、4B 个组级 FP32 partial 槽及权重。
对称区逻辑容量按 2 MiB 对齐，最高 8 GiB，SHMEM 初始化按实际需要预留 heap；超出即报错。
恰好为 8 GiB 时沿用 runtime 的额外 2 MiB guard 分配。
另外还有非对称本地唯一 token cache、每 edge 的 BF16 输出、目录/任务表及原 GMM workspace；
8 GiB 仅是对称区上限，不是本算子的全部 HBM 占用。

```python
buffer = SymmBuffer(group, hierarchical=True, ranks_per_node=16, **model_and_capacity)
# 填充 x/x_sf、topk_idx、topk_weights 后：
buffer.validate_inputs(B)
buffer.prepare(B)
buffer.configure_hierarchy()  # 全 rank 同序调用，建表一次。
fp8_fp4_mega_moe(y, w1, w2, buffer)  # 此后使用已配置的分层路径。
```

输入和路由权重可以每次调用变化，数据每次重新搬运；专家权重也可以按公共接口更新。
路由 ID、B 或 buffer 变化后必须重新集体配置；入口检查 route tensor 身份/版本，拒绝静默使用旧目录。
默认测试复用固定路由，plan 时间单独报告，不能把它解释成动态路由无成本。
当前只支持 `StoreGroup` 控制面，分层调用不接受 per-core timer，使用阶段指标替代。

## 修改范围与后续优化

新增 `hierarchy_plan.hpp` 构建消费者掩码、紧凑 token/edge 目录；
`hierarchy_host.cpp` 负责设备任务表、额外对称区和阶段编排；
`hierarchy_kernels.hpp` 提供复制和归约 kernel。
生产 Dispatch 的读地址可以切换到唯一 cache，Combine 的写地址可以切换到本地 edge 槽位，
原 GMM Wave 与专家计算布局保留。公共 API 增加容量与固定路由配置入口，
Python helper 提供真实路由的独立 payload oracle，runner/launcher 提供全 rank 回归与总成本测试。
未启用 `hierarchical=True` 或未调用 `configure_hierarchy()` 时仍执行原路径。

后续优先级：

1. **逐 Wave 的 Device ready/epoch 协议和滚动缓存。** 把整调用全局阶段屏障改为生产者/消费者局部依赖，
   重叠跨组 GET、组内分发与 GMM，并控制已消费后才复用槽位。
2. **UB 一次读取、多专家展开。** 当前只去掉跨卡重复；把同 token 的多个专家行直接从同一 UB 载入展开，
   并保持原 tile-ready 的发布条件。
3. **去掉全量 BF16 edge 落盘和多级 GM 往返。** 在 GMM2/Combine 消费点累计本地 token partial，
   使用分块归约流水；同时量化比较 FP32 partial 与较窄格式的精度/字节权衡。
4. **真正的持续 DMA 多槽队列。** 当前复制/归约辅助 kernel 以可验证为先，后续增加 MTE2/MTE3 重叠、
   批量请求和跨组顺序优化，再扫描代理通信核数。
5. **Device 建表和分层控制面。** 替代 CPU D2H/TCPStore 全局路由交换，使用小型消费掩码、分层目录交换，
   并针对动态路由把建表成本纳入端到端优化。
6. **处理代理负载不均。** 固定 same-lane 代理保持当前链路规则；随机/倾斜路由下评估负载，
   再考虑可计算的代理轮换与目录编码，避免只减字节却把流量集中到少数卡。

代码编译通过或 CPU oracle 通过不代表 UBX 上的速度已改善；最终以脚本打印的全量回归、真实字节量、
完整调用耗时及每次重建路由的估算成本判断。
