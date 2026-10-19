<!-- Copyright (c) 2026, SimpleBright_Man -->

# 单机 EP16 Prefill：mega-prefill-1019

本分支从 main 移植通信优化到完整 W4A8 MegaMoE。GMM1、SwiGLU、GMM2、
Unpermute 均执行；每个 Wave 的专家、原始 token 行号和计算任务归属保持主线逻辑。
这是完整算子的实验分支，仍需在目标 UBX 上验证数值和性能。

## 默认配置与适用条件

| 项目 | 启用后的行为 |
| --- | --- |
| Dispatch metadata | prepare 时生成本地身份表；按 token 用 MTE2 取入 UB，再由 MTE3 写出，移除逐 token 的 metadata scalar 写入与相应等待 |
| Dispatch 通信核 | AIV1 0–3 本组，4–27 跨组；6 个流水槽 |
| Combine 通信核 | AIV1 0–3 本组，4–27 跨组；8 个流水槽 |
| 本组定义 | 连续 4 个 rank；包含本卡搬运 |
| 跨组顺序 | 对本组 g，依次访问 g XOR 1、g XOR 2、g XOR 3 |
| 组间切换 | 各核独立推进；不新增阶段同步，不在组边界排空流水 |
| 任务分配 | 在各组内部划分连续 token 区间，按 slice 轮转余数；保留实际路由及原目标位置 |
| 计算与就绪 | 保留主线 Wave/slice、GMM2 producer 数和 ready 协议 |

默认 `DEEPEP_MEGAMOE_PREFILL_OPT=1`。实际启用要求：单机 EP16，
`ranks_per_node=16`，28 AIC/56 AIV，B>256、S=0、W4A8、BF16 Combine，
关闭 top-k 权重预取，主线 Dispatch 可分配 6 槽，且 UB 足够容纳原布局及新流水。
未知或不同的 ranks_per_node 不会自动套用连续四卡拓扑。
Combine 八槽必须装入原来的 120 KiB 行缓冲区，原 metadata 区末端为 248 KiB；
超出资源限制时回退原路径。该配置下 H=3584、I=1280、E=384、K=8，
B=2048/4096/8192 是本次测试目标。

`DEEPEP_MEGAMOE_PREFILL_OPT=0` 运行原始完整算子调度。这个环境变量在创建
SymmBuffer 时读取，应在新进程中切换。小 BS、共享专家和其它不满足条件的形状
使用原路径。benchmark 的 `[PREFILL]` 输出和资源 JSON 记录实际启用情况；
Python 调用方也可在 `prepare(B)` 后读取 `buffer.prefill_optimization_config`。

metadata 表按 `(source_rank, token_id, topk_slot)` 索引，与专家选择无关。
同 BS 更换路由无需重建；更换 BS 时由 `prepare(B)` 重建，准备耗时不计入 forward。
EP16/K8 下 B=2048/4096/8192 每卡分别增加 8/16/32 MiB 表存储。
本版本没有加入 alltoall 的 stride 分配、分层去重或已撤回的 Combine metadata 批量预取。

## 编译和运行

先激活匹配的 Python 环境、source CANN 环境，并准备可用的 SHMEM SDK。
不要在旧 wheel 上仅替换 Python 文件：本分支更新了 native API、tiling ABI 和 timer schema。

```bash
git fetch origin
git switch mega-prefill-1019
git pull --ff-only
export PYTHON="$(command -v python3)"
export DEEPEP_BUILD_MEGAMOE=ON
export DEEPEP_MEGAMOE_TIMER=ON
bash experiments/megamoe/scripts/build.sh
```

使用已分配的全部 16 张卡：

运行时离开源码根目录，避免其中的 `deep_ep/` 遮蔽已安装 wheel。
`_build_info.json` 属于 wheel 的编译产物，不应手工复制到源码包。
若显式将仓库路径加入了 `PYTHONPATH`，也应移除该项；保留 CANN 所需的路径。

```bash
REPO="$PWD"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export DEEPEP_MEGAMOE_PREFILL_OPT=1
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:19091
export LD_LIBRARY_PATH="${SHMEM_ROOT:-$REPO/third-party/shmem/install/shmem}/lib:$ASCEND_HOME_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
OUT="$REPO/benchmark-runs/mega-prefill-1019/$(date +%Y%m%d_%H%M%S)"
(
cd /tmp || exit 1
python3 -c 'import deep_ep,pathlib; p=pathlib.Path(deep_ep.__file__); print(p); assert p.with_name("_build_info.json").is_file(), "Installed wheel metadata missing or source package shadows wheel"' || exit 1
python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=16 \
  -m deep_ep_experimental.megamoe.benchmark \
  --config prefill_1019 --mode perf --input fp8 \
  --bs-values 2048,4096,8192 --routing balanced \
  --timer --timer-ranks 0-15 --output-dir "$OUT"
)
```

算子耗时使用关闭 timer 的重复执行；通信带宽来自额外开启 timer 的诊断调用。
`--mode perf` 不做数值回归，可在单独运行中改为 `--mode accuracy` 验证。
这次完整算子耗时不能直接与旧分支的 COMM-KERNEL 耗时对比。
同一台机器的原调度对照可在新进程设 `DEEPEP_MEGAMOE_PREFILL_OPT=0` 后运行相同命令，
使用不同输出目录；原调度不按固定 4/24 核分类，因而不输出这四类带宽。

## 单向聚合有效带宽

每个 rank 分别输出 Dispatch/Combine × INTRA/CROSS 四项，单位 GB/s（十进制）。
数据量来自实际外层 COMM timer 的 `row_count`，不假设路由绝对均衡。
相同 token 发给多个专家按实际搬运次数计算，未做去重。

对某个 rank 的某一阶段、某一通信范围，令每条有效通信区间为 `[s_i,e_i]`，
总有效搬运字节数为 B：

* **envelope**：`B / (max(e_i) - min(s_i))`。包含该类通信首尾之间的计算、等待和空泡，
  用于衡量完整算子流水中的有效交付速度。
* **active/union**：`B / length(union([s_i,e_i]))`。合并所有相关核、所有 Wave 的重叠区间，
  只剔除没有该类通信条覆盖的间隙，用于分析通信执行期间的聚合有效速度。
* **coverage**：`active_time / envelope_time`。解释两种口径之间的差异。

时间从微秒换算时，`GB/s = bytes / (time_us * 1000)`。不再求核心耗时平均，
不把并发核时间相加，也不乘双向系数 2。

H=3584 时，Dispatch 每路由行计 3712 字节（FP8 token 和打包 scale），
Combine 计 7168 字节（BF16 token）。本地 metadata 表、控制信息及本地中转读写不重复计入。
INTRA 含本卡，因而是本组有效搬运带宽；CROSS 是本卡面对其余三组的单向聚合有效带宽。
两者均不是某一对 rank 或单根物理链路的峰值。

COMM 条覆盖完整通信函数，包括地址计算、metadata 和完成等待；active 也不是硬件 MTE
独占忙碌时间。四项使用各自的时间窗口，带宽数值不能相加。
各 rank 先独立计算，再汇总均值和最小/最大值，不对不同 rank 的时间戳直接做减法。
默认收集所有 rank，也可用 `--timer-ranks` 指定子集，输出会标明覆盖范围。

保留原 timer 事件并增加外层 `AIV_DISPATCH_COMM`/`AIV_COMBINE_COMM`，避免与内部条重复计数。
每核版本和完成标记用于拒绝旧 schema 或不完整采样。原始 trace 仍可用来查看逐 Wave、逐核心行为。

## 本次验证

2026-10-19：在 Linux aarch64、CANN 9.2.0 B080、Python 3.12 的环境中，
`megamoe_ext`（Host 扩展及开启 timer 的设备核）编译通过，扩展加载检查通过，
MegaMoE 主机侧回归 254 项通过。未进行 NPU 数值或性能测试。
