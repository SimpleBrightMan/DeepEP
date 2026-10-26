# GMM2 连续写出与辅助核重排实验

分支：`megamoe-urma-gmm2-unitflag-reorder`，基于 `megamoe-urma` 的 `10d6a04`。

本实验保留 M256 group / GMM2 256×256 tile、现有专家顺序和全局 tile 分核。
GMM2 从分段 FIX Scatter 改为连续写入 BF16 expert-major 临时区，使 GMM1/GMM2
都走单次连续输出的 Unit Flag 路径。没有修改 CATLASS 主线实例或通信 CV/QP 协议。

## 一套跨阶段、跨专家的计算上下文

每个 AIC/AIV0 对使用同一个 `ContinuousMatmul` 对象，第一次有任务时 Begin，
全部任务结束时才 End。更新专家、投影或形状只改变操作数地址和布局描述，保留 L1A/B、
scale、L0A/B、L0C 以及反量化 UB 槽位，保留 8/9 号跨核空闲槽信用。GMM1 输入仍按
RowSegment gather，GMM2 输入用连续 ND2NZ；两者输出均为连续 GM。

完成发布用独立 `FIX_S1`：

```text
submit(A 的逻辑 tile)               // GMM1 包含 gate 和 up 两个投影
set(FIX_S1)                        // 先截取 A 的完成边界
if 下一任务输入/激活已就绪:
    submit(B 的逻辑 tile)
    wait(FIX_S1); publish(A)
    set(FIX_S1)                    // 现在才截取 B
else:
    wait(FIX_S1); publish(A)        // 阻塞依赖前必须放行生产者，避免自锁
    wait(下一任务依赖)
最后: wait(FIX_S1); publish(最后任务); End
```

专家边界、GMM1/GMM2 边界不强制排空，也不增加 `PIPE_ALL`。原有 kernel 进入 Reduce
和最终退出时的资源交还屏障保留。连续输出仍使用原有单 L0C Unit Flag 协议，不宣称
M256×N256 放得下两块 FP32 L0C。

## 重排与完成协议

- 每个 GMM2 tile 的 FIX 完成后置 `gmm2MatmulDone[tile]`，只表示临时区已写好。
- 全部28个 AIV1 参与重排，严格沿用 GMM2 的全局 tile ID 模28分工：AIC block b
  计算的 tile 只由同 block 的 AIV1 重排。EP16/8/4 的通信核仍分别为15/7/3个，
  激活、scale gather、Pack 的静态分工不变，不增加队列共享或任务争抢。
- 通信核先完成本轮 Pack 配额，再于每次 Dispatch/Combine quiet 前处理本核当前
  已就绪的全部重排 tile；没有就绪 tile 就继续通信，不等待其 AIC。无可发送 Combine、
  空 peer 和发送已经结束的路径也继续推进重排，直到本核任务完成，不能提前退出。
- 非通信辅助核仍优先执行就绪 SwiGLU；Pack 优先于重排。重排完成仍以最终 GM
  写出为准，不能仅依据计算临时区就绪放行通信。
- 每个重排任务用两块 M128×N256 BF16 UB 槽，共128 KiB。按原 RowSegment 把远端行
  写到原 `gmm2Output` source-major 布局，本卡行直接写最终 symmetric return 区。
  只有所有该 tile 的 MTE3 写出完成后才给原 `gmm2Done[group]` 原子加一。
- Combine 和按 token 的 Reduce 只认原 `gmm2Done`；它们不能消费 `gmm2MatmulDone`。
  本卡入站累计完成还要求 **全部28个AIC完成且全部28个AIV1重排完成**，避免提前进入
  Reduce 的“所有来源完成、免读逐token flag”路径。空任务核心也各报告一次。
- 所有临时 flag/计数沿用 forward 启动 stream 上清零；新的张量加入 recordStream、
  workspace 统计和 debug snapshot。

新增临时区为 `receiveRows × H × 2` 字节，另有每 GMM2 tile 一个128字节完成槽。
相对直接 FIX Scatter 增加约 `2 × receiveRows × H × 2` 字节逻辑 GM 读写。
该量不等于实际 HBM流量，也不能直接当成新增串行时延；缓存和与计算通信的重叠要实测。

重排仍使用原来的低128 KiB UB，并在每个 tile 末尾等待 MTE3 后发布完成。
通信核长期保留的 signal / submitted / pending 数据移到与此缓冲不重叠的区域，
防止在 quiet 前重排时破坏尚未发布的 Combine 批次。此次只扩大重排分工和调整调度，
没有取消逐 tile 写出等待或改变整 M-group 的 Combine 就绪条件。

每个 AIV1 的重排游标直接读取对应 AIC 使用的 `computeTasks` / `computeBatches`，
过滤出其 GMM2 子序列。该 AIC 的完成标志按这个子序列发布，因此首个未就绪任务之后
不会已有本核的已发布任务；无需扫描其它核的 tile，也无需重复遍历整张路由表。
成功的 ready 检查缓存到本 tile 消费，避免为关闭等待计时再次读取同一个 GM flag。

通信核调度示意：

```text
Dispatch PUT(X, scale)
完成本核本轮向前准备的 Pack 配额
重排对应 AIC 当前已发布的全部 tile（遇到未就绪立即返回）
Dispatch quiet → 发布 CV 完成

扫描并提交已就绪的 Combine CV
重排对应 AIC 当前已发布的全部 tile
Combine quiet → 发布这一批 CV 完成

没有可发送的 CV：仍推进本核重排，不能只等 Combine 就绪
发送结束：先发布 peer finished，再持续重排并观察入站完成
本核重排全部完成且入站观察结束后退出通信循环
```

`WaitStartPeer` / `WaitDispatch` / `WaitLocalDispatch` / `WaitCombineReady` 中执行
重排时，先结束当前 wait 记录，重排后再开始新的 wait，保持同一引擎的 trace 不重叠。
没有任务就绪的轮询不会新增逐轮 wait 记录。

## Combine quiet 前提前准备一轮

在28核一一对应重排的基础上，每次 **Combine quiet** 前增加下一CV的本核重排保证。
原来只执行 `DrainReady()`，若本核下一个 GMM2 tile 尚未发布就立即进入 quiet；
quiet 等待期间该 tile 即使就绪，本核也无法重排，可能阻碍下一轮 Combine 提交。

现在按以下顺序执行：

```text
当前一批 Combine PUT 已提交
找到本 peer 按 CV 顺序的第一个尚未提交的 CV（跳过已提交的洞）
从该 CV 的计算组依赖中，找出本 AIC 真正拥有 tile 的最晚一个组
推进本核的重排游标，直到该组的本核 tile 全部完成：
    已就绪 → 重排
    尚未就绪 → 等待本 AIC 的 matmulDone，并显式退避200 cycle
继续重排本核其它当前已就绪的 tile
Combine quiet → 发布当前批次的完成 flag
重新扫描 CV，提交已就绪的数据
```

这个保证只针对下一CV中**归本核负责的重排**；其它 AIV1 的贡献仍由它们独立推进。
不在此处等待整组 `gmm2Done`，不增加跨核/跨卡同步、不转移 tile 所有权。
该路径只在本卡 Dispatch/Pack 完成后执行，因此不会扣住尚未发布的 Dispatch 数据。
生产调度 `BuildComputeTasks` 按专家和 fragment 递增提交 GMM2，允许用最晚依赖组界定
本核的重排前缀；不是根据绝对均衡路由推导依赖。

下一CV没有本核任务、当前没有待 quiet 的 PUT，或所有CV已提交时，不增加生产者等待。
最后一个实际未发送CV仍单独 PUT/quiet；最终 quiet 不等待无关的未来重排。
Dispatch quiet 前继续保持 Pack 优先、只做当前就绪重排的原逻辑。
`WaitGmm2Reorder` 与 `Gmm2Reorder` 分开计时，二者都位于 `CombineQuiet` 计时范围外；
轮询失败不逐次打点，没有新增 `PIPE_ALL`。

这会把对下一批生产者的部分等待移到当前 quiet 之前，当前已提交 PUT 可在此期间推进；
也可能延迟当前批次完成 flag 的发布，最终收益需上板对比，不能仅据 quiet 条缩短判断。

本次393项相关CPU回归通过，包含21项直接执行生产 lookahead / OwnedReorder / quiet
逻辑的新检查：晚到生产者、非连续依赖、乱序已提交CV、无本核任务、最终quiet、空批次、
目标边界以及Wait/Reorder计时不重叠。未进行 CANN 编译或 NPU 验证。

## 诊断与测试

命令行与基线一致，切分支后必须重新编译安装完整 wheel。先用同一机器、相同 routing / BS /
CV / ahead 配置对比关闭 timer 的 profiler kernel 时间，再单独打开 timer 看重叠与尾部。
EP4/8/16、H3584/I1280/E384/TopK8 的入口参数不变。

trace 新增 `Gmm2Reorder` 和 `WaitGmm2Reorder`（raw ID40/41），33–39不作为新设备事件，
避免与历史 JSON 合成 GMM 条的ID冲突。默认展示隐藏后者 wait。GMM submit 和延迟 drain 可能不相邻，不能
跨过其它 tile 强行合成同一条；计时不额外排空流水。

首版273项CPU测试通过：执行真实的 deferred flag 状态机、重排地址交集计算和尾部完成协议，覆盖短M尾、
任意来源分段、EP4/8/16、空核与自锁反例，并检查 trace/debug 导出。其中45项通过异步引擎
模型执行生产 `ReorderGmm2Tile`，检查所有14个N tile的逐元素输出、未写区域与写出后发布顺序。
CPU通过不代表 CANN 编译、NPU数值正确或性能收益；本次未进行 CANN/NPU验证。

28核重排更新共314项相关CPU回归通过，新增直接执行生产任务游标及通信循环的检查：
EP4/8/16 分工、随机稀疏任务、空核、空 peer、最高512个CV的持久UB状态、Pack优先、
最后CV独立quiet、完成计数恰好一次，以及 wait/Reorder trace 区间不重叠。
本次未进行 CANN 编译或 NPU 运行，性能收益与设备精度仍需实测。

```bash
python3 -m pytest \
  experiments/megamoe/tests/test_urma_unitflag_reorder.py \
  experiments/megamoe/tests/test_urma_reorder_device.py \
  experiments/megamoe/tests/test_urma_reorder_progress.py \
  experiments/megamoe/tests/test_urma_reorder_comm.py \
  experiments/megamoe/tests/test_urma_reorder_lookahead.py \
  experiments/megamoe/tests/test_urma_tail_protocol.py \
  experiments/megamoe/tests/test_urma_transport.py \
  experiments/megamoe/tests/test_urma_trace.py \
  experiments/megamoe/tests/test_urma_trace_merge.py \
  experiments/megamoe/tests/test_urma_trace_coalesce.py \
  experiments/megamoe/tests/test_urma_trace_text.py \
  experiments/megamoe/tests/test_urma_debug.py -q
```

## 编译与 EP16 测试命令

沿用已激活的 Python 和已 source 的 CANN 9.2.0 环境。依赖版本与基线相同，
已验证的 SHMEM PR944 SDK 可以复用。下面正式轮次均关闭内部 timer/trace，
profiler 统计每轮跨 rank 最大值均值、最小值均值；最后额外一轮输出诊断流水图。
不传 `--golden-dir` 时不做数值精度认证。

```bash
cd /home/t00953803/projects/DeepEP
git fetch origin megamoe-urma-gmm2-unitflag-reorder
git switch megamoe-urma-gmm2-unitflag-reorder
git pull --ff-only
export PYTHON="$(command -v python3)"
export DEEPEP_BUILD_MEGAMOE=ON
export DEEPEP_MEGAMOE_TIMER=ON
bash experiments/megamoe/scripts/build.sh

export SHMEM_ROOT="$PWD/third-party/shmem/install/shmem"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib${CONDA_PREFIX:+:$CONDA_PREFIX/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export DEEPEP_SHMEM_ENDPOINT="tcp://127.0.0.1:$("$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
OUT="$PWD/benchmark-runs/gmm2-unitflag-reorder/$(date +%Y%m%d_%H%M%S)"
cd /tmp
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=16 \
  -m deep_ep_experimental.megamoe.urma_benchmark \
  --config urma --bs-values 2048,4096,8192 --routing balanced \
  --cv-rows 256 --gmm1-ahead-mgroups 12 --warmup 10 --iters 20 \
  --timer --trace --trace-capacity 8192 --timer-clock-mhz 1000 \
  --output-dir "$OUT"
```

EP8/EP4 只需相应选择可用设备并把进程数改为8/4；任务分工随EP自动调整。
比较两个实验时请保持相同的机器、EP、路由、输入、CV和提前量。
