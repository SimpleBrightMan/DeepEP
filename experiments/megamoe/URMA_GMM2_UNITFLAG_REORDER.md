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
- 非通信 AIV1 按全局 GMM2 tile ID 模 helper 数量静态分工，EP16/8/4 分别13/21/25核。
  就绪激活始终最高优先；其余 scale gather、Pack、重排按任务轮换，避免任何一项长期饥饿。
- 每个重排任务用两块 M128×N256 BF16 UB 槽，共128 KiB。按原 RowSegment 把远端行
  写到原 `gmm2Output` source-major 布局，本卡行直接写最终 symmetric return 区。
  只有所有该 tile 的 MTE3 写出完成后才给原 `gmm2Done[group]` 原子加一。
- Combine 和按 token 的 Reduce 只认原 `gmm2Done`；它们不能消费 `gmm2MatmulDone`。
  本卡入站累计完成还要求 **全部28个AIC完成且全部重排helper完成**，避免提前进入
  Reduce 的“所有来源完成、免读逐token flag”路径。空任务核心也各报告一次。
- 所有临时 flag/计数沿用 forward 启动 stream 上清零；新的张量加入 recordStream、
  workspace 统计和 debug snapshot。

新增临时区为 `receiveRows × H × 2` 字节，另有每 GMM2 tile 一个128字节完成槽。
相对直接 FIX Scatter 增加约 `2 × receiveRows × H × 2` 字节逻辑 GM 读写。
该量不等于实际 HBM流量，也不能直接当成新增串行时延；缓存和与计算通信的重叠要实测。

## 诊断与测试

命令行与基线一致，切分支后必须重新编译安装完整 wheel。先用同一机器、相同 routing / BS /
CV / ahead 配置对比关闭 timer 的 profiler kernel 时间，再单独打开 timer 看重叠与尾部。
EP4/8/16、H3584/I1280/E384/TopK8 的入口参数不变。

trace 新增 `Gmm2Reorder` 和 `WaitGmm2Reorder`（raw ID40/41），33–39不作为新设备事件，
避免与历史 JSON 合成 GMM 条的ID冲突。默认展示隐藏后者 wait。GMM submit 和延迟 drain 可能不相邻，不能
跨过其它 tile 强行合成同一条；计时不额外排空流水。

273项CPU测试通过：执行真实的 deferred flag 状态机、重排地址交集计算和尾部完成协议，覆盖短M尾、
任意来源分段、EP4/8/16、空核与自锁反例，并检查 trace/debug 导出。其中45项通过异步引擎
模型执行生产 `ReorderGmm2Tile`，检查所有14个N tile的逐元素输出、未写区域与写出后发布顺序。
CPU通过不代表 CANN 编译、NPU数值正确或性能收益；本次未进行 CANN/NPU验证。

```bash
python3 -m pytest \
  experiments/megamoe/tests/test_urma_unitflag_reorder.py \
  experiments/megamoe/tests/test_urma_reorder_device.py \
  experiments/megamoe/tests/test_urma_tail_protocol.py \
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
