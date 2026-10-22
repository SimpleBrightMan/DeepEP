# URMA benchmark 的起跑对齐与连续计时

`urma_benchmark.py` 使用 **torch_npu profiler 的设备 kernel 耗时**，保留参考 [mega_test](https://gitcode.com/SimpleBright_Man/mega_test/blob/main/benchmark/timing.py) 的 HCCL + Exp 起跑对齐。不再创建计时 Event，也不测量或打印 ROUTE_PREP、CACHED_FORWARD、FULL_E2E。

```text
准备一次路由、输入、输出及可选诊断 buffer → synchronize
→ 启动 profiler
→ HCCL device barrier
→ 对固定有限输入执行 exp × exp-repeat（整段仅一次）
→ 连续入队 warmup + iters 次 forward_prepared（内部 timer/trace 均关闭）
→ 如果有 --timer/--trace：最后追加一次诊断 forward（只有这次开打点）
→ HCCL device barrier → 一次 NPU synchronize
→ 停止并导出 profiler
→ 只读取 mega_moe_urma_kernel 的设备耗时
→ 丢弃 warmup 和末尾诊断，汇总全部 iters 个正式样本
```

两个 barrier、Exp、独立清零 kernel 和 Host 下发时间不计入目标 kernel 耗时。profiler 必须先启动再执行 guard，避免启动开销落在 Exp 与目标 kernel 之间。连续循环中没有 Host barrier、synchronize、D2H、读取耗时或打印，也不逐轮调用 profiler.step。只在整段结束后 step/export 一次。

profiler 使用 `Level0 + AiCoreNone + Text`，关闭 shapes、memory、stack、FLOPs 和 modules 采集；接口见 [Ascend profiler](https://github.com/Ascend/pytorch/blob/master/torch_npu/profiler/profiler.py) 和 [experimental_config](https://github.com/Ascend/pytorch/blob/master/torch_npu/profiler/experimental_config.py)。正式样本无内部 timer/trace，但仍可能有 profiler 采集成本。

`--trace` 仍隐含 `--timer`。两者共用末尾的**一次**额外诊断，不分别重跑；只导出这个末尾调用的原始 timer、逐事件 trace 和 allrank.json。其设备耗时单独保存，不进入正式均值。带宽来自该诊断调用，开启 trace 时也包含它对诊断流水的影响。profiler 自己导出的任务时间线保留整个采集块，用于复核所有正式轮次。

这减少 Host 启动和排队时机带来的 rank 错位，不保证消除硬件、频率、通信路径或实际计算造成的快慢差，也不会强行让各轮 kernel 同时开始。仍保留各 rank 样本与 rank-min/max，不能把它们抹平当作性能提升。

## 通信域和 stream 顺序

控制面继续使用原 StoreGroup。本 benchmark 额外创建 HCCL ProcessGroup，其 rendezvous 使用同一个 TCPStore 下本次随机 session 的独立 `urma-benchmark-hccl/` PrefixStore，不新增监听端口、不清空其他作业的通信域。benchmark 开始时若已有 torch.distributed 默认 ProcessGroup，则明确报错，避免接管别人的通信域。

HCCL barrier 是设备集合通信，不以 StoreGroup Host barrier 替代。guard 在调用流运行；现有 `UrmaBuffer._enqueue_forward` 的两个依赖保证：

```text
caller: HCCL barrier → exp
owner:  wait_stream(caller) → native URMA forward
caller: wait_stream(owner)
```

因此 owner 专用 stream 不能越过 exp 提前执行。stream 顺序所需的依赖仍保留，取消的是 Event 耗时测量。每轮 kernel 沿用原有输入就绪、结果完成协议。正常退出只销毁本 benchmark 创建的 HCCL 组；失败按原作业 abort/启动器清理流程退出，不重置设备、不重启服务。HCCL 初始化/集合通信超时由 `--launch-timeout` 控制，默认 300 秒；环境沿用当前作业的 CANN/HCCL 设置，脚本不改 `HCCL_IF_IP`、TLS 或驱动配置。

## 参数与内存

默认与参考一致：`--exp-shape 100000 20000 --exp-repeat 50`。两个 FP32 buffer 合计 **16,000,000,000 字节，约 14.90 GiB/rank**，另加已有权重、算子 workspace 和输入。始终对同一份零输入做 exp 并写到另一个 buffer，避免原地反复 exp 溢出。

输出日志和每 rank 的 `manifest.json` 记录 shape、repeat、分配字节数及计时范围；分配失败在所有 rank 进入 HCCL 初始化前通过 Store 传播错误。

若卡上余量不足，可以缩小 guard，例如：

```bash
# 保留原 torchrun、环境和其他参数，在 benchmark 参数中增加：
--exp-shape 10000 20000 --exp-repeat 50
```

该示例双 buffer 约 1.49 GiB/rank；缩小后应结合 trace 和多轮稳定性判断是否给 Host 留出了足够排队时间。`--exp-repeat 0` 完全不分配 exp buffer，只保留 HCCL device barrier，用于对照；它不保证消除 Host 下发造成的空隙。所有 rank 必须传入相同参数。

## 统计与原始数据

每种 BS 单独一个连续 block。设 `t[r,i]` 为 rank r 第 i 个正式 kernel 的 profiler duration，打屏：

```text
[URMA PROFILE] BS=4096 cv_rows=256 round-max-mean=...us round-min-mean=...us rounds=20 (device kernel; timer/trace off)
```

- `round-max-mean = mean_i(max_r(t[r,i]))`。
- `round-min-mean = mean_i(min_r(t[r,i]))`。
- 两者都使用全部正式 `iters` 轮，不再截取后半轮；也不是先对每个 rank 求均值再取最大/最小。
- 按同一 rank/stream 的执行次序匹配轮次，不跨 rank 相减时间戳。设备时钟不需要共同零点。

只接收目标 `mega_moe_urma_kernel` 的记录，拒绝数量不符、重复记录、多个设备/stream 和无效 duration，避免将 HCCL、Exp 或其它 MegaMoE 实现混入。原始 `kernel_details*.csv` 位于每 rank/BS 的 `profile/`，`kernel_samples.json` 保留逐条出处、warmup、正式轮和诊断样本；`summary.json` 的 `profile` 保存各 rank 正式轮次以及每轮最大/最小值。

这里测得的是融合 kernel 本身，包含 Pack、Dispatch、GMM、激活、Combine、Reduce 及 kernel 内的等待；不含路由准备和 kernel 外的清零/stream 等待。它与历史 Event 区间不是同一口径。CPU 调度与解析测试不等价于 NPU 实测，拉齐效果仍需目标机验证。
