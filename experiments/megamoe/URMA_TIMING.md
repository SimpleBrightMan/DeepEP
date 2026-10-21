# URMA benchmark 的起跑对齐与连续计时

`urma_benchmark.py` 参考 [mega_test 的计时方法](https://gitcode.com/SimpleBright_Man/mega_test/blob/main/benchmark/timing.py)，将 **CACHED_FORWARD** 改为：

```text
准备路由、输入、输出和所有 Event
→ HCCL device barrier
→ 对固定有限输入执行 exp × exp-repeat（整段仅一次）
→ 连续入队 warmup + iters 次 forward_prepared
→ HCCL device barrier → 一次 NPU synchronize
→ 读取 Event、丢弃 warmup、汇总正式样本
```

两个 barrier 和 exp 都在每个 Event 区间之外，不计入融合算子耗时。连续循环中没有 Host barrier、synchronize、D2H、读取耗时或打印。第一次 guard 位于所有 Event 创建和路由准备之后，给 Host 连续下发留出设备执行时间。原有单独 timer 和 trace 诊断调用也各加一组同样的前置 guard，保持正式测量关闭 timer/trace；`--trace-include-waits` 的过滤逻辑不变。

这减少 Host 启动和排队时机带来的 rank 错位，不保证消除硬件、频率、通信路径或实际计算造成的快慢差，也不会强行让各轮 kernel 同时开始。仍保留各 rank 样本与 rank-min/max，不能把它们抹平当作性能提升。

## 通信域和 stream 顺序

控制面继续使用原 StoreGroup。本 benchmark 额外创建 HCCL ProcessGroup，其 rendezvous 使用同一个 TCPStore 下本次随机 session 的独立 `urma-benchmark-hccl/` PrefixStore，不新增监听端口、不清空其他作业的通信域。benchmark 开始时若已有 torch.distributed 默认 ProcessGroup，则明确报错，避免接管别人的通信域。

HCCL barrier 是设备集合通信，不以 StoreGroup Host barrier 替代。guard 在调用流运行；现有 `UrmaBuffer._enqueue_forward` 的两个依赖保证：

```text
caller: HCCL barrier → exp → begin Event
owner:  wait_stream(caller) → native URMA forward
caller: wait_stream(owner) → end Event
```

因此 owner 专用 stream 不能越过 exp 提前执行，Event 包含完整 native forward。每轮 kernel 仍沿用原有输入就绪、结果完成协议。正常退出只销毁本 benchmark 创建的 HCCL 组；失败按原作业 abort/启动器清理流程退出，不重置设备、不重启服务。HCCL 初始化/集合通信超时由 `--launch-timeout` 控制，默认 300 秒；环境沿用当前作业的 CANN/HCCL 设置，脚本不改 `HCCL_IF_IP`、TLS 或驱动配置。

## 参数与内存

默认与参考一致：`--exp-shape 100000 20000 --exp-repeat 50`。两个 FP32 buffer 合计 **16,000,000,000 字节，约 14.90 GiB/rank**，另加已有权重、算子 workspace 和输入。始终对同一份零输入做 exp 并写到另一个 buffer，避免原地反复 exp 溢出。

输出日志和每 rank 的 `manifest.json` 记录 shape、repeat、分配字节数及计时范围；分配失败在所有 rank 进入 HCCL 初始化前通过 Store 传播错误。

若卡上余量不足，可以缩小 guard，例如：

```bash
# 保留原 torchrun、环境和其他参数，在 benchmark 参数中增加：
--exp-shape 10000 20000 --exp-repeat 50
```

该示例双 buffer 约 1.49 GiB/rank；缩小后应结合 trace 和多轮稳定性判断是否给 Host 留出了足够排队时间。`--exp-repeat 0` 完全不分配 exp buffer，只保留 HCCL device barrier，用于对照；它不保证消除 Host 下发造成的空隙。所有 rank 必须传入相同参数。

## 不混淆三种结果

| 输出 | 实际范围 | 调度方式 |
| --- | --- | --- |
| `CACHED_FORWARD` | Event 测量的完整 native forward：Pack、Dispatch、GMM、激活、Combine、Reduce；路由已经准备 | 上述连续 block，guard 不计时 |
| `ROUTE_PREP` | 路由、D2H counts、Host 交换、描述符构建的 Host wall time | 保留原逐轮 Host 协调与同步 |
| `FULL_E2E` | 每轮重新准备路由，加完整 forward 和最终同步的 Host wall time | 保留原逐轮 Host 协调与同步，不称为连续设备计时 |

原有初始完整路径 warmup 保留；CACHED_FORWARD 另外有自身连续的 warmup，用来预热与正式测试相同的缓存路由调用。三个 BS 各自单独执行一个连续 block。

为保持历史结果可比，统计仍使用原来的 `tail_half_ceil`：先从连续执行中剔除 warmup，再从 `iters` 个正式样本取最后 `ceil(iters / 2)` 轮。日志 `rounds/recorded_rounds` 明确显示这一选择。`cached_forward.raw_per_rank_rounds_us` 另存全部正式样本，可离线按全部 iters 重算。它与参考项目的“全部 iters、融合 kernel 全 rank 均值”归约不是同一口径，不能直接混用。

本次只做 CPU 调度与参数回归；HCCL/exp 的实际设备对齐效果需用目标机实测确认。
