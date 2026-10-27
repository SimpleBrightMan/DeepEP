# 单 QP、RO 数据与 SO 完成通知

分支 `megamoe-urma-unitflag-so`，基于方案2
`megamoe-urma-gmm2-unitflag-reorder` 的 `0e36097`。

保留 GMM1/GMM2 连续 GM 输出、UnitFlag、共用计算上下文及下一 tile 提交后发布前一 tile。
本次只改通信、完整行重排的分工和完成计时。EP4/8/16 的通信核分别为3/7/15个；
剩余 AIV1 继续执行 SwiGLU、输入 scale gather 和静态 Pack，不再承担 Reorder。

初版相关 CPU 回归分组执行共700项通过，覆盖实际通信循环、异步 RO/SO 模型、完整行 DMA、
路由/Reduce 容量和 timer 解析。未进行 CANN 编译或 NPU 运行，尚无设备精度及性能结论。
后续 self Reorder 前移、Dispatch 每 CV quiet 和片段级提前 Reorder 修改
按要求未编译、未运行测试；
上述初版回归结果不代表本次修改已验证。

## 执行顺序

```text
Host stream: 清本次接收 CV / finished flags → kernel
通信 AIV1: MTE PublishStart
全部 AIV: Pack 前两轮 → 本地启动 join

每个通信 AIV1 唯一对应 peer，使用 QP0:
    等对端 Start（对端已清好接收 flags）
    for Dispatch CV:
        等本地 packReady
        RO PUT(X) → RO PUT(scale) → SO atomic_add(ready, epoch)
        准备本核两轮以后的静态 Pack 配额
        quiet(QP0)，记录 DispatchQuiet 和该 peer 最新完成时刻
        空 CV 只做静态 Pack 配额，不调用 quiet
    本地确认所有 Dispatch 已完成 quiet、所有 Pack 配额已完成

    while 本 peer 还有未发送的 Combine CV，或本核还有 self CV:
        从头扫描未提交 CV，再扫描该 CV 尚未搬运的行片段
        对应 M256 组的全部 N tiles 已完成，就提前重排该片段的完整H行
        未完成的片段直接跳过，不挡住后面的就绪片段
        同一 CV 本轮所有就绪片段共用双64KiB槽，最后统一等待本批写出
        记录已搬片段；如果该 CV 全部片段已搬完:
            RO PUT(返回数据) → SO atomic_add(CV ready, epoch)
        本 peer 所有 Combine 提交完后，仅执行一次:
            SO atomic_add(finished, epoch)
        同样提前处理本核静态分配的 self CV 片段，直接写 returnY
        self CV 全部片段写完后才发布本地 ready
        未就绪的 self CV 留待下一轮扫描，不进入最终 quiet
        finished 已提交且本核所有 self CV 已重排后，仅执行一次:
            quiet(QP0)
            记录 CombineSendComplete（timer/trace开启时）
        观察入站 finished；无进展时显式退避

    本核 self 重排与入站完成观察结束后退出

AIV0 / 非通信 AIV1: 原六槽 Reduce，就绪扫描与缓存不变
所有本地引擎最终 join 后 kernel 返回
```

没有 Dispatch 末尾提前 Combine，没有第二个 QP，也没有新增全卡同步。
Dispatch 每个非空 CV 都在下一 CV 提交前 quiet，保证每个发送 rank→peer 的 QP
最多一个未完成的数据 CV；多个发送 rank 仍可能同时访问同一目标。
本地 Dispatch join 前，各通信核已完成各自全部 Dispatch quiet。
Combine 保持逐 CV RO/SO 提交、末尾一次 quiet；self 重排不会被该 quiet 阻塞，
远端 finished 通知也不依赖 self 是否已就绪。
SHMEM 的队列信用不足时仍可能内部等待 CQ，不能把 NBI 理解为永不阻塞。

## 数据与完成保证

- 数据 PUT 显式使用 `ACLSHMEMX_UDMA_OP_CONFIG_RO`；不能使用默认 NO。
- 通知用 `aclshmemx_udma_atomic_add<uint64_t, ACLSHMEMX_UDMA_OP_CONFIG_SO>`，
  此 SDK 接口使用 QP0。SO 等同 PE/QP 上所有先行 RO/SO 完成后才更新对端标志。
  不用内部含 GET/quiet 的 atomic_set，也不用无序的普通 PUT(flag)。
- 每次调用在同一 Host stream 清零接收方 Dispatch/Combine CV 和 finished 槽，
  然后才发 Start。发送方等到 Start 才能写入，因此 add(epoch) 可以恢复精确 epoch，
  不依赖上一轮此 CV 是否存在。Start 槽保持单调，不清零。
- Pack / Reorder 的本地 MTE3 完成后才下发网络 PUT。发送 GM 每行只写一次，
  在最终 quiet 之前不复用；复用的是 UB 槽，不是在途 PUT 的 GM 源地址。
- `gmm2Done[group]` 改为全部 N tile 的 FIX 完成计数，只代表连续计算临时区完成。
  CPU 预计算 peer/CV→完整行区间表，通信核不再遍历全部专家/来源寻找地址。
- 提前 Reorder 沿用该区间表，不修改路由、计算 M-group 或 CV 的 PUT 粒度。
  每个 owner 的 remote/self 各用一个8KiB UB位图，CV保存已搬片段数，
  搬过的片段直接跳过。位图只在本核访问，不增加 GM 进度表或原子操作。
  Pack 结束后复用其 UB，位图与双64KiB DMA槽、CV计数互不重叠；
  仍满足192KiB UB要求，不依赖均衡路由。Host检查每peer最多65536个片段，
  该上限覆盖BS8192/TopK8每个来源的全部展开行。
- 每轮只对实际就绪片段启动DMA，一批片段共用槽位和事件；批末等MTE3完成。
  不对每个片段单独排空。CV部分完成时只更新本核位图/计数，
  只有所有片段完成写出后才下发一次完整CV PUT或self ready。
- 自卡结果按 `self_cv % (world-1)` 静态分给通信核；直接重排到最终 returnY，
  不发网络 PUT。Reduce 的自卡依赖也改为 self CV ready，不能仅看 GMM2 完成。
- 空 peer 仍完成 Pack 配额、发送 finished 并最终 quiet，也参与自卡任务与入站观察。
  入站计数达到 world 时，保留原 Reduce 免读逐 token flag 的路径。

## Timer 含义

每个通信 AIV1 最后只有一条 `CombineSendComplete`（raw ID42），覆盖最终 quiet。
其末端是本核所属 peer 的全部发送完成时刻，位于入站完成等待之前。
它在非 timer 模式下也必须 quiet，打点本身不增加 quiet 或 PIPE_ALL。
完成时刻是 CPU/Scalar 观察到 quiet 返回的时刻，不是网络上最后一个字节的硬件时间戳。

`CombinePut` / `DispatchPutX` 仍只记录提交调用。aggregate record word5=1 标记此新语义，
word7 记录 Dispatch 最新 CV quiet 完成时间或 Combine 最终发送完成时间；
word0..4 保留提交统计，Dispatch 的 Pack/quiet 不混入提交耗时。

- Combine 完成包络：本卡最早 Combine PUT → 最晚对应 peer 的 CombineSendComplete。
- 整体通信包络：本卡最早 Dispatch PUT → 最晚对应 peer 的 CombineSendComplete。
- Dispatch 完成包络：本卡最早 Dispatch PUT → 最晚对应 peer 的 DispatchQuiet 结束。
  aggregate timer 读取 word7 恢复完成包络及有效带宽，包含中间 Pack/quiet 间隙。
- Combine 最终 quiet 在本核 self 重排完成后观察，完成时刻可能晚于实际网络完成；
  不应把该条缩短直接解释为网络变快。
- 接收端仍按 CV flag 执行计算/Reduce，Reduce 泳道不添加完成采样或细粒度打点。
- 不同 rank 时钟不直接相减，包络包含内部等待和重排间隙。
- `Gmm2Reorder` 每条代表某个CV在本轮扫描中实际搬运的一批就绪片段；
  同一CV可能有多条，rows/bytes只统计该批实际搬运量，累计不重复。
  批次可能包含不连续目标行，因此不填连续row起点；只有单一计算组时填写fragment。

`urma_trace_to_text.py` 输出 `[COMPLETION]` 汇总；各 peer 的完成时刻也保留在详细事件中。
旧网页若只按 PUT 条宽统计带宽会低估完成时间；请先按新完成条审阅，不沿用旧提交口径。

## 编译和 EP16 诊断

先进入仓库，激活 Python 并 source CANN。请只在分配给本任务的16张空闲卡上执行。

```bash
git fetch origin
git switch megamoe-urma-unitflag-so
git pull --ff-only
export PYTHON="$(command -v python3)"
export DEEPEP_BUILD_MEGAMOE=ON
export DEEPEP_MEGAMOE_TIMER=ON
bash experiments/megamoe/scripts/build.sh

export SHMEM_ROOT="$PWD/third-party/shmem/install/shmem"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib${CONDA_PREFIX:+:$CONDA_PREFIX/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export DEEPEP_SHMEM_ENDPOINT="tcp://127.0.0.1:$("$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=16 \
  -m deep_ep_experimental.megamoe.urma_benchmark \
  --bs-values 2048,4096,8192 --routing balanced --cv-rows 256 \
  --gmm1-ahead-mgroups 12 --warmup 10 --iters 20 --timer --trace \
  --output-dir "benchmark-runs/urma-unitflag-so-$(date +%Y%m%d_%H%M%S)"
```

正式 profiler 轮次保持内部 timer/trace 关闭；最后另执行一次诊断并导出 allrank JSON。
EP4/8 改可见设备列表和 `--nproc-per-node`。精度使用既有 `--golden-dir` / `--accuracy-only`
入口；没有指定 Golden 的性能运行不代表精度通过。本次不预先宣称 CANN/NPU 验证通过。
