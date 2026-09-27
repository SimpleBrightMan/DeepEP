# Elastic dispatch / combine 接口与限制

`ElasticBuffer.dispatch` 先执行 Notify 生成路由元数据，再搬运 payload；
`ElasticBuffer.combine` 使用同一个 `EPHandle` 将专家输出归约回源 rank。
本文件统一说明这三个 kernel，不再提供独立 Notify 测试入口。

Python 调用流程集中在 `deep_ep/buffers/elastic.py`：公开 `dispatch()`、`combine()`
方法直接执行对应流程，私有 `_notify()` 负责 fresh 路由元数据。
三者的空间布局和 size hint 统一放在同目录的 `_elastic_layout.py`；
`deep_ep/_runtime.py`、`_native.py` 继续提供 V1/V2 共用的资源与原生扩展加载功能。

## 支持范围

| 项目 | 当前约束 |
| --- | --- |
| 环境 | Ascend950、CANN 9.2、`dependencies.lock.json` 固定的 SHMEM SDK |
| EP | dispatch / combine：2、4、8、16、32、64、128；内部 Notify 上限 256 不代表完整链路支持 EP256 |
| 拓扑 | 所有 rank 的 SHMEM 工作区可通过超节点 MTE 映射；连续 8 rank 为一个逻辑 server，可跨主机 |
| Shape | H=7168，1≤T≤10240，1≤K≤16，EP≤E≤1024，E 能被 EP 整除 |
| 各 rank 配置 | T/K/E、dtype、weights 选项、fresh/cached 模式、容量与调用顺序一致 |
| 编码 | `EP*EP*T*K < 2**31`；dispatch 的二次幂地址步长还须通过编码检查，例如 EP128/K16 不支持 T=10240 |
| Tensor | 同一 NPU、连续存储、无 autograd |
| 布局 | expanded，`do_expand=None/True`、`expert_alignment=None/1`，不支持普通去重布局或 padding |

计数读取和容量检查包含 CPU 同步，`do_cpu_sync=None/True`。
`async_with_compute_stream=True` 返回供消费者等待的事件，不表示调用全程没有 Host 等待。
支持 `previous_event`；不支持 epilogue event、stats、列主序 scales 和
`allocate_on_comm_stream=True`。

## Dispatch

输入 `x` 为 BF16 `[T,7168]`，或 `(FP8 E4M3FN [T,7168], FP32 scales [T,56])`。
`topk_idx` 为 int64 `[T,K]`，可选 `topk_weights` 为 FP32 `[T,K]`。
无效 expert ID（负数或 ≥E）不产生输出；重复 expert 的每个槽位均保留独立输出行。

返回 `(recv_x, None, recv_weights, handle, event)`，按 expert/source/token/slot 顺序展开。
`recv_x` 为 `[R,7168]`，FP8 时为 `(recv_x, recv_scales=[R,56])`；
weights 为 `[R]` 或 None。R=0 时返回合法空 Tensor。
hidden/scales 按位搬运，不做量化；weights 独立搬运，不乘入 hidden。

fresh 调用执行 Notify（chunk=256、32 AIV）；cached 调用传 `handle=handle` 并省略
`topk_idx`，复用路由但可以更换 payload 和 weights。T/K/E 与路由不能改变。
默认复制输入索引；`do_handle_copy=False` 时必须保持原始索引不变。
BF16 `num_compute_units=0/32`，FP8 `0/64`；`num_qps=0/2`，
构造参数 `num_allocated_qps=0/6`，0 表示采用固定配置。

## Combine

输入为 BF16 `[handle.num_expanded_tokens,7168]`，专家内行顺序与 dispatch 输出一致。
返回 `(out, None, event)`，其中 out 为 BF16 `[T,7168]`。
空专家卡输入 `[0,7168]` 仍参与通信；FP8 dispatch 后可传入 BF16 专家计算结果。

`topk_weights=None`、`bias=None`；当前不实现独立权重回传，也不在 combine 内对 hidden 乘权。
`num_compute_units=0/64`、`num_qps=0/2`。同一个 handle 可重复 combine 或 cached dispatch。
server 内 FP32 累加后写 BF16 partial，source 再将各 server partial 转成 FP32 累加、写 BF16；
因此与一次性全 FP32 求和可能有舍入差异。

回传归约结果前会等待该 chunk 的归约完成，包括全部贡献都来自同一卡的情况。
`chunk_masks` 仅表示远端 gather 参与者；mask 为 0 时通过 `forward_list` 判断本地额外贡献，
不会因此跳过必要的归约等待。跨 server 的判断在进入 chunk 时计算一次，不在轮询中重复扫描。

```python
from deep_ep import ElasticBuffer

# 先设置 NPU 设备和 DEEPEP_SHMEM_ENDPOINT。
buffer = ElasticBuffer(rank=rank, world_size=world_size,
                       num_max_tokens_per_rank=1024, hidden=7168,
                       num_topk=8, explicitly_destroy=True)
try:
    recv, _, weights, handle, event = buffer.dispatch(
        x, topk_idx=topk_idx, num_experts=1024, do_expand=True)
    event.current_stream_wait()
    expert_out = run_experts(recv)  # BF16，保持专家内行顺序
    out, _, event = buffer.combine(expert_out, handle)
    event.current_stream_wait()
finally:
    buffer.destroy()
```

## EPHandle 与空间

Notify 产生 `server_mask`、`dst`、`forward_list/counts`、`backward_list/counts`、
`chunk_ranges/masks`、`rank_rows`、`gather_rows`、`expert_totals` 及每专家计数/前缀和。
`forward_list` 为 `[EP,T*K,6] int32`，`backward_list` 为 `[min(EP,8),EP*T*K,3] int64`；
只消费 counts 指定的有效前缀。backward 第三列低 32 位是 gather row，
高 32 位是 chunk 尾标记 `source*chunks+chunk+1`。
`dst` 使用 `rank*address_stride+row` 编码，转发路由按位取反，INT32_MIN 表示无效；
dispatch 单独生成二次幂步长的 `dispatch_dst`，保留原始 Notify 编码。

handle 绑定创建它的 buffer，禁止修改其内部 Tensor；完整 dispatch 返回
`metadata_only=False`、`metadata_version=1`。普通去重布局字段未生成，不能把 `dst`
当成 `dst_buffer_slot_idx`。持久元数据和返回 Tensor 拥有独立存储，下次调用或 destroy 不覆盖它们。

ElasticBuffer 构造时初始化一个 MTE+URMA SHMEM Session 并分配对称工作区，combine 复用它。
前半区用于 Notify/dispatch，后半区用于 combine；显式 `num_bytes` 必须包含两半，
各 rank 相同、2 MiB 对齐、总量不超过 32 GiB。`get_buffer_size_hint` 已包含两半，
默认接收/gather 行预算为 `min(EP,4)*T*K`；严重热点需要更大的显式容量。
真实行数超过容量时在 payload launch 前报错，不会自动扩容。
SHMEM 之外还会分配元数据和输出 Tensor，因此 hint 不代表全部显存用量。

当前尚未实现按执行阶段复用 SHMEM 数据区的逻辑：dispatch/combine 使用固定的独立半区，
combine 的专家输入和中间归约区也分别保留，尚未复用这些区域来降低峰值空间。
因此满足上述 Shape 范围不代表任意路由均可运行，实际接收/gather 行数还必须满足容量限制。
例如 EP64、T=1024、K=16、E=512、H=7168、BF16，所有槽位都指向同一专家时，
目标 rank 展开为 1,048,576 行；按当前布局，每个 rank 的 Session 需要 56.25 GiB，
超过 32 GiB 上限。该极端热点场景暂不纳入 combine 用例，待空间复用优化后再验收。

Host/Device 共用 `tiling.hpp` 计算布局，数据区按 512B 对齐。
算子内部保留原有写入归属、UB 分区、事件顺序与 collective 参与范围。
一个进程同时只允许一个 Session；所有 rank 按相同顺序调用并显式 `destroy()`。
直接 `rank/world_size` 模式无需 Gloo/HCCL 进程组，仅检查本 rank；调用者负责跨 rank 配置一致。
传入 `group` 的原接口仍保留跨 rank 检查。

## 编译与运行

先加载服务器上的 CANN/SHMEM 环境；SDK 必须包含 Device 私有头文件，详见 [构建说明](build.md)。

```bash
python -m pip install -r requirements-build.txt
export DEEPEP_NPU_ARCH=Ascend950
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:8900
python tests/elastic/test_dispatch.py --ep-size 8 --nproc-per-node 8 \
  --tokens 257 --topk 6 --experts 512 --dtype both --route all --repeat 3 \
  --output /tmp/dispatch-check-001
python tests/elastic/test_combine.py --ep-size 8 --nproc-per-node 8 \
  --tokens 257 --topk 6 --experts 512 --route all --repeat 3 \
  --output /tmp/combine-check-001
```

多机从共享目录编译一次，各 Python 环境安装相同 wheel。以下命令在四台主机分别执行，
`NODE_RANK` 为 0..3，`MASTER_ADDR` 为首台可达 IP，各 rank 使用同一个空闲 SHMEM endpoint：

```bash
export DEEPEP_SHMEM_ENDPOINT="tcp://${MASTER_ADDR}:8900"
python tests/elastic/test_dispatch.py \
  --ep-size 32 --nproc-per-node 8 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port 8899 \
  --tokens 257 --topk 6 --experts 512 --dtype both --route all --repeat 3 \
  --output "/tmp/dispatch-ep32-node${NODE_RANK}-001"
```

combine 换成 `test_combine.py`、去掉 `--dtype`、更换输出目录即可。

EP64（8 台机器、每台 8 卡）的完整 combine 闭环示例，在各节点分别设置 `NODE_RANK=0..7`，
`MASTER_ADDR` 使用首台主机的可达 IP，并在所有节点启动同一条命令：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/elastic/test_combine.py \
  --ep-size 64 --nproc-per-node 8 --node-rank "$NODE_RANK" \
  --shmem-ip-port "tcp://${MASTER_ADDR}:19091" \
  --tokens 1024 --topk 16 --experts 512 --route all --repeat 3 \
  --output "/tmp/combine-ep64-node${NODE_RANK}-$(date +%s)"
```

当前 combine 的 `all` 不含极端热点，BF16 和 H=7168 固定，无需传 `--dtype` 或 `--hidden`。
两个测试逐 rank 输出阶段日志，包含路由、轮次、累计耗时和距上一条日志的间隔。
`dispatch.enter` / `combine.enter` 表示进入公开接口（包含 Host 准备和 kernel 提交）；
`return` 仅表示接口返回，异步任务可能仍在执行；`readback.done` 表示对应输出已完成设备回读，
随后 `precision.passed` 才表示比较通过。这些间隔包含 CPU 工作与等待，不能当作 kernel 性能数据。
异常会在 `buffer.destroy` 前立即打印完整堆栈；若之后停在 `buffer.destroy.begin`，
说明清理尚未完成，应先查看之前的原始错误，并检查其他 rank 的日志。
测试直接用 Python 启动本机进程，不创建进程组；也兼容外部 launcher 的 rank 环境变量。
dispatch 输出目录须全新；共享目录中的多机输出路径应包含 node rank。
`--route empty` 表示所有索引无效，不产生有效专家行，但所有 rank 仍参加通信。

只保留两个算子测试：dispatch 检查 Notify 元数据及 BF16/FP8 payload；combine 检查完整归约闭环。
combine 的 `--route all` 覆盖 normal、duplicates、invalid、empty、local_duplicates、local_experts，
暂不提供 hotspot 选项。`local_duplicates` 将每个 token 的全部槽位指向同一卡的同一专家；
`local_experts` 将槽位分布到同一卡的不同专家（需要 E/EP≥2 才有不同专家），两者按 source/token
轮转目标 rank，避免全局热点；用于验证无远端 gather 时的归约等待。可单独指定 `--route` 运行。
dispatch 保留 hotspot 检查。两个用例覆盖 fresh/cached、weights、重复/无效/空路由、
事件依赖、连续调用和输出生命周期，
并用故意破坏的 golden 检查比较器。combine 用有界整数验证精确结果，不代表任意数值分布的精度资格。
大 shape 的全 rank CPU golden 和热点容量开销较大，这两个入口用于正确性验证。
CPU 检查可运行 `python -m pytest tests/elastic/test_dispatch.py tests/elastic/test_combine.py -q`；
combine 的 Host 回归使用 C++17 编译器（或已安装的 ziglang）执行实际 kernel 判断函数，
验证等待条件；不验证 AIV 并发、通信时序或设备性能。缺少编译器时该项跳过。
真实 kernel 的编译和精度必须在 Linux/CANN/NPU 上用上面两个入口验收。
