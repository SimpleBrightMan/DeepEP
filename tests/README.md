# 测试入口

Elastic 设备 ST 入口：

| 文件 | 内容 |
| --- | --- |
| `elastic/test_dispatch.py` | Notify 元数据、BF16/FP8 dispatch、weights、fresh/cached、路由和输出生命周期 |
| `elastic/test_combine.py` | dispatch → 专家输出 → BF16 combine 闭环、重复调用、事件依赖和精度 |
| `elastic/test_ep_low_latency.py` | LL 专用 dispatch/combine 精度、R/C、重复 combine 与完整调用性能测量 |

编译、单机和多机 Python 启动命令统一见 [Elastic 接口与限制](../docs/elastic.md)。
设备 ST 使用已安装包。原有 dispatch/combine ST 直接使用 SHMEM 初始化，不需要 Gloo/HCCL 进程组；LL ST 默认使用已有 HCCL group，支持 spawn 和 torchrun，详见 [LL ST 入口](elastic/test_ep_low_latency.py)（`--help` 查看参数）。
Notify 的元数据验证已并入 fresh dispatch；通用未来功能矩阵不作为当前 Elastic 的运行入口。

CPU 检查：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -m "not framework" -q
```

其中 dispatch 文件包含路由参考、地址范围、启动器和安装包导入的 CPU 检查；
combine 和 LL ST 是显式运行的 NPU 脚本，不会被普通 pytest 当作设备验证执行。
`elastic/test_ll_st_contract.py` 检查 LL ST 的启动器、参考数据及计时合同；
`elastic/test_low_latency_host.py` 检查 LL API Host 分支和计数通信。

仓库基础测试继续保留：`api/` 检查接口契约，`runtime/` 检查构建与资源生命周期，
`integration/` 检查集成链路，`legacy/` 保留尚未实现的 V1 设备资格入口。
`test_reference.py`、`test_precision.py` 等验证独立 CPU 参考和比较器；
`utils/` 是这些基础测试的公共辅助，不是新增的算子入口。

普通 pytest 不自动占用 NPU；CPU 通过不能代替真实 kernel 编译或设备精度结果。
完整构建安装流程见 [构建说明](../docs/build.md)。

LL ST 运行时必须通过命令行、JSON 或环境变量显式提供 master 地址、master 端口和 SHMEM endpoint；不再提供默认值或自动推导 SHMEM 端口。环境变量为 `MASTER_ADDR`、`MASTER_PORT`、`DEEPEP_SHMEM_ENDPOINT`，命令行/JSON 显式值优先。`--plan` 不要求完整网络配置。
示例配置中的 `xxx.xxx.xxx.xxx` 和 `xxxxx` 需替换后使用；JSON 的 `master_port`、`shmem_port` 应填写不带引号的整数。

FP8 ST 可通过 `--dtype fp8 --fp8-scale-dtype int32 float32 --do-cpu-sync 0 1` 同时覆盖两种 scales 格式和 CPU 同步开关；JSON 的 `matrix` 同样支持 `"fp8_scale_dtype": ["int32", "float32"]`。该选项默认 `int32`，仅对 FP8 生效。FP32 用例使用随源 rank、token 和分组变化的非单位 scales，按位核对 dispatch 搬运及行对应关系，专家参考计算使用收到的 scales 反量化后验证 combine。结果记录 scales 格式，性能仍分别测量 dispatch/combine kernel。


### Notify 与紧凑表 Event 计时

`tests/elastic/run_peer_table_check.py` 在原有测试参数上增加
`--kernel-profile --routing-profile`，会额外执行 fresh-handle 采样，
按 `--warmup` / `--iterations` 预热与记录，每次重新执行 Notify 和紧凑表构建。
这部分独立于复用 handle 的 dispatch 性能采样，不添加计时前全 rank barrier。
各 rank JSON 的 `routing_device_ms` 保存以下 Event 区间（毫秒）：

- `notify_device_ms`：Notify kernel、后置 barrier 与轮转/转发表准备、expert totals 保存、总时间（包含建表）。
- `prepare_device_ms`：Notify 内部的表清零、建表 kernel、末尾 Event 间隔、准备总时间；这是 Notify 总时间的子区间，不能重复相加。

建表 kernel 包括 peer-table、紧凑 forward plan 和原有 admission 同步；
不包含主机读取 admission 状态。Notify 和 prepare 均不包含 Python 分配与校验。
普通 API 不设置 `_routing_diagnostics`，不创建这些计时 Event。
运行 `dispatch_metrics.py <结果目录>` 后，`kernel-summary.json` 的 `routing`
按每次采样取各 rank 最大值，再计算均值与中位数。不同阶段的 rank 最大值不能直接相加作为端到端耗时。


### Peer-ready 持久 generation 协议

dispatch 不再逐次清零接收控制标志，
移除 payload kernel 的入口 `aclshmemx_barrier_all_vec`；地址发布的本地同步、
MTE 完成后再发布 generation 的顺序，以及出口完成协议保持不变。
64 位 generation 不复用，耗尽后要求重建 Session。

冷启动、布局变化、Notify/LL 工作区复用后的控制区初始化在 Host 侧执行；
缓存调用复用已初始化的控制区。每次调用均保留 stream drain + host 全 rank join，
保证上次输出拷贝和标志消费完成后才能启动下一轮，不支持跨轮重叠。
Host join 和冷初始化均在 dispatch Event 区间之外，因此 kernel Event 不能代表
包含这些同步成本的完整 API 时延。原 profile 的 entry_barrier 槽保留但不再执行设备全局同步。
