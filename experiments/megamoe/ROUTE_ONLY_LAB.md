# RoutePrepare-only 实验

这是独立实验分支，不是完整 MegaMoE。两个 kernel 入口均在现有
`PrepareDeviceRoutes` 后写出 trace header 并直接返回；不运行 Pack、GMM、
激活、Dispatch、Combine、Reduce。原有工作区容量与 Host forward 参数检查保留。

`route_only_benchmark` 拒绝未标记 `urma_route_only=True` 的生产 wheel。
性能仍使用原来的 `torch_npu.profiler` 核耗时；默认 warmup10/iters20，
每次调用前 Device all-reduce，整个测量块前 HCCL barrier + 50次 exp。
all-reduce/exp/工作区清零/输入准备都不属于目标核的计时。

示例（单机EP8；多机EP16需由受保护启动脚本设置两机torchrun参数）：

```bash
"$PYTHON" -m torch.distributed.run --nnodes=1 --nproc-per-node=8 \
  --master-addr=127.0.0.1 --master-port=29681 \
  -m deep_ep_experimental.megamoe.route_only_benchmark \
  --bs-values 8192 --hidden 5120 --intermediate 2304 --topk 6 \
  --routing random-balanced --ids-dtype int64 --cv-rows 256 \
  --warmup 10 --iters 20 --exp-repeat 50 --timer \
  --check-patterns random,hot,duplicates --check-plan \
  --variant optimized --output-dir "$OUT"
```

多机加 `--allow-multi-host`。支持 `--bs-values 1024,8192,16384`，但工作区
按最大BS的原算子最坏容量分配，尚未删减未使用的GMM缓冲。若最大容量与
默认exp缓冲合计超出本机HBM，应分开BS运行；不要让不同variant使用不同
exp配置而直接比较结果。

每个BS在profile结束后严格验证：专家计数和前缀、稳定排序order、inverse、
Reduce dependency、实际接收行数和M-group数量。额外pattern检查不计入性能。
`hot` 将全部数据路由到rank0的专家，`duplicates` 允许重复专家号，不依赖均衡。
这只验证路由，不声称FFN输出精度通过；返回的BF16输出未计算，不应使用。

`summary.json`：

```text
scope: route_prepare_only
world, variant, config, launch_guard, passed
cases[]:
  bs, routing
  profile.rank_max_mean_us / rank_min_mean_us / raw_per_rank_rounds_us
  route_validation.passed / per_rank[]
  extra_validation[]
  coarse_trace (启用--timer时)
  passed
```

coarse是最后一次独立诊断调用，仍在同一个profile上下文内，不计入正式均值。

## 2026-11-05 优化

参考主仓 [89e1939 的 Reg 路由压缩代码](https://github.com/lueelu/DeepEP/blob/89e19399260c16c483b3614f57d2e3c90eb9fc2e/experiments/megamoe/csrc/kernels/mega_moe/stage/mega_moe_route_compact.hpp)。
该上游代码是带条件的快路径；这里复用向量 Compare/Compact/StoreUnAlign 写法，
改为总长度不超过本核输入条数的 CSR，不继承固定专家槽或 topk 去重假设。

最终实现：

- 每个AIV仅从GM载入一次topk；单VF分桶，CSR及计数保留在高UB。
- power-of-two CV 使用向量生成order/inverse/依赖，peer/localExpert用整数二分，
  不使用浮点倒数；其他CV保留原Scalar映射路径。
- order每专家UB源按32B对齐，所有源保持不变，批量DMA后统一等待。
- 56个AIV按实际M-group/CV的连续区间静态建表；先计长度、前缀确定独占写区，
  不使用抢任务或atomic append。各表经UB缓冲DMA发布，避免多核Scalar假共享。
- 9次本卡同步；保留入口epoch和计数交换协议。
- Config每次完整读取前覆盖全部64B缓存行，再进行普通volatile读取；
  保留3次reload，没有靠删除一致性步骤获取收益。

并行计划在`route_info_`末尾增加GM scratch，因此必须重新构建完整native模块，
不能只替换kernel `.so`。最大UB地址为182816B，小于192KiB。
unitflag的描述符路径在条件编译中保留串行；本分支NPU实测对象是commfirst。

同一单机EP8、BS8192、H5120/I2304/E384/K6、CV256、int64、random-balanced，
warmup10/iters20、Exp50与每次Device all-reduce：

| 实现 | round-max-mean (µs) |
|---|---:|
| 原始RoutePrepare-only | 568.814 |
| 基线复测 | 568.076 |
| 保留CSR与批量映射写出 | 435.548 |
| CSR + 单VF分桶 | 387.098 |
| 单VF + 描述符DMA | 359.365 |
| 再向量化映射 | 337.833 |
| 去掉映射阶段Scalar辅助表 | 318.242 |
| 并行建表 + 映射v2 | 297.209 |
| 并行建表 + 缓存读取Config（映射v1） | 204.611 |
| 最终组合 | 186.513 |
| 最终组合复测 | 186.555 |

这是独立RoutePrepare核的收益，不能直接当作完整MegaMoE的加速比。
`23b6fd8`保留了本实验框架的原始算法基线，便于在同机重新对照。

`--check-plan` 在profile之后逐字段检查Fragment/Segment、GMM tileBegin、
Send/Recv Chunk、Link、ComputeBatch/Task、Pack的全部16槽和动态Config。
它是实验校验功能，不参与生产路由热路径。没有执行FFN，不声明FFN精度通过。

最终组合另外通过 EP8 下 BS16384/TopK8/int64/CV256，以及
BS1/1024/16384/TopK8/int32/CV384；每组含 random、hot、duplicates
校验，验证空接收rank、重复路由、最大UB容量和非2次幂CV回退。
实验wheel会拒绝普通`urma_benchmark`入口，避免把未执行FFN的结果误作完整算子结果。
