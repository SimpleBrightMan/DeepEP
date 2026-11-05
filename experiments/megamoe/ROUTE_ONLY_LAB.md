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
  --check-patterns random,hot,duplicates \
  --variant baseline --output-dir "$OUT"
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
