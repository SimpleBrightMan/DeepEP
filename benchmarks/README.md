# EP benchmark

提供配置检查、确定性负载规划、完成边界计时引擎、逐轮 rank MAX 聚合和 JSON/CSV 结果记录。本通用框架当前没有原生测量适配器或设备实测数据；非 dry-run 命令会在设备初始化前明确失败，不以参考实现或模拟耗时代替通信测量。

LL 已有独立的公开 API ST 性能入口，见 [LL ST](../tests/elastic/test_ep_low_latency.py)：默认执行正确性及 kernel 耗时测量。它不使用本通用框架，且当前尚无设备实测结果。

## 无设备使用

从仓库根目录运行，Python 3.11+，无需 Torch、NumPy 或 NPU：

```bash
python -m benchmarks.bench_ep --help
python -m benchmarks.bench_ep --list-scenarios
python -m benchmarks.bench_ep --dry-run --preset smoke
python -m benchmarks.bench_ep --dry-run --preset upstream-bf16
python -m benchmarks.bench_ep --dry-run --preset upstream-fp8
python -m benchmarks.bench_ep --dry-run --operation roundtrip --tokens-per-rank 16,0 --route duplicate --output-dir benchmark-runs/plan-001
```

`--config <file.json>` 与 `--preset` 二选一，显式 CLI 字段覆盖所选配置，其余使用默认值。`--tokens-per-rank` 必须逐一列出全部 rank 的实际 token 数。权重开关为 `--with-weights` / `--no-with-weights`。已有输出目录不会覆盖。

dry-run 输出 `status=planned`，仅保存 `config.json` 和 `plan.json`，不生成时延、带宽或 CSV 性能摘要。`upstream-*` 预设是固定上游典型规模的**规划样本**，不代表全功能覆盖或性能基线。

## 测量入口

原生后端及其私有适配器实现并通过功能资格后，可使用以下命令形式：

```bash
torchrun --standalone --nproc-per-node=2 -m benchmarks.bench_ep \
  --config benchmarks/configs/smoke.json --revision <full-source-commit-sha> \
  --output-dir benchmark-runs/run-001
```

当前执行会报 `native benchmark adapter is not implemented` 并返回 2。不提供绕过能力门禁或模拟测量的参数。退出码 0 表示规划成功或全部测量步骤完成；配置、能力、正确性和运行时错误返回非零。详细错误只写本地 stderr，结构化失败输出不携带异常文本。

## 文件职责

- `bench_ep.py`：CLI、预设、能力门禁和会话生命周期。
- `_config.py`：严格配置、输入/路由公式及逻辑字节数。
- `_backend.py`：固定第一方适配器入口与私有协议；不实现通信。
- `_utils.py`：预热、完成边界计时、正确性投票、统计和结果持久化。
- `configs/`：smoke、上游典型 BF16/FP8 规模。
- `results/`：版本化字段说明与 CSV 模板，不含实测数据。

自动调优、V1 低时延专用接口、异步重叠吞吐和 MegaMoE 融合流程不在本入口范围内。

[详细设计](../docs/benchmark-design.md) · [测量口径](../docs/performance.md) · [CPU 协议测试](../tests/test_benchmark.py)
