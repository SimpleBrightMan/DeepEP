# main 计算优化：完整 EP8 验收结果

测试日期：2026-10-10，CPU7 / Ascend950DT 八卡，zyx 环境。结论：五个 BS 均通过完整通信全量精度检查，三组交替性能测量均有正收益。BS4096 平均减少 **201.365 μs（4.519%）**。

## 测量口径

- 旧版：`codex/megamoe-main-baseline-ep8`，`a2714e029976334d1f90f3ebbcf7da2444c2dba4`。计算源码与原 main `8ffafeb` 相同，仅补本机 EP8 启动/构建支持。
- 新版：main `6b9624f487c855bebc04bbf99e31a97f60e10d7a`，含 GMM2 wave 流水复用、删除重复 FIX 排空、GMM1 异步完成通知与 wave 复用。本报告提交只增加文档，不改变被测代码。
- **完整通信**：Dispatch → GMM1 → SwiGLU → GMM2 → Combine → Unpermute；不启用 compute replay 或本地预置输入。
- EP8，每卡 BS；H=7168、专家中间维度=3072、专家数=384、TopK=6、共享专家=0；FP8 输入 / FP4 权重、balanced 路由、seed=20260922。
- CANN 9.2.0 B070，timestamp `20260921_000125261`；torch `2.12.0+cpu`、torch_npu `2.12.0.post2`；Python 3.12.14。
- 两版复用相同 pinned SHMEM `4dd80e2` 与 CATLASS `2b4bd7c`，分别构建独立 wheel，每次切版本显式重新安装并核对身份。
- 先运行两版五个 BS 的完整精度，再按旧1→新1→旧2→新2→旧3→新3交替性能测试。每轮启动前检查八卡没有其他 NPU 进程，结束后八卡无残留进程；未停止其他用户任务。
- warmup=10、iters=20；按既有规则取后10轮。每版每个 BS 共三组、30个选中轮次/每rank；所有组等权平均，不挑最好值。
- 采用 `instrument-megamoe-timer` 的独立采样规则：**性能采样 timer-off，另外采集 timer 并自动导出8个rank的JSON/CSV**。没有为了打点增加同步，本轮没有单独重测 timer-on/off 开销。
- profiler 为现有 Level1 / PipeUtilization 配置；两版相同。指标是 profiler 的 MegaMoE kernel Duration，不是 Python wall time。alignment Exp 前缀不计入该 kernel Duration。

## 性能结果

下面沿用现有日志 `MegaMoe E2E` 的 **rank-min-mean**：每轮取8个rank的最小kernel Duration，再对选中轮次求均值；不是完整调用的最慢rank口径。

| 每卡 BS | 旧版 μs | 优化版 μs | 减少 μs | 耗时降低 |
| --- | ---: | ---: | ---: | ---: |
| 32 | 877.645 | 842.393 | 35.252 | 4.017% |
| 96 | 912.328 | 875.788 | 36.540 | 4.005% |
| 256 | 967.723 | 929.505 | 38.219 | 3.949% |
| 1024 | 1286.250 | 1255.669 | 30.581 | 2.378% |
| 4096 | 4455.515 | 4254.149 | 201.365 | 4.519% |

同时核对 **rank-max-mean**：每轮取最慢rank的kernel Duration再平均。结论一致，不用rank-min隐藏慢rank。

| 每卡 BS | 旧版 rank-max μs | 优化版 rank-max μs | 耗时降低 |
| --- | ---: | ---: | ---: |
| 32 | 879.103 | 843.776 | 4.019% |
| 96 | 913.566 | 877.300 | 3.970% |
| 256 | 969.505 | 930.931 | 3.979% |
| 1024 | 1287.868 | 1257.489 | 2.359% |
| 4096 | 4458.474 | 4257.328 | 4.512% |

三组原始均值，单元格为「旧 / 新」，单位 μs，rank-min-mean：

| BS | 第1组 | 第2组 | 第3组 |
| --- | ---: | ---: | ---: |
| 32 | 876.819 / 842.086 | 878.479 / 842.826 | 877.637 / 842.267 |
| 96 | 912.508 / 875.060 | 912.132 / 877.272 | 912.342 / 875.032 |
| 256 | 967.348 / 928.783 | 968.323 / 929.137 | 967.498 / 930.594 |
| 1024 | 1287.267 / 1257.279 | 1289.101 / 1250.883 | 1282.382 / 1258.847 |
| 4096 | 4454.977 / 4254.538 | 4456.693 / 4254.035 | 4454.873 / 4253.875 |

BS1024 有约数微秒的组间波动，但三组成对收益都为正。BS4096 旧版组均值范围 4454.873–4456.693 μs，新版 4253.875–4254.538 μs。

## 正确性

两版各检查五个 BS、8个rank全部token，不是抽样；每版共44032个token。专家count与累计count均一致，CPU Golden mixed-error检查全部通过，bad_elements均为0。
两版各BS的误差汇总相同；这不是逐位一致性声明。

| BS | 两版结果 | 每版检查token总数 | 最大 mixed error |
| --- | --- | ---: | ---: |
| 32 | PASS | 256 | 0.00006103515625 |
| 96 | PASS | 768 | 0.00006103515625 |
| 256 | PASS | 2048 | 0.0001220703125 |
| 1024 | PASS | 8192 | 0.00006103515625 |
| 4096 | PASS | 32768 | 0.00006103515625 |

## 原始记录与复现

服务器汇总：`/home/zyx/main-e2e-ab-20261010-161930/`，包含 `manifest.json`、`comparison.json`、`comparison.txt`、8次启动前设备检查和各组日志。
完整逐rank profiler样本、精度报告和timer仍保留在以下运行目录。

旧版根目录 `/home/zyx/DeepEP-main-baseline/benchmark-runs/`：

- 精度：`ep8_20261010_161931_96f8742d`
- 性能1：`ep8_20261010_162847_04f18aa8`
- 性能2：`ep8_20261010_163630_b2e5f1f4`
- 性能3：`ep8_20261010_164355_d6744748`

新版根目录 `/home/zyx/DeepEP-main-validation/benchmark-runs/`：

- 精度：`ep8_20261010_162525_f3cb218b`
- 性能1：`ep8_20261010_163236_fe583dd2`
- 性能2：`ep8_20261010_164000_60c3e931`
- 性能3：`ep8_20261010_164809_ea688524`

每个运行目录内：`bs_<B>/rank_0/bs_<B>/profiling.json` 包含8个rank的选中样本，
`bs_<B>/bs_<B>/timer/` 保存8个rank的流水JSON/CSV；精度run对应 `accuracy.json`。

精确 wheel：

- 旧：`/home/zyx/DeepEP-main-baseline/dist/build-o5vqg_zb/ascend_deepep-0.1.0.dev5-cp312-cp312-linux_aarch64.whl`
  SHA256 `1e1eda56f0ba20c0adc7927ab0cf39900b6180d825eb80fc6baa0e90d4495a07`
- 新：`/home/zyx/DeepEP-main-validation/dist/build-8n73ysvf/ascend_deepep-0.1.0.dev5-cp312-cp312-linux_aarch64.whl`
  SHA256 `3503b9d3a89655b4ee502b56c46d0fe110edd2493d7880bbc74548a98f668d31`

运行命令为各版本仓库下的 `python experiments/megamoe/scripts/batch_launch.py --ep-size 8 --bs-values 32,96,256,1024,4096 --wheel <对应wheel> --timeout 1800`；精度增加 `--mode accuracy`。
离线核对使用 `compare_e2e.py <旧版run目录> <新版run目录>`，三组均通过配置、框架/CANN、选样和rank聚合校验。

汇总与驱动脚本归档：`/home/zyx/main-e2e-results-20261010.tar.gz`；SHA256 `007bf9712e73d2a21e3543733295fae662d6175db3fce83ef4e6e0f8b7fcb69c`。
该小归档不包含大型timer/profiler原始文件，它们保留在上述run目录。测试结束时zyx安装的是本报告的新版wheel。

## 结论边界

这些计算优化的收益已在完整EP8通信路径兑现，五个BS没有性能回退。但本轮没有单独消融每项优化，不能把201 μs分摊给某一项。
之前约3924 μs是本地预置输入的纯计算诊断，输入seed和测量场景也不同，不能与4254 μs直接相减当作通信成本。
本轮未重测DeepSeek版本，不能据此宣称已追平DS。是否存在Combine尾部抵消，需要另按timer/硬件数据验证，不能仅由收益比例推断。
