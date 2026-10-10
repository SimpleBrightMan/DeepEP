# MegaMoE 优化与验收 TODO

更新：2026-10-10。所有性能数字注明输入、版本、timer状态、计时与rank聚合口径。

## 当前已实现的计算优化

- [x] GMM2：删除 TokenCombine 路径逐tile重复 FIX_S，保留消费者所需的完成边界。
- [x] GMM2：同wave复用AIC/AIV0流水和credit，wave末排空后发布slice完成。
- [x] GMM1：FIX异步full通知、AIV1 MTE2等待、消费后empty归还；同wave保持流水状态。
- [x] 配套低开销生命周期timer，GMM1/GMM2异步区间明确标为ISSUE。
- [x] 合入main工作树并恢复完整通信benchmark默认行为，保留旧main对照分支。
- [ ] 完成目标Linux全套CPU回归并推送main与对照分支。

计算回归证据见 [LOCAL_COMPUTE.md](LOCAL_COMPUTE.md)。不包含扩大A窗口/四槽B两个负收益试验。
本地输入诊断入口保持独立，不把“跳过通信”作为main的默认性能路径。

## 第一优先级：旧main与新main完整EP8比较

- [ ] 独占8张NPU后，先检查完整通信正确性，再比较性能。
- [ ] BS依次为 **32、96、256、1024、4096**；DeepSeek V4 Pro、FP8/FP4、balanced、无共享专家。
- [ ] 旧main与新main使用相同CANN、框架、输入seed、路由、warmup、iters与profiler配置。
- [ ] 两边均保存timer；保持现有“性能采样timer-off、另外采一批timer”的同口径默认行为。
- [ ] 记录每rank结果和rank-max-mean，同时保留现有rank-min-mean；不可交叉比较二者。
- [ ] 每版至少3组，建议A/B/A；每次切分支重建/安装对应wheel，禁止误用另一版本的`--skip-build`。
- [ ] 核对Dispatch→GMM1→SwiGLU→GMM2→Combine→Unpermute全部执行，日志不得出现COMPUTE-REPLAY。
- [ ] 尤其检查wave尾统一发布是否推迟Combine，避免计算提速被通信尾部抵消。

| BS | 旧main μs | 新main μs | 减少μs / % | 精度 | timer目录 |
| --- | --- | --- | --- | --- | --- |
| 32 | 待测 | 待测 | 待测 | 待验收 | 待记录 |
| 96 | 待测 | 待测 | 待测 | 待验收 | 待记录 |
| 256 | 待测 | 待测 | 待测 | 待验收 | 待记录 |
| 1024 | 待测 | 待测 | 待测 | 待验收 | 待记录 |
| 4096 | 待测 | 待测 | 待测 | 待验收 | 待记录 |

用户进行这轮设备测试；本次不为补历史profiler指标重跑程序。

## 第二优先级：解释已有纯计算profiler

- [x] 从原始4196μs版、GMM2 wave约4027μs版、GMM1异步约3924μs版提取历史CSV和配置。
- [x] 只比较相同device/BS/seed和选样，核对设备库SHA及完整输出SHA。
- [x] 列出Duration、Wait Time、Block Num实际变化，以及未采集的指标。
- [x] 明确task级等待不等于GMM内部flag等待，Block Num不等于忙碌率。
- [x] 三版均为Level0/ACL_AICORE_NONE，没有Cube/MTE/带宽/cache指标，不能恢复未采集的数据。

结果及字段解释见 [PROFILER_COMPARISON.md](PROFILER_COMPARISON.md)。未重新运行程序。

## 完整EP8复现步骤

旧main原样保存在 `codex/megamoe-main-before-compute-optimizations`（`8ffafeb`）。
实际运行对照使用 `codex/megamoe-main-baseline-ep8`：计算源码与旧main完全相同，只补本机EP8启动/构建支持和说明。
新main保留相同完整通信入口，另附显式调用才启用的本地计算诊断，不自动进入replay。

加载同一zyx/CANN环境后，仓库根目录执行：

```bash
git fetch origin
git switch codex/megamoe-main-baseline-ep8
python experiments/megamoe/scripts/batch_launch.py --ep-size 8 --bs-values 32,96,256,1024,4096
# 记录上面 [OUTPUT] 的完整目录，首次不加 --skip-build。

git switch main
git pull --ff-only origin main
python experiments/megamoe/scripts/batch_launch.py --ep-size 8 --bs-values 32,96,256,1024,4096
# 记录新目录；切版本必须重建，不能复用旧wheel。

python experiments/megamoe/scripts/compare_e2e.py /实际旧版输出目录 /实际新版输出目录
```

先做精度回归可使用相同五BS命令加 `--mode accuracy`。该命令生成/校验CPU Golden，可能耗时较长；
精度之后再次perf，确认相同源码且已安装的wheel匹配时才可以使用 `--skip-build`。
比较脚本只读两组已完成报告，输出每BS的rank-max-mean、减少μs/%及rank-min-mean，
并拒绝replay、缺失rank、不同seed/config/CANN/框架或不一致选样；不启动NPU任务。
它不能证明设备独占，不能用一组输出冒充多组重复测试。

timer目录为 `<输出目录>/bs_<B>/bs_<B>/timer/`，各rank JSON和CSV保留原时间轴。
本次没有替用户重跑这五个BS；上表保持待测，不填入纯计算数据。

## 后续候选：先证据、后单变量实验

1. **GMM2异步完成与Combine提前消费**：若完整E2E尾部Combine明显，保留跨生产核的slice完成聚合，
   在FIX实际完成后让消费者提前推进；保持tile/缓冲不变。验收完整通信精度、死锁及E2E，而非只看纯计算。
2. **GMM1 L0C→UB直供SwiGLU**：核算gate/up输出容量、AIV1与Dispatch/Combine的UB占用，
   成对设计full/empty；避免覆盖未消费输出。单独验证GM往返减少是否抵消拆块/握手成本。
3. **连续tile供数衔接**：检查下一tile的FP4转换、L1→L0搬运和MMAD推进；尤其GMM2短K。
   先采有效的硬件指标，再选预取时机/缓冲改动；旧生命周期下扩大缓冲失败不代表新布局永远无效。
4. **GMM1/GMM2切换成本**：定位资源排空、布局恢复及事件重启；只复用可证明安全的状态，不直接删除End。
5. **按形状优化tile/wave**：待数据流稳定后，小范围搜索并覆盖五个BS，不同时改变多个变量。

每项实验单独记录：父提交、唯一改动、正确性、timer-off A/B/A、原始结果路径、收益/回退决定。
后续硬件计数器采样必须单列配置与开销；高Cube时间占比不等于高有效算力，带宽也必须与对应层级上限比较。
