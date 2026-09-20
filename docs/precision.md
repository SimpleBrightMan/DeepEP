# DeepEP 精度标准、Golden 与比较方法

本文固定 reference tests 的精度口径。它区分通信语义正确性、浮点计算精度和设备资格，避免用小规模精确样本替代真实数值验收。

## 1. 依据与适用范围

- 精度标准：工作区《昇腾算子精度标准2.1》，按浮点计算 L2 标准执行。
- 用例规则：工作区《算子精度测试工程指导》，覆盖 dtype、shape/边界、参数正交、均匀/正态分布及已知易错场景。
- 上游语义：DeepSeek DeepEP 提交 `a56d6156febcd9976e55adc85b5155bfac9f28f8`。该提交的 V2 测试使用 BF16 激活，也覆盖按 token、每 128 列缩放的 `float8_e4m3fn` dispatch，随后恢复为 BF16；top-k 索引使用原生构建选择的整数宽度，top-k weights 为 FP32。

当前 Ascend 原生后端仍未实现，因此 CPU 测试只证明参考、生成器与判定器自洽。只有在目标设备上运行并保存全 rank 结果后，才能声明设备通过。

## 2. 三类输出的判定

| 输出 | 分类 | 判定 |
| --- | --- | --- |
| route/index、每 expert 计数、handle 布局 | 非计算/整数 | 有效区域逐项或二进制一致；保留重复槽位重数 |
| dispatch 的 BF16 或 FP8 payload/scale | 非计算 | 与同 dtype 标杆逐位一致；padding 与未定义槽位不进入比较 |
| combine 浮点输出 | 浮点计算 | CPU 高精度 Golden + 上游同 dtype 标杆双标杆比较 |
| top-k weights/value channel | 搬移或独立归并 | 搬移路径逐位一致；发生浮点归并时使用双标杆 |

禁止把 `combine(dispatch(x)) == x` 当作通用正确性条件。普通布局按目标 rank 去重，展开布局按有效 slot 展开；重复 expert slot 会改变归并重数。

## 3. 双标杆指标

CPU FP64 数学参考为 Golden，GPU 上游结果（未接设备时为固定上游 dtype 路径的明确标注模拟值）为第二标杆。对 NPU 和第二标杆分别计算：

- `MARE = max(abs(actual - golden) / (abs(golden) + 1e-7))`
- `MERE = mean(abs(actual - golden) / (abs(golden) + 1e-7))`
- `RMSE = sqrt(mean((actual - golden)^2))`

误差比为 `NPU 指标 / max(第二标杆指标, err)`。L2 通过线为 MARE `<= 2`、MERE `<= 1.2`、RMSE `<= 1.2`。

小值域单独统计，不允许被整体平均掩盖：

| dtype | `abs(golden)` 小值阈值 | 绝对误差阈值 `err` |
| --- | ---: | ---: |
| BF16 | `2^-8` | `2^-16` |
| FP8 E4M3FN | `2^-4` | `2^-6` |

小值域错误数比 `ErrorCount_npu / max(ErrorCount_benchmark, 1)` 必须 `<= 2`。NaN/Inf 不进入上述除法；其类别和符号按标准单独匹配。本仓当前生成器只生成有限值，比较器遇到非有限值会拒绝，防止静默通过。

单次用例不通过后才进入复检：更换随机种子至少 200 次（推荐 1000 次），Bootstrap 2000 次，取误差比中位数的 95% 置信区间；`CI_lower > 1.0` 或样本数少于 200 时失败。`bootstrap_median_recheck` 提供固定 seed 的可复现实现。复检不能覆盖路由、shape、计数、数据破坏或非有限值错误。

## 4. 资格矩阵

`tests/utils/matrix.py` 继承“框架合入”的分层规划，维护两个互不混淆的矩阵。

CPU 参考矩阵使用 pairwise 场景控制运行量，并覆盖：

- EP：1、2、4、8；experts：8、64、256。
- hidden：16、127、128、256、7168，包含对齐和非对齐边界。
- 每 rank token：1、15、16、17、33、不等长 rank，以及上游默认最大值 4096。
- top-k：1、2、4、6、8。
- dtype：BF16 和 FP8 E4M3FN；FP8 遵循上游每 128 列缩放与 BF16 恢复语义。
- 数值分布：正确性均匀分布 `[-5, 5]`、稳定性近似正态分布（均值 `[-100, 100]`、标准差 `[1, 25]`）、dtype 小值域，以及每 1000 个元素至少一个的乘性离群点；生成公式无全局随机状态，可由 case 配置完全重建。
- 路由：本地/跨 rank、mask、重复目标、倾斜、空 expert、零 token rank等预设场景继续由 `ROUTES` 套件覆盖。

NPU 资格矩阵只使用固定上游 V2 combine 可接受的 `H % 256 == 0` shape，覆盖 EP 1/2/4/8、alignment 1/128、BF16/FP8、T=4096 边界和上游默认 H=7168/K=6/E=256。不同 EP 必须通过独立的 `torchrun --nproc-per-node` 执行；两 rank 小算例不能代替 EP4/EP8 资格。

这不是所有维度的笛卡尔积。设备发布资格还应根据目标模型补充真实 shape，并记录平台、CANN/驱动、源码提交、命令和所有 rank 结果。

## 5. Golden 生成和防篡改

运行：

```bash
python -m tests.golden.generate
python -m pytest tests/test_precision.py -q
```

生成文件 `tests/golden/deepep_reference_v2.json` 包含 schema、上游提交、完整 CPU reference case 配置、设备资格 case 配置、有效 route slot 数、普通/展开布局行数、expert count 摘要、CPU FP64 combine 摘要、上游 dtype 标杆摘要及标杆相对 Golden 的误差指标。

MERE/RMSE 的浮点归约显式使用 `math.fsum`，避免 Python 3.11 与 3.12 的内置 `sum` 算法差异导致 Golden 指标末位不一致（见 [Python 版本说明](https://docs.python.org/3.12/library/functions.html#sum)）。这不放宽精度阈值或 Golden 一致性检查；测试另用精确有理数累加核对小误差项。

Golden 不保存海量张量文本；输入由公开、无随机状态的整数公式完全重建，结果用规范 JSON 的 SHA-256 锁定。修改公式、路由语义、dtype 舍入或归并答案都会使回归失败。更新 Golden 必须同时审核生成器变更，禁止只替换摘要来消除失败。

设备结果不得写回覆盖 CPU Golden。应作为独立 capture 文件输入同一比较器，并保留 rank、case、dtype、shape 和来源标识。真实 GPU 标杆可替换 manifest 中的 dtype 模拟值，但必须使用同一输入和固定上游提交。

## 6. 当前边界

- 已完成：纯 Python 高精度参考、BF16/E4M3FN dtype 模拟、双标杆 L2 比较器、小值域规则、Bootstrap 复检、版本化 Golden、参数矩阵、故障注入和可收集的 V1/V2 设备用例。
- 待真实后端完成后执行：全 rank NPU capture、真实 GPU 第二标杆、NaN/Inf 设备行为、alignment 128/padding、跨节点拓扑、至少 200 次失败复检与压力运行。
- BF16/FP8 是固定上游 V2 EP 测试明确覆盖的 activation 路径；不得从 Python 签名存在推导 FP16/FP32 activation 已获支持。索引和权重 dtype 仍分别按原生 `topk_idx_t` 与 FP32 验证。
