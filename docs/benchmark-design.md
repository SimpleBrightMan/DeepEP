# EP benchmark 详细设计（协议 v1）

## 目标与边界

本批实现可在无框架环境运行的 CLI/负载规划、通用测量引擎、严格统计和 JSON/CSV 输出，并固定后续原生后端的私有接入协议。公开 `deep_ep` API 不变，不引入通信实现、第三方运行依赖或动态插件加载。

原生构建/运行时和 V1/V2 实现属于后续批次。当前只能运行 dry-run 和协议单元测试；真实测量必须在适配器可用且能力匹配后执行。测试中的调用轨迹对象只验证时序，生产入口不可选择它，不用于生成性能基线。

## 分层和执行时序

`bench_ep` 解析配置 → `_config` 规范化并校验 → dry-run 规划或 `_backend` 能力检查 → 原生会话 → `_utils` 测量 → 会话成功退出 → rank 0 写入结果。

| 阶段 | 工作 | 是否计时 |
| --- | --- | --- |
| 配置 | JSON/CLI 合并、shape/容量校验、能力与 shape 查询 | 否；不能初始化设备 |
| 会话 | 校验启动信息、初始化受本次会话所有的资源、所有 rank 比较配置 SHA256 | 否 |
| 准备 | 生成输入、创建 buffer、准备 combine/cached handle | 否 |
| 前置检查 | 独立正确性验证、同步、全 rank 成功投票 | 否 |
| 预热 | barrier → 同步 → invoke → wait → 同步 → release | 否 |
| 每次采样 | barrier → 同步 → 起始时间 → invoke → wait → 同步 → 结束时间 → release | 仅两个时间点之间 |
| 后置检查 | 相同负载正确性验证、同步、全 rank 成功投票 | 否 |
| 统计 | 收集完整 rank-major 样本，逐轮 MAX 后计算统计 | 否 |
| 退出与发布 | 有界资源清理成功后，rank 0 创建新目录并写结果 | 否 |

所有 rank 必须执行相同 warmup/iterations；样本按 rank 和迭代索引对齐。预热不读取时钟、不进入原始样本。输入重复使用，不主动清 cache，这一策略记入结果。

## 配置与确定性输入

`Config` 的全部字段为：`api`、`operation`、`dtype`、`layout`、`dispatch_mode`、`world_size`、`tokens_per_rank`、`hidden`、`topk`、`experts`、`alignment`、`with_weights`、`route`、`seed`、`warmup`、`iterations`。最终有效配置完整写入文件，排序后的 JSON 计算 SHA256，所有 rank 必须先达成一致。

- API：`legacy` / `elastic`；操作：`dispatch` / `combine` / `roundtrip`。
- layout：`ordinary` / `expanded`；dispatch 模式：`fresh` / `cached`。
- dtype 表示 dispatch 的 BF16 或 FP8 E4M3FN；combine 输入及输出使用 BF16。combine-only 的 dtype 描述准备 handle 时的 dispatch 路径，不表示 FP8 combine。
- legacy 本版仅规划普通 BF16、alignment=1；低时延接口另行设计。
- FP8 使用每 128 列 scale，H 必须被 128 整除；elastic combine/roundtrip 按固定上游约束要求 H 被 256 整除。alignment 仅接受 1 或 128。
- E 必须能被 EP 整除，K≤E；每 rank 的 token 数显式提供，允许部分 rank 为 0、全局至少 1 个 token。
- JSON 不超过 64 KiB，拒绝重复/未知键、错误类型（包括把 bool 当整数）、非法枚举及范围。EP≤1024、H/E≤65536、K≤256、全局 token≤1048576、路由槽位≤8388608、预热/采样次数≤10000、保留 rank 样本≤1000000。这些是规划器容量限制，不代表硬件支持。

输入公式由 `_config.py` 提供，适配器必须复用相同语义。对源 rank `r`、token `t`、列 `c`、槽位 `s`：

```text
x(r,t,c) = ((seed + 13*r + 7*t + c) % 127 - 63) / 16
w(r,t,s) = ((seed + 19*r + 7*t + 3*s) % 127) / 128
e(r,t,s) = (seed + 17*r + 7*t + s) % E
```

hidden 值可被 BF16 精确表示。FP8 先生成 BF16，再按[既有参考](../tests/utils/precision.py)的上游 E4M3FN/每 128 列 scale 规则编码。`with_weights=false` 时省略独立 FP32 值通道；不改变 hidden 字节数。

`balanced` 使用上述 expert 公式（用于轮转路由，不保证每个配置严格均匀）；`skew` 再对 `E/EP` 取模，将全部有效路由压到 rank 0；`duplicate` 在 K>1 时令槽位 1 等于槽位 0；`masked` 在 `(seed+r+t+s)%5==0` 时令该槽位为 -1。行顺序、有效区域和 padding 由公开 API 布局契约决定，不能用内部 handle 表替代独立答案。

## 操作语义与正确性

| 操作 | `prepare()` | `invoke()` |
| --- | --- | --- |
| fresh dispatch | 输入与 buffer；V1 静态路由张量可准备 | 本轮所需布局/通知及 dispatch，输出保持存活至 release |
| cached dispatch | 首次 dispatch 生成可复用 handle | 复用同一合法 handle，重复 dispatch 相同路由的输入 |
| combine | dispatch 生成真实 handle 与接收数据；FP8 预先恢复 BF16 | 只执行该 handle 对应的 combine |
| roundtrip | 输入与 buffer；cached 模式另准备 handle | dispatch → identity 专家数据传递 → 使用本次返回的匹配 handle combine |

roundtrip 直接测量，不合成单项数据；identity 专家不执行 GEMM、不额外乘 gate。FP8 roundtrip 必须在本轮计时区间内恢复 BF16，再 combine。普通布局一个源 token 每个目标 rank 仅一行，展开布局每个有效槽位一行，因而 identity combine 也不是无条件恢复原 x。重复槽位、mask、padding 和独立权重输出必须按路由参考验证。

`verify()` 要运行相同配置/路径、独立从源输入与路由得到预期，校验计数/布局/值通道和精度；不能仅检查 API 返回成功。dispatch payload/scale 按非计算规则检查，combine 按 [L2 双标杆及小值域规则](precision.md)检查。前后任一次任何 rank 失败都阻止性能报告。完整设备资格仍由 PR2 功能矩阵负责，本 benchmark 的单个负载不能替代矩阵资格。

## 私有原生适配器契约

唯一入口为固定模块 `deep_ep._native`，不接受用户指定模块、脚本或动态插件。该模块导入及 `describe_benchmark(config_dict)` 必须保持 CPU-only、无设备初始化与通信资源分配。描述符格式：

```json
{
  "protocol_version": 1,
  "capabilities": ["elastic", "dispatch", "expanded", "bf16", "weights", "unequal_tokens", "route_balanced"],
  "shape_supported": true
}
```

能力为精确集合匹配，无 `all` 绕过：基础 API/layout/dtype/dispatch，另按配置要求 combine、roundtrip、cached、alignment128、unequal_tokens、zero_tokens、single_rank、weights 和 `route_*`。具备能力仍需逐 shape 检查；任何缺失或未知协议均先拒绝，不能通过失败后回退的方式冒充支持。

`create_benchmark(config_dict)` 在门禁之后调用，返回上下文管理器，进入后提供 `_backend.BenchmarkRuntime`：

| 成员 | 契约 |
| --- | --- |
| `rank`, `world_size` | 整数，与启动环境和配置一致 |
| `metadata` | 恰好 backend/device/software/topology/variant 五个公开字符串；每项 1–256 字符、无控制字符或表格公式前缀 |
| `agree_config(fingerprint)` | 全 rank 配置一致后返回，否则所有 rank 失败；先于输入与 buffer 准备 |
| `prepare()` | 操作表规定的准备，失败时可安全退出 |
| `verify()` | 正确即返回，错误抛异常；结果全部就绪后完成 |
| `all_ranks_ok(local_ok)` | 集体投票，只有所有 rank 成功时返回 Python `True` |
| `barrier()` | 同一轮开始前的全 rank 栅栏 |
| `synchronize()` | 同步本轮观察流，并覆盖所有相关设备操作 |
| `invoke()` | 执行所选完整操作并返回持有所有输出及必要 handle 的 lease |
| `wait(lease)` | 将通信/拷贝/归并的全部完成依赖连接到观察流，不能只等待 Host 发起完成 |
| `release(lease)` | 在计时结束后释放输出；不能提前破坏缓存 handle |
| `gather_samples(local_us)` | 每个 rank 都得到完整 `[rank][iteration]` 数组，不能只在 rank 0 返回或丢弃失败 rank |

适配器负责启动校验、工作流自身精度检查、超时和资源清理。必须拒绝接管已有调用方进程组，为初始化/通信/集体操作/退出设置有限超时；异常时协调其他 rank 退出。会话只销毁本次创建的资源，构造失败也需清理部分资源；禁止在 `__exit__` 吞掉异常。成功退出须确认所有 rank 测量及退出状态一致，失败不能让 rank 0 提前发布成功结果。这些是后续适配器的资格条件，本批无设备实现，不能据协议测试宣称已经完成验证。

## 统计、结果与兼容

完整定义见[性能口径](performance.md)与[字段契约](../benchmarks/results/README.md)。所有样本必须为正有限数且维度精确匹配。每轮取 MAX 后使用 nearest-rank 百分位（排序数组下标 `ceil(p*N)-1`），mean 用 `statistics.fmean`；带宽基于平均 MAX 延迟与实际路由有效字节数。

`schema_version=1`；planned 不允许性能字段，measured 才有全 rank 原始样本与统计。报告生成时重新从原始样本计算汇总，不信任外部摘要；JSON 禁止 NaN/Infinity。CSV 保留原模板字段顺序并追加协议字段，无匹配基线时留空，`reproduce_command` 保留待填写的公开 launcher 参数。`record_id` 是配置指纹前 16 位（同负载多次运行相同），完整实验身份还需 revision、measured_at 与 metadata。

只有 rank 0 在成功退出会话后写新目录；已有路径拒绝覆盖。报告及 CSV 序列化先于建目录，文件系统写入失败返回非零，残留目录不视为完整实验，重跑必须用新目录。入口不自动采集主机名、地址、路径、环境变量或凭据；导出前需人工核对适配器公开元数据。

## 验证策略与后续接入

- 无框架测试：CLI help/list/dry-run 与未实现路径，拦截 Torch/torch_npu/NumPy/MPI 导入，证明能力失败不触碰设备。
- 独立语义检查：普通/展开、四类路由、三种操作、BF16/FP8 的逻辑字节数对照 PR2 独立路由参考。
- 时序轨迹检查：预热不计时、wait/同步在结束时间前、release/collective 在计时外，前后正确性投票与失败退出。
- 统计和结果检查：交叉 rank 样本验证先 MAX 后 percentile；缺失/零/负/NaN/Inf 拒绝；配置回读、CSV 字段锁定、dry-run 无性能、输出不覆盖。
- 原生适配器接入时追加真实多 rank 生命周期、故障超时、同 handle 往返、异步依赖、FP8 恢复、设备精度及原始样本资格；未通过不能宣称可测量。

运行 `python -m pytest tests/test_benchmark.py -q` 或完整 CPU 测试。持续集成同时执行完整 CPU 契约/参考/benchmark 协议测试及全部 preset dry-run。
