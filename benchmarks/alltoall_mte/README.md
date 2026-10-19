# MTE All-to-All：EP16 / 四卡分组下行带宽

本目录是面向 **Ascend950DT、EP16（2×8 卡）、连续 4 个 rank 为一个 UBX 物理组** 的
All-to-All 微基准。它只搬运 token payload，不包含 Notify、路由、去重、专家计算或
Combine 归约，用来单独回答 MegaMoE dispatch/combine 带宽上不去时的一个问题：

> 在同样的 `token_bytes` 和 `tokens/rank-pair` 下，MTE 下行本身能达到多少
> 本卡内、组内、组间带宽？

实现复用主线 MTE 路径：`aclshmem_malloc` 对称堆 + `aclshmem_ptr` 对端映射 +
`DataCopyPad` 的 GM→UB→远端 GM 写入，与 MegaMoE token 通信使用同一类 MTE3
搬运；`aclshmemx_mte_quiet()` 在每个 kernel 结束时排空。没有 URMA WQE、没有
聚合协议、没有 flag 轮询，因此测得的是纯搬运上限，而不是某个算法实现的上限。

## 1. 拓扑与语义

```text
world = 16, group_size = 4
group(r) = r / 4          # 0..3
peer 视图（以 rank r 为例）:
  local  : {r}                        本卡内
  intra  : group(r) \ {r}             组内其余 3 卡（下行）
  inter  : world \ group(r)           其他 12 卡（组间下行）
  full   : world                      全部 16 卡（含本卡，交叉校验用）
```

每个 rank 的发送张量形状为 `[world, tokens, token_bytes]`：第 `d` 块只发给
rank `d`，块内是 `tokens` 个 token，每个 token `token_bytes` 字节。接收端对称堆
按 **源 rank** 分区：

```text
workspace + source_rank * tokens * token_bytes + token_id * token_bytes
```

因此目的地址偏移与目的 rank 无关，只有映射后的基地址随 peer 变化；这是
All-to-All 的直接映射，不需要额外的收件箱索引。

`token_bytes` 默认 **3584**，`tokens` 默认 **2048**，`--tokens` 支持逗号分隔的
多个值（例如 `1024,2048,4096`），每个值都会完整跑一遍 local/intra/inter/full。

## 2. 构建

本 benchmark 不新增 Python 包，只给 `deep_ep._C` 增加 `alltoall` /
`alltoall_drain` 两个原生入口，并把 `csrc/kernels/alltoall/alltoall_mte.cpp`
加入设备库。构建方式与仓库主线一致：

```bash
source /path/to/ascend-toolkit/set_env.sh
export SHMEM_ROOT=/path/to/shmem/install/shmem
source "$SHMEM_ROOT/set_env.sh"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"
export DEEPEP_NPU_ARCH=Ascend950

python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

只改 Python/文档时不需要重新编译；改 kernel 后重新执行上面的安装命令。增量开发
也可以直接使用仓库根目录的 CMake 构建，把 `_C.so` 与 `libdeepep_kernels.so`
装到 `deep_ep/` 下，然后设置 `PYTHONPATH=<repo>`。

验证扩展和 MTE 会话：

```bash
python -c "from deep_ep._native import require_native; m=require_native('alltoall'); print(m.alltoall_workspace_bytes(16,2048,3584))"
```

输出应为 `117440512`（16×2048×3584）。

## 3. 运行

### 3.1 单机 16 卡（目标机器，推荐）

目标机器是一台 16 卡的 Ascend 服务器（具体型号以现场为准）。先确认无人使用：

```bash
npu-smi info
```

`Process id` 区域应全部是 `No running processes found`。然后在仓库根目录执行：

```bash
cd /path/to/DeepEP
CANN_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh \
SHMEM_ROOT=/path/to/shmem/install/shmem \
PYTHON=/path/to/venv/bin/python \
bash benchmarks/alltoall_mte/run_16cards.sh --tokens 1024,2048,4096
```

脚本会启动 16 个进程（`RANK=0..15`，`--device 0..15`），rank 0 先建立 SHMEM
endpoint，其余 rank 重试连接；结束时自动汇总并打印表格。`MASTER` 默认取本机第一个
IPv4，多网卡时显式指定：

```bash
MASTER=<本机可达IP> ... bash benchmarks/alltoall_mte/run_16cards.sh --tokens 1024,2048,4096
```

`run_16cards.sh` 启动前会解析 `npu-smi info` 的 `Process id` 区域；如果发现已有
进程会拒绝启动（退出码 3）。确认这些进程是你自己的之后，可以用
`ALLOW_BUSY=1` 跳过该检查，但不要在多用户机器上这样做。

### 3.2 两节点 EP16（仅 177.2.21.1 / 177.2.22.1 验证用）

如果目标代码需要在两台 8 卡节点上验证，node1 必须能从 node0 免密 SSH。脚本会通过
SSH 启动 node1 的 rank 8..15，并在结束后把 `rank_*.json` 拷回 node0 汇总：

```bash
cd /path/to/DeepEP
MASTER=177.2.21.1 PEER=177.2.22.1 \
CANN_ENV=/home/pkg/b080/cann-9.2.0/set_env.sh \
SHMEM_ROOT=/path/to/shmem/install/shmem \
PYTHON=/path/to/venv/bin/python \
bash benchmarks/alltoall_mte/run_2nodes.sh --tokens 1024,2048,4096
```

两个节点必须各自有同一路径的仓库和同一份构建产物（`deep_ep/_C*.so` 与
`deep_ep/libdeepep_kernels.so`）。脚本会分别检查两个节点的 `npu-smi info`，
发现占用时同样拒绝启动。

### 3.3 参数

两个启动脚本共享以下参数：

| 变量 | 默认 | 含义 |
| --- | --- | --- |
| `TOKENS` / `--tokens` | `2048` | 每个 rank-pair 的 token 数，可逗号分隔 |
| `TOKEN_BYTES` / `--token-bytes` | `3584` | 单 token 字节数，必须 32B 对齐 |
| `GROUP_SIZE` / `--group-size` | `4` | 物理组大小，必须整除 world |
| `WARMUP` / `--warmup` | `10` | 计时前预热次数 |
| `ITERS` / `--iters` | `50` | 每类计时的采样次数 |
| `CORES` / `--cores` | `64` | 参与搬运的 AIV 核数；不同型号按实际 AIV 数调整 |
| `MASTER` / `--master` | 本机首个 IPv4 | SHMEM endpoint 所在地址 |
| `SHMEM_PORT` / `--port` | `19091` | SHMEM bootstrap 端口 |

所有 rank 必须使用同一个 `DEEPEP_SHMEM_ENDPOINT`，默认是
`tcp://$MASTER:19091`。目标机器只有一台时不需要 `PEER`。

### 3.4 手工启动（调试单 rank）

```bash
source "$CANN_ENV"
export SHMEM_ROOT LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=$PWD DEEPEP_SHMEM_ENDPOINT=tcp://$MASTER:19091
RANK=0 WORLD_SIZE=16 LOCAL_RANK=0 \
python -m benchmarks.alltoall_mte.bench_alltoall \
  --rank 0 --world-size 16 --local-rank 0 --device 0 \
  --token-bytes 3584 --tokens 1024,2048,4096 \
  --group-size 4 --cores 64 --warmup 10 --iters 50 \
  --output-dir benchmarks/alltoall_mte/results/manual
```

每个 rank 写一个 `rank_XX.json`；全部结束后在任意一台有全部文件的机器上：

```bash
python -m benchmarks.alltoall_mte.summarize \
  --input benchmarks/alltoall_mte/results/manual \
  --title "EP16 group4 3584B"
```

会生成 `summary.json`、`summary.csv`、`summary.md`，并把表格打印到 stdout。

## 4. 输出表格

主表是 **每 rank 出向聚合带宽**（`max-rank median`，十进制 GB/s），由
`summarize.py` 在运行结束后写入 `summary.md`：

| tokens/pair | 本卡内 (GB/s) | 组内下行 (GB/s) | 组间下行 (GB/s) | 全量 (GB/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1024 |  |  |  |  |
| 2048 |  |  |  |  |
| 4096 |  |  |  |  |

这里不预填数值：目标机器是单机 16 卡且型号与两节点验证环境不同，必须以目标机器上
`summary.md` 的实际结果为准。

`summary.md` 里还会给出每个目的地的平均份额（聚合带宽 ÷ peer 数），便于和
单链路/单端口带宽对比。

### 已验证范围

- 在 `177.2.21.1` 上完成编译：CANN 9.2.0（b080）、SHMEM
  `4dd80e2129da42bd773982c2bb3758af1ed0bba8`、gcc 12.3.1、CMake 3.31.10，
  `_C.so` 与 `libdeepep_kernels.so` 均成功链接。
- 在 `177.2.21.1` 单节点 8 卡上跑通 8 rank、`tokens=2`、`token_bytes=64` 的
  全流程：`local` / `intra` / `inter` / `full` 四种 scope 均执行，`alltoall_drain`
  回读后逐字节校验通过。
- 同机 8 rank、`tokens=1024,2048,4096`、`token_bytes=3584`、`group_size=4` 的
  sanity 结果（`warmup=2`、`iters=5`，check 全部通过）：

  | tokens/pair | 本卡内 (GB/s) | 组内下行 (GB/s) | 组间下行 (GB/s) | 全量 (GB/s) |
  | ---: | ---: | ---: | ---: | ---: |
  | 1024 | 85.44 | 93.55 | 117.40 | 249.18 |
  | 2048 | 192.42 | 101.09 | 133.73 | 260.31 |
  | 4096 | 295.64 | 103.01 | 136.88 | 271.32 |

  该结果来自单节点 8 卡（2 个 4 卡组）、SHMEM MTE 路径，**不是目标 16 卡单机
  的结论**；它只证明 `3584B` 下 `1024/2048/4096` 的真实 payload 路径、逐字节
  check、计时和汇总链路可用。该节点上组内/组间每个目的地的稳定份额约
  `31–34 GB/s`；`1024` 的 `本卡内` 受固定开销影响明显，不能按线性外推。
  目标机器请以第 3.1 节命令生成的 `summary.md` 为准。
- 未在 2×8 节点上跑 16 rank 的正式带宽：目标机器是单机 16 卡，跨节点数字不可
  迁移。

### 计时与口径

- 计时使用 `torch.npu.Event` 记录在 `Runtime.stream` 上，事件区间只覆盖
  `alltoall` kernel 的流执行，不包含 Python 参数检查、`barrier` 和结果落盘。
- 每次采样前所有 rank 调 `Runtime.barrier()`；采样后再次 barrier，避免上一轮
  流量叠到下一轮。
- 每个 case 取该 rank 的**中位数**；跨 rank 取**最大值**作为主表时间
  （最慢 rank 的带宽），同时保留 mean/min 和中位数原始样本到 `summary.json`。
- `本卡内` 是 GM→UB→GM 的本卡 copy，`组内/组间` 是 MTE3 远端写；所有口径都按
  **发送 payload 字节数**计，不把接收侧读取或协议头算进去。
- `full` 包含本卡，因此其带宽不等于 `local + intra + inter` 的简单相加；它用于
  交叉校验和给 MegaMoE 全量 dispatch 提供参照。

## 5. 正确性检查

默认在每个 `tokens` 值上跑一次 `full` All-to-All，然后 `alltoall_drain` 把
接收堆按 `[source, token, byte]` 拷回张量，与确定性的
`(source, destination, token, column)` 模式逐字节比较。任何一个 source 分区的
首字节、token 号或 payload 不一致都会让该 rank 失败并打印 mismatch 计数。

`--no-check` 可跳过检查，只测带宽；正式记录不应使用该选项。

## 6. 与 MegaMoE 的对应关系

| 本 benchmark | MegaMoE 通信 |
| --- | --- |
| `aclshmem_malloc` 对称堆 | `epHcclBuffer_` / peermem 窗口 |
| `aclshmem_ptr(base, peer)` | `winRankAddr[peer] + offset` |
| `DataCopyPad` GM→UB→远端 GM | token/scale 的 MTE 搬运 |
| `aclshmemx_mte_quiet()` | 通信阶段结束前的 MTE 排空 |
| 无 Notify / 无路由 | 真实 dispatch/combine 的元数据与去重 |

因此它回答的是“链路和 MTE 引擎上限”，不回答“某个 Wave/去重实现是否高效”。
如果这里的组间下行已经明显低于 `local` / `组内`，瓶颈在链路或映射；如果这里
很高而 MegaMoE 很低，瓶颈在路由、元数据、同步或 kernel 内的串行化。
