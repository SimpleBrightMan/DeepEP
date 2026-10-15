# Concurrent Dispatch bandwidth benchmark

This standalone Ascend950 benchmark runs LOCAL, GROUP and CROSS traffic concurrently
in **one mixed kernel on every rank**. It isolates the packed-token copy stage of
MegaMoE Dispatch, without GMM, SwiGLU, quantization or device-side route generation.
It does not change the installed DeepEP extension.

The defaults model one 16-card node with 28 AIC launch groups per card, four
consecutive ranks per direct-link group, H=3584, I=1280, E=384 and TopK=8.
Only AIV1 in each launch group copies data; AIC and AIV0 do no work. The 28
communication workers have fixed roles:

| Traffic | Source ranks relative to receiver | Dedicated workers |
| --- | --- | --- |
| LOCAL | Receiver itself | 2 |
| GROUP | Other three ranks in the same four-rank group | 5 |
| CROSS | Twelve ranks in other groups on this node | 21 |

Workers never switch classes. The three classes start in the same kernel, but
may finish at different times. This measures concurrency under a fixed Dispatch
load; it does not keep all three classes saturated after the smaller ones finish.
Rank order follows `--devices`: keep its order consistent with the physical groups.
This benchmark uses MTE peer access within one node, not RoCE or multi-node SHMEM.

## Build

Use the repository's pinned CANN/SHMEM toolchain (see
[development instructions](../../docs/development.md)). Prepare the SHMEM SDK as
for the normal DeepEP build, or point `SHMEM_ROOT` at an existing compatible SDK.
The build script checks SDK compatibility and does not install packages or build
the SDK. A Python environment with pybind11 and the normal host build tools is
required; running the benchmark does not require PyTorch.

```bash
source /path/to/matching/ascend-toolkit/set_env.sh
export SHMEM_ROOT=/path/to/shmem/install/shmem
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:$ASCEND_HOME_PATH/lib64:${LD_LIBRARY_PATH:-}"
bash experiments/dispatch_bandwidth/build.sh
```

The executable and its device library are placed together under
`build-dispatch-bandwidth/experiments/dispatch_bandwidth/`. Override `BUILD_DIR`
when building and pass `--binary /absolute/path/to/dispatch_bandwidth_bench` when
running if using a different directory.

## Run

Run on the allocated, idle 16-card target node:

```bash
npu-smi info
python experiments/dispatch_bandwidth/run.py \
    --bs-values 2048,4096,8192 --timer
```

For only BS4096:

```bash
python experiments/dispatch_bandwidth/run.py --bs-values 4096 --timer
```

The launcher starts one native process per card, uses a fresh local SHMEM
endpoint for each batch size, and prints three lines: `LOCAL`, `GROUP`, `CROSS`.
It saves raw rank reports, logs, a summary JSON, and, with `--timer`, per-worker
CSV and Chrome trace JSON. Results go to a unique directory under
`benchmark-runs/dispatch_bandwidth/`; timer export is optional, timing itself is
always enabled. Trace timestamps are relative to each rank's own origin, not
synchronized timestamps across different cards.

Useful options:

```bash
# Inspect commands without accessing NPUs, allocating ports or creating files.
python experiments/dispatch_bandwidth/run.py --bs-values 4096 --plan

# Change worker allocation while keeping all three classes concurrent.
python experiments/dispatch_bandwidth/run.py --workers 2,6,20 --timer

# Select only the first local expert, rather than the default complete Dispatch.
python experiments/dispatch_bandwidth/run.py --scope first-expert --timer
```

`--cores` and the sum of `--workers` must agree; each class needs at least one
worker. The host rejects a launch exceeding the device's AIC/AIV resources.
`--buffer-count` selects 2..6 pipeline slots (default 6). `--warmup`, `--iters`
and `--repeat` default to 10, 20 and 1. Repeats replay the same input and overwrite
the same output, increasing cache reuse; interpret them separately from a larger
batch. The intermediate dimension is checked and recorded for model identity,
but does not affect the copy stage.

## Workload and byte counts

Routes reproduce the repository's balanced stream: for source rank `s`, token
`t` and TopK slot `k`, global expert is
`(s * BS * TopK + t * TopK + k) % E`; destination rank is `expert % EP`.
The default includes **all local experts**. Each destination receives all its
matching routes from all source ranks. A source token can be read more than once
when multiple selected experts are on that destination; these are distinct
Dispatch rows, not deduplicated tokens. Task order is local expert, cyclic source
rank, then source token/TopK slot. Dedicated class queues replace MegaMoE's wave
scheduling, while preserving the source indices, per-row copy size and counts.

For H=3584, a read is 3584 FP8 bytes plus 128 bytes of padded scale storage:
**3712 bytes per row**. Of the scale bytes, 112 are valid. The kernel uses the
Dispatch-style GM-to-UB read and pipelined UB-to-GM writes, with separate local
writes for 3584 token bytes, 112 scale bytes and 32 metadata bytes. It reads a
16-byte precomputed task record per row as additional local bookkeeping.
Source contents are deterministic bit patterns for exact validation, not
numerical FP8 values; all output rows are verified outside the timed region.

For BS4096, EP16, E384 and TopK8:

| Traffic | Rows per receiving card | Read bytes per card |
| --- | ---: | ---: |
| LOCAL | 2048 | 7,602,176 |
| GROUP | 6144 | 22,806,528 |
| CROSS | 24576 | 91,226,112 |
| Total | 32768 | 121,634,816 |

BS2048 halves these counts and BS8192 doubles them. Other shapes may have small
integer differences across ranks; the report uses the actual planned counts.

## Bandwidth definition

For each class, rank and iteration, the device timer records each worker's start
and completion after its outstanding output writes drain. Class elapsed time is
`max(end_cycle) - min(start_cycle)` over workers with nonzero rows. The bandwidth
is `class_rows * repeats * row_bytes / elapsed_us / 1000`, in decimal GB/s.
Cycles use the same Ascend950 system-counter conversion as MegaMoE's timer
(1000 ticks per microsecond). ACL events separately record the whole kernel time.

Each printed class bandwidth is the mean over iterations and then over cards;
rank min/max and iteration percentiles accompany it. These are **per-card useful
read-payload rates**, not the sum of individual core rates, physical-link rates,
or read-plus-write HBM rates. LOCAL measures the same packed read plus output
writes entirely on one card. GROUP/CROSS include local output writes in their
timed pipelines. Forwarding-chip traffic, protocol overhead and task/metadata
bytes are not included in the communication numerator.

The summary also records each class's contribution over the common whole-kernel
ACL event window. This is useful when a shorter class finishes early. No report
subtracts raw timestamps from different cards: their clocks are independent.
Host SHMEM barriers separate iterations; they do not guarantee identical device
start times on all cards. This benchmark is intended to compare the three
traffic classes under concurrent all-to-all load, not to measure end-to-end MoE
latency or an isolated link's peak bandwidth.

## CPU checks

```bash
python -m pytest -q experiments/dispatch_bandwidth/tests
```

The planner test needs a C++17 host compiler. These checks validate routes,
per-class counts, worker partitioning, report completeness and timing arithmetic;
they do not require NPUs.
