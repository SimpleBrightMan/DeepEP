# Concurrent Dispatch bandwidth benchmark

This standalone Ascend950 benchmark defaults to `--schedule concurrent`, running
LOCAL, GROUP and CROSS traffic concurrently in **one mixed kernel on every rank**.
It isolates the packed-token copy stage of
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

In this default schedule, workers never switch classes. The three classes start
in the same kernel, but may finish at different times. This measures concurrency under a fixed Dispatch
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

## Group round-robin schedule

On branch `codex/dispatch-group-round-robin`, select
`--schedule group-round-robin` explicitly to compare a fixed **12 INTRA / 16
CROSS worker split** with the original concurrent schedule. INTRA includes reads
from the receiver itself and the other three ranks in its physical group. The
host prepares separate task tables for the two classes; the kernel keeps the
original continuous six-slot `Run()` loop. This mode keeps **one kernel launch
per rank and iteration and all 28 AIV1 workers**, with both classes concurrent.
AIC and AIV0 still do no work. It requires EP16, four consecutive ranks per group,
`--cores 28` and `--buffer-count 6`; it does not select a smaller launch or scan
different worker/slot counts. Explicit `--workers` is rejected in this mode;
the 12/16 allocation is fixed.

```bash
python experiments/dispatch_bandwidth/run.py \
    --schedule group-round-robin \
    --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    --group-size 4 --cores 28 --buffer-count 6 \
    --bs-values 2048,4096,8192 \
    --hidden 3584 --intermediate 1280 --experts 384 --topk 8 \
    --metadata-mode dma --timer

# Run the original schedule with the same model, batches and metadata mode.
python experiments/dispatch_bandwidth/run.py \
    --schedule concurrent --workers 2,5,21 \
    --bs-values 2048,4096,8192 \
    --hidden 3584 --intermediate 1280 --experts 384 --topk 8 \
    --metadata-mode dma --buffer-count 6 --timer

# Inspect the group schedule without launching a kernel or allocating ports.
python experiments/dispatch_bandwidth/run.py \
    --schedule group-round-robin --bs-values 4096 --plan
```

Physical AIV1 workers 0..11 read only from the receiver's own group. Workers
12..27 read only from the other three groups, in the order below. For a receiving
rank in physical group `G`, CROSS segment `p` reads from source group
`(G - p + 4) % 4`, for `p=1,2,3`. Since these are GETs, this corresponds to a
sending group `Gi` supplying receiving group `(Gi + p) % 4`:

| Queue / segment | Source group relative to receiver | Traffic | Workers sharing rows |
| --- | --- | --- | ---: |
| INTRA queue | Same group | LOCAL and GROUP | 12 |
| CROSS segment 1 | Previous group, modulo four | CROSS | 16 |
| CROSS segment 2 | Two groups earlier, modulo four | CROSS | 16 |
| CROSS segment 3 | Three groups earlier, modulo four | CROSS | 16 |

The host divides all INTRA rows across 12 workers independently, with row counts
differing by at most one. Each remote source group's full row list is divided
across the 16 CROSS workers: `q, r = divmod(group_rows, 16)`. Each CROSS worker
gets `q` or `q+1` rows. An extra-owner cursor starts at zero and advances by `r`
after each remote group; the next group's complete chunks are assigned with
that rotated ownership. Thus CROSS row counts differ by at most one both within
each group and over the complete three-group task list. The two classes have
separate queues and never exchange worker roles.

Each CROSS worker's task table concatenates its three remote-group chunks in
the order above; INTRA workers independently process their own-group tasks.
The physical-to-logical worker mapping shifts once per kernel invocation within
each class, including warmups, spreading task ownership and extra rows across
physical workers without changing the 12/16 split. Reports record
`worker_rotation = launch_index % 48`; INTRA uses this value modulo 12 and CROSS
uses it modulo 16, giving a joint period of 48 kernel launches.
`--repeat` replays the complete task table with the same worker mapping; it does
not advance this rotation. Source indices, packed read size and total row counts
are unchanged. For the BS4096 default, the INTRA queue contains 2048 LOCAL and
6144 GROUP rows per receiving card, giving each INTRA worker 682 or 683 rows.
Each remote-group segment contains 8192 CROSS rows, giving each CROSS worker
512 rows per group and 1536 rows in total, for the original 24576 CROSS rows
per card.

The remote-group segments exist only in the host-prepared task order. **There is no phase
fence, phase drain, phase timer, or host/cross-rank/cross-core phase barrier.**
Each CROSS worker's six-slot ring carries on across remote-group boundaries,
allowing reads and writes from adjacent groups to remain in flight together.
INTRA and CROSS workers do not wait for one another. Initialization and iteration
boundaries retain ordinary SHMEM synchronization. Workers and cards may reach
the next group at different times. The ordering does not guarantee removal of
physical-link hotspots or identical per-worker durations.

Timer output remains **28 whole-worker intervals**, tagged `INTRA` for physical
workers 0..11 and `CROSS` for workers 12..27, using the original start and
completion timestamps. Each class has its own envelope
`max(class_end) - min(class_start)` and prints its useful read-payload bandwidth
over that envelope. INTRA combines LOCAL and GROUP bytes; the report does not
claim separate LOCAL/GROUP elapsed times. `TOTAL` bandwidth uses all payload
bytes divided by the same-card span across both classes; the report also gives
all payload bytes over the single whole-kernel ACL window. Source-group
diagnostics retain byte counts for the own group and three remote groups, with
no per-group timers. Use the **whole-kernel ACL event time as the main comparison** with
`--schedule concurrent`, and class envelopes and whole-worker timers to inspect
overlap and tails. No timestamps are subtracted across different cards. This
mode adds no plotting or sweep step.

## Workload and byte counts

Routes reproduce the repository's balanced stream: for source rank `s`, token
`t` and TopK slot `k`, global expert is
`(s * BS * TopK + t * TopK + k) % E`; destination rank is `expert % EP`.
The default includes **all local experts**. Each destination receives all its
matching routes from all source ranks. A source token can be read more than once
when multiple selected experts are on that destination; these are distinct
Dispatch rows, not deduplicated tokens. In the concurrent schedule, task order
is local expert, cyclic source rank, then source token/TopK slot. Dedicated class queues replace MegaMoE's wave
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

For the concurrent schedule, the device timer records each worker's start and
completion on every rank and iteration after outstanding output writes drain. Class elapsed time is
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

## Metadata pipeline comparison

`--metadata-mode scalar|dma|preload` selects three implementations of the same
Dispatch workload. The default is `dma`, removing per-row Scalar synchronization;
select `scalar` explicitly to reproduce the original benchmark.
Native reports include `metadata_mode`. The original concurrent format remains
schema version 2; group ordering uses schema version 5 with `mode` set to
`group_round_robin`, `worker_counts` of `intra: 12` and `cross: 16`, and
class-tagged whole-worker timers. Its `traffic_rows` retains LOCAL/GROUP/CROSS
counts for byte accounting, while printed class bandwidth uses INTRA/CROSS.
Aggregation rejects mixed metadata modes or report formats. Timer CSV and trace
files also identify the mode.

```bash
for mode in scalar dma preload; do
    python experiments/dispatch_bandwidth/run.py \
        --metadata-mode "$mode" --bs-values 2048,4096,8192 --timer
done
```

These runs keep rank selection, task order, per-row remote reads, three output
arrays, worker allocation and the communication bandwidth formula identical.
Both new modes still write all 32 metadata bytes per output row and run the same
full final-output validation. Neither prepopulates the output metadata.

| Mode | Metadata preparation inside the kernel | Per-row Scalar synchronization |
| --- | --- | --- |
| `scalar` | Three Scalar `SetValue` stores into the reused UB slot | `S_MTE3` and, when reusing a slot, `MTE3_S` |
| `dma` | One local 32B GM-to-UB DMA per row, alongside the unchanged remote GET | None |
| `preload` | One contiguous local GM-to-UB DMA for the worker's whole metadata table | None |

The host creates the read-only input metadata table in task order:
`[source_rank, source_token, topk_slot, 0, 0, 0, 0, 0]`. Like the existing task
plan, host table construction and H2D initialization are outside kernel timing.
The DMA that stages this table into UB is **inside both the device timer and ACL
event timing**. It is local bookkeeping traffic and does not enter the remote
read-payload bandwidth numerator. `dma` adds 32 local input bytes per row, about
0.862% of the 3712-byte remote read, and one MTE2 descriptor per row. `preload`
reads those bytes once per kernel; `--repeat` reuses the same read-only UB table.
The three output writes still occur on every repeat in every mode.

The `dma` slot protocol is:

```text
MTE2: wait MTE3_MTE2(slot) when reusing the slot
      read remote token+scale into rowUB[slot]
      read local metadata into metaUB[slot]
      set MTE2_MTE3(slot)
MTE3: wait MTE2_MTE3(slot)
      write token, scale, metadata to the three output arrays
      set MTE3_MTE2(slot)
```

The MTE3-to-MTE2 event protects both UB regions from being overwritten before
their previous writes have finished. No Scalar instruction writes those regions
in the loop, so there is no corresponding per-row Scalar reuse dependency.
`preload` uses the same data-slot protocol, but its metadata UB table is immutable:
the initial preload precedes the first remote read on MTE2, and the first
MTE2-to-MTE3 event makes both available to MTE3. Each pending output keeps its own
metadata row index, including across repeat boundaries.

The final MTE3-to-Scalar completion wait remains in both DMA modes, after all
pending output writes have been submitted. It is needed to read the end timestamp
and safely reuse UB for the timer record. Removing Scalar synchronization from
the row loop does not remove the requirement to finish all output before ending
the measurement. Data-slot MTE3-to-MTE2 reuse events also remain necessary.

`preload` needs `max_worker_rows * 32` metadata UB bytes plus the data ring and
headroom. With the default worker split, metadata requirements are:

| BS | Maximum rows per worker | Maximum metadata UB bytes |
| --- | ---: | ---: |
| 2048 | 615 | 19,680 |
| 4096 | 1229 | 39,328 |
| 8192 | 2458 | 78,656 |

The host checks the actual platform UB budget. It rejects an oversized `preload`
configuration explicitly; use `dma` for bounded ring storage instead. There is
no silent mode fallback. The new modes remove one dependency, but do not promise
that hardware queues remain full or that bandwidth reaches 134 GB/s: queue
capacity, Scalar task loads, remote service time and physical-link utilization
can still limit throughput. Compare both class envelopes and whole-kernel ACL
time, and inspect per-worker timers for tails.

### Applying the result to MegaMoE

This comparison changes only the standalone benchmark. Dynamic routing in the
production operator must generate metadata on the device; a host-generated
table is not a replacement for that work. If the comparison shows a gain, the
next production candidate is to generate metadata in route-index batches or
with Vector instructions into independent buffers, then let MTE3 consume it.
Metadata should still be published before the corresponding GMM tile-ready
notification, and the full operator time must include its preparation.

The three routing fields do not depend on the GET payload. When production
`TopkWeightsPrefetch` is enabled, however, metadata word 3 contains weight bits
extracted from the returned packed row. That path has an additional data
dependency and cannot simply reuse the benchmark's precomputed zeros. A device
batch implementation must keep this weight dependency and the consumers' layout
correct while moving synchronization away from the Scalar row loop.
