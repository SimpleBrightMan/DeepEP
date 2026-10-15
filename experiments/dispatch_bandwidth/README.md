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

## Metadata pipeline comparison

`--metadata-mode scalar|dma|preload` selects three implementations of the same
Dispatch workload. The default is `dma`, removing per-row Scalar synchronization;
select `scalar` explicitly to reproduce the original benchmark.
Native reports use schema version 3 and include `metadata_mode` and
`traffic_selection`; aggregation
rejects mixed modes. Timer CSV and trace files also identify the mode.

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

## CROSS-only core sweep

The `codex/dispatch-cross-bandwidth-sweep` branch adds `--traffic cross` and an
automatic 40-case sweep. Every selected rank participates concurrently. Each
kernel launches exactly N mixed groups and only their N AIV1 lanes perform
copies; all N workers fetch from the twelve ranks outside their four-card group.
LOCAL and GROUP own zero workers and zero output rows. The N values are logical
communication workers, not a promise of stable physical core placement.

Defaults for the sweep are BS=4096, EP=16, group size=4, E=384, TopK=8, I=1280,
`metadata-mode=dma`, six data-ring slots, 10 warmups and 20 measured iterations.
The four hidden sizes are **3712, 4096, 6144, 8192**; each is tested with
**2, 4, 6, 8, 10, 12, 16, 20, 24, 28** CROSS workers. These are forty independent
all-rank process groups, run sequentially on the same node. Task order and total
selected rows stay fixed for each H while the worker count changes.

| Hidden H | Aligned token storage | Valid scale bytes | Packed GET bytes/row | CROSS rows/card | GET bytes/card |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3712 | 3840 | 116 | 3968 | 24576 | 97,517,568 |
| 4096 | 4096 | 128 | 4224 | 24576 | 103,809,024 |
| 6144 | 6144 | 192 | 6336 | 24576 | 155,713,536 |
| 8192 | 8192 | 256 | 8448 | 24576 | 207,618,048 |

For H=3712, the benchmark keeps the requested hidden dimension 3712; 3840 is
only its packed source alignment. Output writes contain H valid token bytes,
H/32 valid scale bytes and 32 metadata bytes. CROSS-only destinations are compact
and retain the original source rank, token and TopK slot identities.

With CANN loaded and a compatible SHMEM SDK prepared, run from the repository
root on your allocated 16-card UBX node:

```bash
export SHMEM_ROOT="${SHMEM_ROOT:-$PWD/third-party/shmem/install/shmem}"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:$ASCEND_HOME_PATH/lib64:${LD_LIBRARY_PATH:-}"
# Only needed if matplotlib is missing from this Python environment:
python -m pip install -r experiments/dispatch_bandwidth/requirements-sweep.txt
bash experiments/dispatch_bandwidth/run_cross_sweep.sh
```

The shell entry point builds the native target, then starts the full sweep and
plots the results. It checks the plotting dependency before NPU work. It does
not install dependencies automatically. The run uses a unique results directory
and retains every case's logs, raw rank reports and timers along with a combined
JSON and CSV. The program prints the final paths of four PNG figures, one for
each hidden size. Plotting uses a headless backend and requires no desktop.

The default output layout is:

```text
benchmark-runs/dispatch_cross_sweep/cross_<timestamp>_<id>/
  sweep.json
  sweep.csv
  h_3712/cores_2/rank_0.json ... rank_15.json
  ...
  plots/cross_bandwidth_h3712.png
  plots/cross_bandwidth_h4096.png
  plots/cross_bandwidth_h6144.png
  plots/cross_bandwidth_h8192.png
```

To redraw from saved results, including a fresh verification of all native
reports, use the actual path printed by the run:

```bash
python experiments/dispatch_bandwidth/plot_sweep.py \
    /path/to/your-run/sweep.json --verify-raw
```

The one-click sweep always verifies raw reports before plotting. Standalone
redrawing without `--verify-raw` validates the complete embedded summaries, which
also permits moving only the combined JSON to a plotting machine. `--svg` adds
four SVG exports. Shell `--plan` and `--help` skip compilation and SDK checks.

Each figure uses the actual numeric worker counts on the x-axis and mean
**CROSS read-payload GB/s per card** on the y-axis. Rank min/max indicate variation
across the sixteen cards; plots share a y-axis range. A target reference, when
shown, is explicitly distinguished from measured values. The plotter requires a
complete, consistent Cartesian grid of successful cases. Missing or failed
measurements are not filled with zeros or presented as a completed curve.

To inspect the full plan without compiling, querying NPUs or writing output:

```bash
python experiments/dispatch_bandwidth/sweep.py --plan
```

To inspect one point after building:

```bash
python experiments/dispatch_bandwidth/run.py --traffic cross --cores 2 \
    --hidden 3712 --bs-values 4096 --metadata-mode dma --timer
```

`--traffic cross` derives `--workers 0,0,N`; explicitly passing different worker
counts is rejected. In `--traffic all`, the original three positive worker
counts and concurrent LOCAL/GROUP/CROSS behavior remain available. `preload` is
not the sweep default: two CROSS workers at BS4096 require 384 KiB of metadata
per worker before adding the data ring, which may exceed UB. Selected traffic
with zero rows is rejected before initializing ACL/SHMEM.

These curves describe the measured Dispatch copy pipeline. Changing N changes
the aggregate ring capacity and the request schedule; changing H changes packed
record size, total transferred bytes and the source working set. Interpret the
curves with these controls, rather than as a direct measurement of physical-link
peak bandwidth. No performance values are synthesized by the sweep or plotter.
