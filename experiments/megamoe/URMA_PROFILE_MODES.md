# Unified MegaMoE URMA measurement

`megamoe-urma` and `megamoe-urma-unitflag-so` use this identical capture policy.
They do not share operator algorithms. The main branch incorporates the
wave12-after operator optimizations; unitflag retains its compute/Reorder/Pack
and communication schedule.

| Options | Warmup10 + iters20 | Captured data |
| --- | ---: | --- |
| none | 30 calls | 20 uninstrumented device kernel durations |
| `--timer` | 31 calls | above + one coarse diagnostic |
| `--timer --fine` or `--fine` | 32 calls | above + one fine diagnostic |

`--trace` is a compatibility alias for `--fine`; omit it for coarse only.
There is exactly ONE profiler context per BS, no diagnostic warmup block:

```python
with profiler:
    guard.begin()                     # original HCCL barrier + exp50
    for _ in range(warmup + iters):
        device_all_reduce()
        megamoe()                     # no internal timer
    if timer:
        device_all_reduce()
        megamoe(coarse=True)           # separate buffer, once
    if fine:
        device_all_reduce()
        megamoe(fine=True)             # separate buffer, once
    guard.end()
```

The existing exp guard is preserved, default repeat=50. Every invocation,
including warmup and diagnostics, is preceded by a one-element HCCL all-reduce.
The native owner stream inherits caller-stream dependencies. No host synchronize
is inserted per call. A device collective reduces skew, not a guarantee of
identical physical device start times. The host Store fallback is rejected.

Only exact `mega_moe_urma_kernel` rows from profiler CSV contribute to timing.
Exp, HCCL all-reduce, route preparation and workspace reset kernels are excluded.
`round-max-mean` = mean of each formal round's cross-rank maximum; the two
diagnostic calls never enter that mean. Coarse/fine print their own single-call
rank-max/min, rather than pretending to be multi-round measurements.

## Coarse boundaries

GMM1/GMM2 share one compute envelope, and both dequant stages share another.
Both coarse and uninstrumented modes use the same logging-free compute template.
No per-tile logging clocks, GM writes, cache flushes or added fences run inside
that loop. Existing algorithm waits/flags are retained. Timestamps are taken at
entry and after the final context completes, so initial input waits ARE included.
Target compiler spill/register-pressure effects are not ruled out by CPU tests.

Other envelopes: initial Pack only, startup/final LocalSync, first Dispatch PUT
through last outgoing Dispatch completion, first Combine PUT through final
outbound quiet, helper Activation loop (including later Pack and ScaleGather),
and Reduce. Main Dispatch completion is the final QP quiet covering X/scale and SO CV ready flags;
unitflag Dispatch completion is the last CV's quiet, which includes its SO flag.
Unitflag Combine includes interleaved Reorder and may wait for self Reorder
before its existing final quiet. No Reorder algorithm is changed by diagnostics.

Coarse records are written after the final local join. Its raw trace envelope
does not include final logging cost; its profiler kernel duration does. Fine
retains detailed instrumentation and can perturb producer/consumer timing.

## Build and run

Both branches need a native rebuild once to add the coarse argument/ABI.
Use the same CANN, Python, routing, CV size and device allocation for comparisons.
Keep HBM1.4T and HBM1.6T runs in separate output directories.

```bash
cd /home/t00953803/projects/DeepEP
# Select megamoe-urma or megamoe-urma-unitflag-so before each build.
git pull --ff-only
export PYTHON="$(command -v python3)"
export DEEPEP_BUILD_MEGAMOE=ON
export DEEPEP_MEGAMOE_TIMER=ON
bash experiments/megamoe/scripts/build.sh

export OUT="/home/t00953803/projects/DeepEP/benchmark-runs/hbm14-$(git branch --show-current)-$(date +%Y%m%d_%H%M%S)"
cd /tmp
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:29608 \
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  -m deep_ep_experimental.megamoe.urma_benchmark \
  --bs-values 1024,2048,4096,8192 --routing balanced --cv-rows 256 \
  --gmm1-ahead-mgroups 12 --warmup 10 --iters 20 --exp-repeat 50 \
  --timer --output-dir "$OUT"
```

Add `--fine --trace-capacity 16384` only when detailed diagnosis is needed.
For EP4 or EP16 change BOTH visible device list and `--nproc-per-node`; use
devices allocated to your job. This task does not launch NPU workloads.

Outputs:

```text
summary.json, mode_summary.csv
rank_0/bs_1024/profile/...                 # one profiler directory per BS/rank
rank_0/bs_1024/kernel_samples.json         # every sample, with explicit phase
rank_0/bs_1024/coarse/{urma_trace.json,urma_trace.bin,kernel_samples.json}
rank_0/bs_1024/fine/{urma_trace.json,urma_trace.bin,kernel_samples.json}
bs_1024/coarse/allrank.json
bs_1024/fine/allrank.json
```

Only requested diagnostic directories exist. Compare each trace against the
matching diagnostic profiler sample, not the formal average. Rank clocks are
independently zeroed; cross-rank absolute timestamps cannot establish simultaneity.
Use Insight for coarse traces; detailed dashboards may not recognize coarse IDs.
