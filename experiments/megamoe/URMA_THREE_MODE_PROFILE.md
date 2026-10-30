# Wave12 before/after: identical three-mode measurement

Branches:

| Branch | Algorithm base |
| --- | --- |
| `megamoe-urma-wave12-before-profile` | `ed3c73b`, before wave12 and deferred same-stage completion |
| `megamoe-urma-wave12-after-profile` | `604516e`, wave12 and deferred same-stage completion |

These are measurement branches. The existing `megamoe-urma-commfirst-so` is unchanged.
EP4 BS1024 is the wave12 case. Other sizes can also be compared, but do not use wave12.

Each BS runs three blocks, in order `none`, `coarse`, `fine`. Each block has the
same warmup and measured iteration count. Each block retains the existing HCCL
barrier + exp alignment (default **80 exp calls**). Before **every** MegaMoE
invocation, a one-element NPU HCCL all-reduce is enqueued on the caller stream;
the native owner stream waits for that stream as before. No host synchronize is
added between calls. This reduces inter-rank skew, not a guarantee of identical
device start timestamps. Host-Store fallback is rejected for performance runs.

Only exact `mega_moe_urma_kernel` rows in `kernel_details.csv` contribute to
performance. HCCL all-reduce, exp, workspace resets and route preparation are
excluded. `round-max-mean` is mean(max across ranks per measured iteration), not
max of per-rank means. All three modes report this statistic separately. The
last iteration's rank maximum is also printed for comparison with its trace.

## Coarse scopes and perturbation

Coarse and uninstrumented GMM/Dequant use the **same `NoopTimer` template
instantiation** of the compute loop. No per-tile timer branches, clock reads,
timer stores, cache flushes or added fences are executed inside that loop.
Algorithm-required FIX waits, flags and data transfers remain.

Only task-loop entry and final context completion are timestamped outside that
loop. Thus GMM/Dequant include the **initial input wait** as well as internal
gaps; they must not be interpreted as Cube busy time or compared directly with
fine trace's first-tile start. No claim of zero compiler register-pressure or
spill impact is made without target disassembly. Boundary timestamps and final
record writes still have cost, which the paired profiler samples expose.

Each engine emits only its applicable scopes:

- AIC: local startup join, combined GMM1/GMM2 envelope, final local join.
- AIV0: initial Pack, local startup join, combined Dequant envelope, Reduce,
  final local join.
- Communication AIV1: initial Pack, joins, first Dispatch PUT through final
  publication drain, first Combine PUT through final outbound quiet.
- Helper AIV1: initial Pack, joins, Activation loop including ScaleGather,
  later Pack and idle waits, then Reduce.

Later Pack is not separately traced. Dispatch/Combine include internal gaps;
waiting before the first PUT remains a blank region. Records are buffered
locally at boundaries and written **after the final local join**, never after
each tile. Consequently the raw trace envelope excludes final logging costs;
the profiler kernel duration includes them. Fine mode retains the existing
per-tile timer/trace behavior.

## Build and run on UBX

Use the existing Python/CANN/SHMEM environment. Build each branch separately;
do not reuse the preceding branch's installed wheel. Run the same routing,
visible devices, CV size, warmup, iterations and exp configuration on both.

```bash
cd /home/t00953803/projects/DeepEP
git fetch origin
git switch megamoe-urma-wave12-before-profile
git pull --ff-only
export PYTHON="$(command -v python3)"
export DEEPEP_BUILD_MEGAMOE=ON
export DEEPEP_MEGAMOE_TIMER=ON
bash experiments/megamoe/scripts/build.sh

export OUT="/home/t00953803/projects/DeepEP/benchmark-runs/wave12-before-$(date +%Y%m%d_%H%M%S)"
cd /tmp
ASCEND_RT_VISIBLE_DEVICES=12,13,14,15 \
DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:29604 \
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  -m deep_ep_experimental.megamoe.urma_benchmark \
  --bs-values 1024 --routing balanced --cv-rows 256 \
  --gmm1-ahead-mgroups 12 --warmup 10 --iters 20 --exp-repeat 80 \
  --timer --trace --trace-capacity 16384 --output-dir "$OUT"
```

Repeat after switching/building `megamoe-urma-wave12-after-profile`, and set
`OUT` to a new `wave12-after-*` directory. The four selected devices must be
allocated to your job. Use the CV size of your original 867us experiment on
both sides if it differs from 256. The `--timer --trace` flags remain accepted;
performance runs now collect all three modes even without them.

Output:

```text
OUT/mode_summary.csv                       # all BS × three modes
OUT/summary.json                           # profiles and per-rank raw samples
OUT/bs_1024/coarse/allrank.json             # final measured coarse invocation
OUT/bs_1024/fine/allrank.json               # final measured fine invocation
OUT/rank_0/bs_1024/{none,coarse,fine}/kernel_samples.json
OUT/rank_0/bs_1024/{none,coarse,fine}/profile/...
OUT/rank_0/bs_1024/{coarse,fine}/urma_trace.json
```

Open coarse/fine JSON in a Chrome-trace-compatible viewer such as Insight.
The existing detailed bandwidth dashboard does not yet accept the new coarse
event IDs; do not substitute fine metrics for coarse results. Coarse raw
metadata includes `coarse_stage_envelopes_us` for same-rank stage envelopes.
For trace vs profiler comparisons, use that mode's **last** profiler sample,
not the uninstrumented multi-round average. Each rank's clock is independent.

CPU tests validate collection order, sample filtering, aggregation and trace
export. They do not establish NPU correctness or prove absence of target spills.

## Independent Dispatch-to-Combine transition

Both comparison branches remove only the local Dispatch/Pack rendezvous.
Each owner completes its own Dispatch and its original static Pack quota,
then scans ready Combine CVs without waiting for other owners. The original
Pack assignment, lookahead, short/empty-peer remainder loop and completion
counter updates are unchanged. For balanced routes the remainder loop is
normally empty; skewed routes retain the work needed for correctness.

There is no `WaitLocalDispatch` event or Dispatch-done counter update.
Per-CV GMM2 readiness, SO publication, final outbound quiet and the final
local kernel join remain. Different peer owners may be in different phases.
