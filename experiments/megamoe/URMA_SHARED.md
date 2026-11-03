# Comm-first shared expert (S0/S1)

This branch started with dimension generalization and shared-expert support.
The production `megamoe-urma` branch is unchanged. Shared gating is not implemented.

## Dimensions and API

- Hidden H: multiples of 256, 256..8192.
- Routed intermediate I: multiples of 256, 256..16384.
- Shared intermediate Is: independent multiples of 256, 256..16384.
- Exactly S0 (disabled) or S1; EP4/8/16, E384, TopK 1..8.

Pass `num_shared_experts=1, shared_intermediate_hidden=Is` to `UrmaBuffer`.
Transform shared weight pairs with `transform_weights_for_mega_moe(..., shared=True)`
and supply `shared_l1_weights`/`shared_l2_weights` to `forward_prepared` or `forward`.
Shared weights/scales are separate tensors; their intermediate dimension need not
match routed weights. Model code is responsible for providing the same replicated
shared weights on ranks when that is the intended model.

S0 uses the original `mega_moe_urma_kernel` entry, selecting ProcessRouted on the
Host. S1 launches a separate `mega_moe_urma_shared_kernel`; its device branch and
register requirements are not injected into the S0 entry. S0 retains the original
ProcessRouted and communication bodies. It retains the
56-AIV initial Pack, original startup join, routed tasks, helper priorities and
41/49/53-worker Reduce. There are no shared task polls or buffers in this path.
H3584 keeps the original Reduce implementation; other H values use the generic
column reducer introduced in commit one. This is a scheduling guarantee, not a
claim of bit-identical compiler output or measured zero performance difference.

## S1 execution

AIC/AIV0 immediately run all local shared GMM1 work, then routed GMM1/GMM2, then
shared GMM2. Shared GMM2 waits for the corresponding shared activation M-group.
Both shared GMMs use contiguous row-major output and the existing unit-flag MMAD
policy. GMM1 completion flags are published only after the captured FIX event
completes; same-stage next-tile submission may precede the previous notification.

Only the 28 AIV1 workers perform the first two Pack rounds in S1. No startup
SyncAll blocks the AIC/AIV0 shared GMM1. Existing PackReady and input-scale-ready
checks protect remote and self input rows before routed GMM1.

For S1 with BS < 4096, each peer's communication owner may submit at most
`max(1, floor(BS / 1024))` Dispatch batches before all 28 local AICs finish
shared GMM1. A batch is one merged X PUT + Scale PUT + the covered CV's SO flags.
While this gate is closed, batching follows the normal 1,1,2,... ramp even if
Pack is already complete; it cannot take the send-all-remaining shortcut.
Thus BS1024 admits one batch, BS2048 two, and BS3072 three (1+1+2 CVs).
At its limit, the owner flushes pending Pack-ready notifications and polls one
local completed-AIC counter with 200-cycle backoff. Every AIC increments it once
after its shared GMM1 context finishes, including cores with no assigned tiles.
The counter uses an existing reserved localFlags slot, reset before each call.
It does not wait for shared activation/GMM2, which would depend on routed compute.
No quiet or rank barrier is added. Pack task ownership and quotas are unchanged.
BS >= 4096 and S0 retain the unrestricted Dispatch path. Fine traces show this
throttle wait as WaitDispatch; coarse Dispatch includes the wait in its envelope.

Shared activation tasks are static over all 28 AIV1 workers: worker, worker+28,
worker+56, ... Each task is one M128 x N256 activation slice. Polling is bounded,
skips unready tasks, and executes at most one actual shared slice per turn.

- Communication workers finish their own Dispatch submissions and any remaining
  static Pack quota. Each subsequent full Combine scan may submit ready CVs,
  then execute at most one ready shared activation slice. Empty/short peers still
  perform their original Pack quota. Outbound finished is published once as soon
  as all Combine PUTs are submitted, independently of remaining shared activation.
- Helpers keep routed activation > input-scale gather > Pack > shared activation.
  Shared activation has no phase-dependent priority promotion.
  The shared-mode Combine submitted bitmap resides above the activation UB region;
  activation cannot overwrite it between communication scans.
- Every worker completes its static shared activation quota before entering Reduce.

All 56 AIV0/AIV1 workers perform routed Reduce with static, disjoint, contiguous
token ranges. Worker ID is `2 * block + subblock`; each worker owns `BS / 56`
tokens plus one for the first `BS % 56` workers. No token or TopK sum is split
between workers, and no work stealing or GM atomic accumulation is introduced.
AIV0 enters Reduce immediately after its own shared GMM2 dequant context finishes
and a local pipeline drain releases its UB/events. It does not wait for other
workers or a global join. AIV1 enters after its existing communication/activation
work. Incoming completion monitoring and counters remain solely with the original
AIV1 communication owners; AIV0 reads readiness but never increments these counters.
Communication owners defer their final QP quiet until their Reduce completes;
no source buffer is reused before quiet. AIC completion used by the routed Reduce
fast path is published after ROUTED GMM2, before shared GMM2. The existing local
join after all workers finish Reduce/shared compute remains in place.

After a local SyncAll, all 56 AIV workers add shared output to routed output,
followed by the final local join. There is no added cross-rank barrier.
The numerical contract matches main Unpermute: BF16(routed FP32 weighted sum +
FP32(shared BF16 output)), with only one final BF16 rounding. S1 Reduce writes a
separate BS*H*4-byte FP32 buffer; S0 still directly writes its original BF16 output.
All 56 AIVs perform final summation using six 2048-element slots (144 KiB UB):
MTE2 reads FP32 routed/BF16 shared, VEC casts shared/Add/casts output, MTE3 writes
BF16. Slot reuse is guarded by V_MTE2/MTE3_V; no per-tile MTE3_S or PIPE_ALL.
Only the terminal drain precedes the existing local join. Golden/cascade use
single_cast; previous shared Golden caches must be regenerated. This trades
2 additional routed bytes per write/read for matching main's FP32 precision.

## Build and run

In a terminal with the matching CANN environment and Python selected:

```bash
export PYTHON="$(command -v python3)"
export DEEPEP_BUILD_MEGAMOE=ON
export DEEPEP_MEGAMOE_TIMER=ON
bash experiments/megamoe/scripts/build.sh

export OUT="$PWD/benchmark-runs/commfirst-shared-$(date +%Y%m%d_%H%M%S)"
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:29604 \
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  -m deep_ep_experimental.megamoe.urma_benchmark \
  --bs-values 1024,2048,4096,8192 \
  --hidden 3584 --intermediate 1280 --shared-experts 1 --shared-intermediate 2560 \
  --cv-rows 256 --warmup 10 --iters 20 --exp-repeat 50 \
  --timer --output-dir "$OUT/s1"
```

For S0 use `--shared-experts 0` and another output directory. Add `--fine
--trace-capacity 16384` only for detailed diagnostics. The existing single-profiler,
per-call device all-reduce and exp50 launch alignment remain unchanged.

Shared fine events: SharedGmm1Tile/Drain, SharedActivation, SharedGmm2Tile/Drain,
SharedAdd. Coarse GMM/Dequant cover the full head+routed+tail sequence. S1 Combine
coarse uses CoarseSharedCombineSubmit: final quiet is deferred past Reduce, so the
submission envelope must not be treated as independently completed bandwidth.

## Accuracy

Old Golden caches need regeneration because the reference source hash changed.
For a quick EP4/BS1024 check (CPU Golden generation uses no NPU):

```bash
"$PYTHON" -m deep_ep_experimental.megamoe.golden --config urma \
  --ep-size 4 --ranks 0-3 --bs-values 1024 --input fp8 --routing balanced \
  --hidden 3584 --intermediate 1280 --shared-experts 1 --shared-intermediate 2560 \
  --urma-shared --check-tokens 16 --golden-dir benchmark-golden/shared
```

Use the same dimensions/routing and add `--golden-dir benchmark-golden/shared
--check-tokens 16 --accuracy-only` to the URMA benchmark command (omit timer/fine
for the accuracy-only run). This compares against CPU Golden and NPU cascade.
Compilation and CPU model checks are not NPU accuracy or performance validation.

## Validation status (2026-11-03)

CANN 9.2.0 device kernels and native extensions built successfully in the isolated
25-machine directory. Selected Python checks (199), 90 queued-DMA generic-Reduce
cases, 40 shared-task ownership/tail cases, and original routed argument-prefix
layout assertions passed. S0 Process/communication bodies are checked against the
dimension commit. No NPU correctness or performance run has been performed.

Shared sum pipeline update (2026-11-03): 105 selected CPU tests passed, including
execution of the real FP32 Reduce + SharedAdd headers under three queued-engine
schedules, partial tiles, empty workers and repeated slot reuse. Two additional
independent Golden single-cast tests passed. The queued model detects cases where
premature BF16 rounding changes the output. CANN device/native compilation passed;
no NPU accuracy or performance execution was performed for this update.

## Variable TopK

`--topk 1..8` overrides routed TopK in both urma_benchmark and golden; default 8.
The API accepts `num_topk=1..8`. Shared count/width are independent of TopK.
Metadata DMA uses exact-byte DataCopyPad for K != 8; route indices, weights,
readiness dependencies and output strides use the selected K. UB/flag capacities
remain sized for K8; larger K is rejected. H3584/K8/S0 retains its original reducer.
New traces carry BS/world/model metadata; the shared dashboard reads these values.
Rebuild the wheel before using the new API/CLI; the dimension capability marker
rejects older wheels. TopK changes require regenerating the corresponding Golden.

TopK update validation: 170 Python API/Golden/trace/scheduling regression tests
passed, and dashboard TopK6 metadata inference passed. Native planner and queued
C++ model cases have been extended to K1..8, but their execution is currently
blocked by the local compiler sandbox. Remote CANN upload approval timed out;
CANN compilation and NPU validation for this TopK update remain outstanding.

## Compact diagnostic export

`urma_shared_trace_to_text.py` accepts single-rank and allrank Chrome JSON.
It reports every rank's stage boundaries, then selected-rank lane/peer summaries.
Default selection is the rank with the largest metadata diagnostic envelope.
`--detail` preserves all available waits, quiet/drain and final synchronization;
no gaps are removed or clocks aligned across ranks. Hidden source waits cannot
be reconstructed. Event positions are rounded to two decimals for text only.

```bash
python3 experiments/megamoe/scripts/urma_shared_trace_to_text.py /path/coarse/allrank.json --detail --output coarse.txt
python3 experiments/megamoe/scripts/urma_shared_trace_to_text.py /path/fine/allrank.json --detail --output fine.txt
```

Use `--ranks 0,7` to inspect specific ranks, or `--ranks all` for every rank.
Optional `--cores 0,1,15-17` limits detail; the all-rank summary remains complete.
Start with one bad BS, include the exact command, git revision, formal profile
result and both diagnostic files. Use `--fine --trace-include-waits` when collecting
new diagnostics. Coarse and fine are different invocations, not interchangeable
hardware timelines. PUT gaps alone do not distinguish readiness from scheduling.
