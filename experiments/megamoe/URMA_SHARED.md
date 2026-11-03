# Comm-first shared expert (S0/S1)

This branch has two commits: dimension generalization, then shared-expert support.
The production `megamoe-urma` branch is unchanged. Shared gating is not implemented.

## Dimensions and API

- Hidden H: multiples of 256, 256..8192.
- Routed intermediate I: multiples of 256, 256..16384.
- Shared intermediate Is: independent multiples of 256, 256..16384.
- Exactly S0 (disabled) or S1; EP4/8/16, E384, TopK8.

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

All 28 AIV1 workers perform routed Reduce. Incoming completion monitoring continues
inside its scans. Communication owners defer their final QP quiet until their
Reduce completes; no source buffer is reused before quiet. AIC completion used by
the routed Reduce fast path is published after ROUTED GMM2, before shared GMM2.
AIV0 stays with the shared GMM2 dequantization and does not perform routed Reduce.

After a local SyncAll, all 56 AIV workers add shared output to routed output,
followed by the final local join. There is no added cross-rank barrier.
The numerical contract is BF16(BF16(routed FP32 weighted sum) + shared BF16 output),
with the final addition performed in FP32. Golden/cascade explicitly model this
rounding and the independently configured shared intermediate width.

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
