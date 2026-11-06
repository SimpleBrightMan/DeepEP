# Paired consumer experiment v3

This branch combines the A/B supply experiments with paired AIV1 completion and tile Reorder. See GMM_SUPPLY_EXPERIMENT.md for the complete current policy.

- GMM1 owner = 27 - (global logical tile % 28), including shared GMM1. GMM2 owner = global tile % 28. Paired AIV0 follows the exact AIC sequence; routed activation follows the reversed GMM1 owner.
- AIC queues real FIX completion (full12/13, empty14/15); issued means submission only. The paired AIV1 consumes hardware completion before publishing gmm1Done or directly running same-core GMM2 Reorder.
- gmm2Done now means all stores of a reordered N tile reached the final layout. Only ReorderGmm2Tile increments it, after MTE3 drain. AIC completion or scratch readiness cannot authorize Combine.
- Every AIV1 performs its paired AIC GMM2 N256 tile scatter. Self results go directly to returnY; remote results go to source-major output. Communication owners only check complete prefixes, PUT data, and publish CV/self/finished flags.
- The two M128 reorder slots occupy low128KiB and drain before another task reuses UB/events. Communication send/self counters occupy high UB after its Pack completes. The final quiet, incoming completion observation, and invocation-local joins remain.
- Added A-window prefetch and L1 B4 with independent AIV0 UB input4/output3. Cross-core context-end is flag2, separate from B4..11 and paired completion12..15.

Native schedule marker: unitflag-shared-consumer-direct-v3. Rebuild the wheel; older kernels are rejected by the Python guard. No DCache-bypass experiment is included.

CPU validation:125 checks passed, including14,000 owned-cursor configurations and delayed DMA Reorder tests. This is the historical v2 CPU matrix, not v3 NPU validation. v3 has additional batch-driver checks; no v3 NPU precision or performance result is claimed.

## Direct completion to Reorder (v3)

One issued snapshot supplies at most two consecutive GMM2 tasks per turn. For each: wait for real FIX full, return the event credit, then immediately call ReorderGmm2Tile with the already-resolved descriptor. Do not poll issued again inside the batch, and stop before a following GMM1; a GMM1-led turn processes one completion then yields to local supply/activation.

The gmm2MatmulDone GM allocation, reset, per-tile write/read and debug copy are removed (the ABI address is reserved as zero). The second Reorder group/tile cursor is also removed. gmm2Done remains mandatory: only final MTE3 completion publishes a group tile contribution.

Performance tradeoff: direct Reorder has higher priority than scale/activation during this bounded batch, so it can delay new compute supply; the two-tile cap prevents an issued backlog monopolizing AIV1. FIX credit is returned BEFORE Reorder to let AIC progress concurrently. Per-tile Reorder MTE3_S and fine trace record writes remain; this change does not provide a persistent cross-tile DMA ring or guarantee a speedup. Compare formal round-max-mean plus both GMM and Combine tails.
