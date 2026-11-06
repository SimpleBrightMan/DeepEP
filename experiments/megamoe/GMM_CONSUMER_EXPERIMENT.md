# Paired consumer experiment v2

This branch combines the A/B supply experiments with paired AIV1 completion and tile Reorder. See GMM_SUPPLY_EXPERIMENT.md for the complete current policy.

- GMM1 owner = 27 - (global logical tile % 28), including shared GMM1. GMM2 owner = global tile % 28. Paired AIV0 follows the exact AIC sequence; routed activation follows the reversed GMM1 owner.
- AIC queues real FIX completion (full12/13, empty14/15); issued means submission only. The paired AIV1 consumes hardware completion before publishing gmm1Done/gmm2MatmulDone.
- gmm2Done now means all stores of a reordered N tile reached the final layout. Only ReorderGmm2Tile increments it, after MTE3 drain. AIC completion or scratch readiness cannot authorize Combine.
- Every AIV1 performs its paired AIC GMM2 N256 tile scatter. Self results go directly to returnY; remote results go to source-major output. Communication owners only check complete prefixes, PUT data, and publish CV/self/finished flags.
- The two M128 reorder slots occupy low128KiB and drain before another task reuses UB/events. Communication send/self counters occupy high UB after its Pack completes. The final quiet, incoming completion observation, and invocation-local joins remain.
- Added A-window prefetch and L1 B4 with independent AIV0 UB input4/output3. Cross-core context-end is flag2, separate from B4..11 and paired completion12..15.

Native schedule marker: unitflag-shared-consumer-abc-v2. Rebuild the wheel; older kernels are rejected by the Python guard. No DCache-bypass experiment is included.

CPU validation:125 checks passed, including14,000 owned-cursor configurations and delayed DMA Reorder tests. CANN build is in progress at submission; no NPU precision or performance result is claimed for v2. Prior v1 results do not validate this combined version.
