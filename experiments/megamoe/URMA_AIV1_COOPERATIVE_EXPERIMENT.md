# AIV1 cooperative schedule — v3

All28 paired AIV1 workers consume their AIC's actual FIX completion. Routed activation ownership is27-(logical GMM1 tile%28); Reorder ownership is logical GMM2 tile%28. Shared activation retains its original low-priority M128 distribution.

Each turn advances bounded ready peer communication, consumes an issued completion/direct Reorder batch if present, then prioritizes scale gather, routed activation, static Pack, shared activation. No task claims or dynamic queues are introduced.

Dispatch retains RO data plus per-CV SO epochs. Combine retains128-row payloads and SO publication per original CV. A communication owner enters Combine after its Dispatch submission and its static Pack quotas finish. Reorder itself can run before Pack finishes: Pack.Close retires outstanding copies before each low-UB/event handoff.

Communication owners no longer execute payload Reorder. They scan final-layout group counts, check that all intersecting N tiles have been reordered, and may submit only a contiguous ready prefix. Self-CV owners publish after the entire CV is ready; their existing aggregate counter protects Reduce's all-arrived shortcut. Empty peers still observe peer start and publish finished.

AIC and AIV0 preserve one routed compute context. Final completion drain and resource retirement remain. Detailed tile bars are submit scopes; coarse GMM includes final drain. Full resource and validation notes: GMM_SUPPLY_EXPERIMENT.md and GMM_CONSUMER_EXPERIMENT.md.

## Direct completion to Reorder (v3)

One issued snapshot supplies at most two consecutive GMM2 tasks per turn. For each: wait for real FIX full, return the event credit, then immediately call ReorderGmm2Tile with the already-resolved descriptor. Do not poll issued again inside the batch, and stop before a following GMM1; a GMM1-led turn processes one completion then yields to local supply/activation.

The gmm2MatmulDone GM allocation, reset, per-tile write/read and debug copy are removed (the ABI address is reserved as zero). The second Reorder group/tile cursor is also removed. gmm2Done remains mandatory: only final MTE3 completion publishes a group tile contribution.

Performance tradeoff: direct Reorder has higher priority than scale/activation during this bounded batch, so it can delay new compute supply; the two-tile cap prevents an issued backlog monopolizing AIV1. FIX credit is returned BEFORE Reorder to let AIC progress concurrently. Per-tile Reorder MTE3_S and fine trace record writes remain; this change does not provide a persistent cross-tile DMA ring or guarantee a speedup. Compare formal round-max-mean plus both GMM and Combine tails.
