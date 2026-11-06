# AIV1 cooperative schedule — v2

All28 paired AIV1 workers consume their AIC's actual FIX completion. Routed activation ownership is27-(logical GMM1 tile%28); Reorder ownership is logical GMM2 tile%28. Shared activation retains its original low-priority M128 distribution.

Each turn advances bounded ready peer communication, consumes an issued completion if present, then prioritizes scale gather, routed activation, paired tile Reorder, static Pack, shared activation. No task claims or dynamic queues are introduced.

Dispatch retains RO data plus per-CV SO epochs. Combine retains128-row payloads and SO publication per original CV. A communication owner enters Combine after its Dispatch submission and its static Pack quotas finish. Reorder itself can run before Pack finishes: Pack.Close retires outstanding copies before each low-UB/event handoff.

Communication owners no longer execute payload Reorder. They scan final-layout group counts, check that all intersecting N tiles have been reordered, and may submit only a contiguous ready prefix. Self-CV owners publish after the entire CV is ready; their existing aggregate counter protects Reduce's all-arrived shortcut. Empty peers still observe peer start and publish finished.

AIC and AIV0 preserve one routed compute context. Final completion drain and resource retirement remain. Detailed tile bars are submit scopes; coarse GMM includes final drain. Full resource and validation notes: GMM_SUPPLY_EXPERIMENT.md and GMM_CONSUMER_EXPERIMENT.md.
