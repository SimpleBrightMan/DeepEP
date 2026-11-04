# Unitflag dimensions and shared migration

Base: origin/megamoe-urma-unitflag-so b904c14. Commit one generalizes H (256..8192,
multiple256), routed I (256..16384, multiple256), and TopK1..8, E384, EP4/8/16.
CLI: --hidden / --intermediate / --topk. Shared experts remain disabled in this commit.

Keep dense unitflag GMM2 output, peer-owned span reorder, 128-row Combine PUTs,
CV SO epoch flags and self-reorder readiness. The generic reducer MUST read these
per-CV epochs, not the comm-first branch's bitmap flags. Original H3584/K8 reducer
remains selected. Other K uses exact-byte metadata DMA; H is sliced into 2048 columns.
Input-scale gather uses H/32 bytes with aligned UB strides. Reorder already uses
full H rows in two bounded 64KiB slots. No performance claim without device testing.


## Commit two: shared expert and comm-first task schedule

S1 head: shared GMM1 on AIC/AIV0, only28 AIV1 perform startup Pack2. Route tasks
use comm-first order: EP4 waves of12 actual M-groups, EP8/16 per-expert lookahead.
The routed unitflag continuous context remains intact, planar gate/up for all
shapes; GMM1 completion stays immediate to unblock activation, GMM2 notifications
remain deferred. This preserves unitflag's compute implementation, not the
comm-first scatter-output GMM classes. S0 retains the first commit's original
unitflag task ordering, Pack/transport/reorder paths and reducer assignment.

S1 Dispatch takes the comm-first Pack-after-PUT ramp, persistent6 Pack slots and
small-BS shared-GMM1 throttle. Combine remains unitflag's peer-owned span reorder,
128-row data PUT plus SO epoch per complete CV. This is intentionally not the
comm-first bitmap/merged-CV transport. All remaining static Pack tasks complete
before Combine/reorder/shared activation; one ready shared activation slice runs
after each communication scan, lowest priority. Shared activation uses low128KiB;
reorder span bitmaps and per-CV counters remain above that region.

Each communication owner scatters its static self CV quota and publishes self
readiness only after writes drain. Reduce's all-incoming fast path still needs
AIC28 AND every self-reorder owner. AIV0 never increments incoming counters.
All56 AIVs statically partition token ranges for FP32 routed Reduce; AIV0 starts
after its own shared-GMM2 dequant context. The final QP quiet is after the owner's
Reduce (before workspace reuse). Final local join then six-slot vector SharedAdd
uses FP32 routed+shared arithmetic and one BF16 output rounding. No shared gate.
Shared scratch, parameters and independent I are allocated only with S1 enabled.

Use the same benchmark command with --shared-experts1 (separate arguments:
`--shared-experts 1 --shared-intermediate 2304`). H/I/TopK flags remain available.
The trace exporter/compressor includes shared/reorder scopes and actual model
metadata. Rebuild the matching wheel; capability checks reject older unitflag or
comm-first wheels. New Golden caches use --urma-shared and single_cast rounding.

Validation: source/interface review only for this migration; no claim of CANN
compilation, NPU accuracy or performance validation until those are run.
