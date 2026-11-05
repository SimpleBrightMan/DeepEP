# Routed GMM completion on paired AIV1

Branch: `megamoe-unitflag-exp-consumer`.

This is an independent experiment based on the unitflag shared implementation.
It does not include the A-window prefetch or deeper B-buffer experiments. The
normal benchmark command selects it automatically; no extra runtime switch is
required. The native/Python schedule marker is
`unitflag-shared-consumer-fix-v1` so an older wheel is rejected.

## Completion protocol

Routed GMM1 and GMM2 retain their immutable, continuous GM output layouts. AIC
does not perform a Scalar `FIX_S` wait after each routed tile. Instead it enqueues
these operations in its FIX pipe:

```text
wait empty[sequence % 2]
tile's actual FIX writes
arrive full[sequence % 2]
```

Scalar then advances an `issued` counter in a per-core 128-byte GM slot. That
counter indicates submission only, never data readiness. Paired AIV1 reconstructs
the same tile order from the immutable device-generated compute plan. It waits
for the matching hardware `full` event, waits for its MTE2-to-Scalar completion,
and only then publishes the original `gmm1Done` or `gmm2MatmulDone`/`gmm2Done`.
This prevents a GM descriptor-ring overwrite: waiting in the FIX pipe does not
stop Scalar from overwriting a mutable descriptor, so no such ring is used.

Channels 12/13 are full, 14/15 are empty, targeting the paired AIV1. The existing
AIV0 B-supply channels 6..9 and context-retirement channel 10 remain separate.
Every AIV1 initializes two credits, including zero-task cores; every AIC consumes
the final two credits at kernel completion. UnitFlag and the existing L0/L1
resource lifetimes are unchanged. Shared GMM completion remains independent.

## Cooperative AIV1 work

All 28 AIV1 workers consume completion and activate the GMM1 tiles owned by their
paired AIC (`global tile % 28`). Input scale gather is spread over the same 28
workers. Peer communication ownership stays one core per peer. Each bounded
communication turn yields to completion, then scale gather, routed activation,
and static Pack; shared activation remains the last-priority one-tile task.

Pack retains its outgoing-row static partition. Incoming local AIC tiles and
outgoing local Pack rows are different route domains and cannot share tile IDs.
Pack is closed before another DMA stage takes its events. Reorder's persistent
high-UB bitmaps become live only after this worker finishes all Pack quotas.

Dispatch uses RO payload PUTs followed by the existing same-QP SO publication,
without a quiet per CV. Combine retains 128-row payload PUTs and one SO flag per
original CV. The required final quiet and incoming completion observation remain.

## Validation and interpretation

- `owned_compute_cursor_test.cpp` exercises the actual production cursor against
  an independent full enumeration: 14,000 core configurations, including 310
  empty-core cases.
- `routed_fix_channel_test.cpp` uses the actual channel methods with ordered pipe
  queues to check two-slot reuse and final credit retirement for zero, odd, and
  long task sequences. This model does not validate device memory ordering.
- Native CANN compilation and NPU accuracy remain separate checks. Do not infer
  a speedup from CPU checks or compilation.

Fine mode records tile submission on AIC and real output-completion waits on
AIV1. The AIC final drain remains included in its GMM envelope. Compare formal
profile and coarse GMM spans; summing AIC tile bars alone is no longer a complete
compute-duration measurement. AIV1's completion work can interrupt Pack, so this
experiment must also be assessed for communication and activation tail changes.
