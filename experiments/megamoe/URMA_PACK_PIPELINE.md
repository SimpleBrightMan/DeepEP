# Persistent six-slot Pack pipeline

Both `megamoe-urma` and `megamoe-urma-gmm1Ahead` use the same Pack implementation.
Their existing GMM schedule difference is retained. Token ownership, original
CV quotas, startup two-round Pack, communication batch sizes, and the single-QP
PUT/SO protocol are unchanged.

## Submit, capture, retire

`PackPipeline` retains six UB slots and their cursor across consecutive peer
segments and Pack rounds. `PackBatch` keeps one pending ready quota and one
MTE3-to-Scalar fence. For each nonempty segment:

1. Enqueue its rows using the inherited slot cursor.
2. After `min(5, segment_rows)` rows have been submitted, wait on the **previous**
   segment's captured MTE3_S event, then atomically publish that previous quota.
3. Submit the remaining rows and capture a new MTE3_S event immediately after
   the current segment's final payload store. Do not wait for it yet.

```text
segment A: 0 1 2 3 4 5 0 1 | SetFlag(MTE3_S, A)
segment B: 2 3 4 5 0 | WaitFlag(A), AtomicAdd(A.ready, A.rows) | 1 ...
```

The old fence is captured before the new stores, so retiring it does not add a
dependency on the new segment's stores. Slot reuse still waits for the matching
MTE3_MTE2 credit. Only completed rows are added to PackReady, exactly once.

## Boundaries

- Startup: close after both initial rounds, before the local join/dequant.
- Communication workers: publish pending ready before polling PackReady for a
  PUT batch. Retain the free-slot credits/cursor across scalar UDMA PUT/SO calls.
  Scalar Combine scans/PUTs can run after each Pack round while these credits
  remain live. Close after the remaining static Pack quota, before the final
  Combine completion loop/quiet.
- Helpers: keep Pack state only across consecutive Pack work. Close before
  executing activation or ScaleGather because those stages reuse DMA event IDs.
  Close when Pack finishes and before exiting to Reduce.
- Empty segments do not invent a notification. Short segments retire the old
  quota after submitting all available rows. A final pending quota is flushed
  even when no next segment exists.

`Close` consumes the six MTE3_MTE2 credits, then waits on MTE2_S to certify the
credit waits finished before another stage rearms those IDs. It adds no PIPE_ALL.
The existing single index buffer and its per-segment MTE2_S remain; double-buffered
index prefetch is not part of this change.

## Diagnostics

Fine `Pack` bars now describe current submission plus retirement of an older
quota. They do **not** certify completion of the current quota. Explicit boundary
flush/close is recorded as `PackRetire` (ID54), with zero payload bytes to avoid
double counting. Retirement within the next Pack bar is not separately nested,
so raw trace timestamps remain non-overlapping on each engine. Coarse keeps only
the initial Pack envelope and includes its final close. It does not gain per-row
or per-segment writes.

Rebuild the wheel; startup output identifies the local pipeline as
`pack6-persistent-deferred5-reduce6-output2-v2`. Existing run commands and the
`--timer`/`--fine` controls remain unchanged. Throughput improvement and NPU
correctness require target-machine execution; CPU event-model tests do not
substitute for CANN compilation or hardware testing.
