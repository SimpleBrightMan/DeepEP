# Unitflag dimensions and shared migration

## Current consumer: shared A8W8 (2026-11-09)

`megamoe-unitflag-exp-consumer` uses FP8 E4M3 shared weights with E8M0 MX
scales. Routed experts remain FP4; shared experts remain OFF by default.
The previous implementation is preserved at
`megamoe-unitflag-exp-consumer-shared-a8w4` (`2b7d894`). The migration notes
below describe earlier implementations, not this branch's current schedule.

With one shared expert enabled:

- AIC: shared W8 GMM1 -> original routed W4 compute -> shared W8 GMM2.
  W8 loads both matrices directly into L1; UnitFlag and the same context/slot
  cursors persist across tiles and M-groups within each shared GMM. Shared
  gate/up stays planar; routed FIX-to-AIV1 UB and paired consumers are unchanged.
- AIV0: RouteSummary -> its static half of startup Pack2 -> wait for same-core
  shared GMM1 L1 release -> routed W4 dequant -> immediate routed Reduce.
  It no longer dequantizes either shared GMM, and Reduce overlaps shared GMM2.
- AIV1: the other static half of startup Pack2 -> original cooperative loop
  (remaining Pack, communication/scale gather, activation/Reorder) -> Reduce.
  Startup has 56 owners; steady Pack still has 28. Each startup row is assigned
  exactly once. Inverse-map readers wait for all 56 Pack completion credits.
- L1 handoff is per core, not an additional full-core/rank barrier. Shared GMM1
  destroys its AIC context before releasing AIV0. AIV0 drains its routed
  prologue before releasing shared GMM2. Final local join and FP32 shared-add /
  BF16 output rounding are unchanged. S0 keeps its existing schedule.

Public shared input pairs (prepare outside forward, no hot-path conversion):

```python
# All tensors on the same NPU. FP8 weights are contiguous ND, not packed FP4/NZ.
# sw1: [1, 2 * shared_I, H], sw2: [1, H, shared_I], torch.float8_e4m3fn
# ss1: [1, 2 * shared_I, H // 64, 2], ss2: [1, H, shared_I // 64, 2]
# scales use torch.float8_e8m0fnu; shared_I is independent of routed I.
shared_l1, shared_l2 = transform_weights_for_mega_moe(
    (sw1, ss1), (sw2, ss2), shared=True)
# Pass shared_l1_weights=shared_l1, shared_l2_weights=shared_l2 to forward.
```

The benchmark CLI is unchanged (`--shared-experts 1 --shared-intermediate ...`).
For comparison against existing CPU Golden and NPU cascade fixtures, it expands
their FP4 code values exactly into FP8 before timing and keeps the scales.
This preserves the reference math; the device kernel only receives/loads W8.
The public interface also accepts FP8 values outside that fixture's FP4 subset.
The shared contract is `unitflag-shared-a8w8-early-reduce-v1`; rebuild/reinstall
the complete wheel. The routed compute schedule ID remains v5.

Performance tradeoff: shared weight payload doubles, with additional AIC B-load
pressure. Freeing AIV0 and starting Reduce earlier do not guarantee lower total
latency, especially when inbound Combine data is still unavailable.

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


## BS16384 capacity

Both S0 and S1 accept BS/capacity1..16384. The benchmark passes max(bs-values)
as capacity; the API default is16384. Hidden remains256..8192 and intermediate
remains256..16384 (both aligned256); these are separate from the BS limit.
Native/Python capability checks require a newly rebuilt matching wheel.

- At K8/CV128 a source may contribute1024 CVs to one peer (not only balanced counts).
- Generic Reduce covers400 tokens/worker at the production41-worker minimum,
  16384 remote dependency bits and all worst-case local compute-group indices.
  Including FP32 output slots its UB footprint is173696 bytes, below192KiB.
- H3584/K8 legacy Reduce is selected only at BS<=8192 AND maxPeerChunks<=512.
  A small actual BS in a large-capacity buffer can therefore use generic Reduce.
- Shared activation completion bitmap grows from3 to5 uint64 words, covering293
  tasks/core at BS16384 and shared I16384. Shared-disabled allocations stay disabled.
- Unitflag additionally doubles peer Reorder span bitmaps to16KiB each; its two
  bitmaps and per-CV counters use128..168KiB after Pack closes, outside low128KiB
  activation/Reorder DMA slots and below the192KiB minimum.

HBM allocations still depend on H/I/EP/route imbalance and may exceed available
memory for extreme combinations. The existing max_recv_rows overflow check remains;
BS16384 support does not promise every possible dimension combination fits HBM.
Existing sweep defaults remain through8192; request --bs-values 16384 explicitly.

BS16384 validation (2026-11-04): each branch passed324 native CPU planner cases,
8192 shared-task ownership/bitmap entries, and48 generic-Reduce cases exercising
BF16/FP32 output, dependency16383/self8287, metadata tails and asynchronous DMA
slot reuse. Python accepted16384 and rejected larger capacities before runtime
allocation. No CANN build or NPU run was performed for this capacity update.


## Fastest / slowest rank comment export

```bash
python3 experiments/megamoe/scripts/urma_rank_compare.py /path/bs_16384/fine/allrank.json --output bs16384_pair.txt
```

Standard library only, no rebuild or NPU run. Selects the smallest/largest rank
metadata diagnostic envelope, keeping rank clocks independent. Output is ASCII,
<=60000 characters AND bytes (change with --max-chars4096..60000, using a space
between flag/value). Both ranks are interleaved for core/expert/peer comparisons.
It reports AIC input/activation waits, shared/routed stage boundaries, per-expert
progress, limited group delays and PUT gaps, plus Reduce/shared Add tails. Composite
GMM records are unpacked using original integer ticks when available. Hidden source
waits remain unknown; missing rows are not interpreted as zero waits. Per-lane
unrecorded gaps are not claimed to be hardware idle. Header/footer disclose source
status and any budget omissions. Coarse and fine are different invocations.
