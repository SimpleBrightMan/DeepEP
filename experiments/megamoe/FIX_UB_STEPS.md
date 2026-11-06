> Current GMM2 placement is documented in [AIV0_GMM2_REORDER.md](AIV0_GMM2_REORDER.md). The two-step history below describes the earlier AIV1 Reorder version.

# FIX-to-UB staged experiment

Step1: routed GMM2 writes BF16 directly from L0C to paired AIV1 UB0, row stride256, single destination subBlock1. Preserves W4 x64 restore and UnitFlag. Reorder reads this UB directly and scatters to final remote/self layout; no intermediate GMM2 GM buffer/read. Routed GMM1 remains planar GM output.

A single routed FULL/EMPTY ownership credit covers BOTH GMM1 and GMM2. GMM1 is consumed immediately through its existing GM-input activation; GMM2 through direct UB Reorder. EMPTY is returned only after the consumer's final UB reads/stores, never just after FIX completion. Producer waits on PIPE_FIX, not a new Scalar fence. Zero-task cores still initialize/drain one credit. This is a smaller outstanding window than the old GM-backed two-credit pipeline and can backpressure AIC.

248KiB usable AIV1 UB required: low128KiB is reserved for paired RX/activation. Pack uses its original high128KiB placement. Scale gather and soft shared activation use high128KiB scratch after Pack.Close. Shared M128 tasks are processed as up to2 M64 sub-slices, retaining one original completion contribution. Communication persistent counters occupy240..248KiB, outside all scratch. Shared GMM outputs remain GM-backed.

Native/Python marker: unitflag-shared-fixub-gmm2-v4. Rebuild before testing. CPU contracts only; no claim of CANN/NPU qualification or speedup for this step. The next commit will add fused routed gate128+up128 GMM1 direct-UB output and change AIV1 scheduling.

## Step2 (current HEAD)

Routed GMM1 uses offline gate128+up128 packed W1 and weight-scales. Logical task width is128, physical output M256xN256 BF16 occupies the same128KiB paired mailbox as GMM2. Physical matrix block count is unchanged: old logical GMM1 task made two projections; each fused task makes one. Task prefixes, reverse GMM1 owners, AIV0 weight tiles, completion iterator and activation-ready counts all use I/128. Shared experts remain planar and GM-backed.

UbRoutedActivation consumes the mailbox in place, produces FP8 activation and E8M0 scales, then publishes its group contribution. Shared activation uses high-UB M64 sub-slices and its existing soft flags. Single-credit release occurs only after output stores finish. Both routed GMM scratch buffers and routed GMM1 per-tile done storage are absent; diagnostic snapshots mark routed GMM1 unavailable rather than exporting an uninitialized shadow. CPU Golden/cascade final-output verification remains separate.

AIV1 turn order: one bounded static Pack quantum; Dispatch for communication owners or ready scale gather for noncommunication helpers; consume the exact issued FIFO head with hard wait and direct activation/Reorder; ready Combine for communication owners; shared activation only on an otherwise idle turn. Scale gather is distributed over29-EP noncommunication AIV1s. Shared activation never touches low UB. Exact FIFO order takes precedence over stage preference, avoiding GMM1 waiting past a GMM2 that owns the only mailbox.

Native/Python schedule marker: unitflag-shared-fixub-both-v5. Required routed weight layout: gate128-up128-v1. Rebuild each commit before comparing. UB requirement248KiB (253952 bytes).

Performance tradeoffs: single mailbox can backpressure FIX/Cube; Pack-first adds delay before consuming an already-full mailbox; scale-gather parallelism shrinks from28 to29-EP; shared M128 activation now uses two smaller calls. Savings are removal of routed GMM1/GMM2 GM output roundtrips and per-tile readiness traffic. The code does not claim a speedup. Formal profile, coarse GMM and Combine/Reduce tails should all be compared. No CANN build or NPU execution was performed for these two commits; user will validate both.
