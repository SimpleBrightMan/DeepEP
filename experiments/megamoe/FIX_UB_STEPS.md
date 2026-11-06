# FIX-to-UB staged experiment

Step1: routed GMM2 writes BF16 directly from L0C to paired AIV1 UB0, row stride256, single destination subBlock1. Preserves W4 x64 restore and UnitFlag. Reorder reads this UB directly and scatters to final remote/self layout; no intermediate GMM2 GM buffer/read. Routed GMM1 remains planar GM output.

A single routed FULL/EMPTY ownership credit covers BOTH GMM1 and GMM2. GMM1 is consumed immediately through its existing GM-input activation; GMM2 through direct UB Reorder. EMPTY is returned only after the consumer's final UB reads/stores, never just after FIX completion. Producer waits on PIPE_FIX, not a new Scalar fence. Zero-task cores still initialize/drain one credit. This is a smaller outstanding window than the old GM-backed two-credit pipeline and can backpressure AIC.

256KiB AIV1 UB required: low128KiB is reserved for paired RX/activation. Pack uses its original high128KiB placement. Scale gather and soft shared activation use high128KiB scratch after Pack.Close. Shared M128 tasks are processed as up to2 M64 sub-slices, retaining one original completion contribution. Communication persistent counters occupy248..256KiB, outside all scratch. Shared GMM outputs remain GM-backed.

Native/Python marker: unitflag-shared-fixub-gmm2-v4. Rebuild before testing. CPU contracts only; no claim of CANN/NPU qualification or speedup for this step. The next commit will add fused routed gate128+up128 GMM1 direct-UB output and change AIV1 scheduling.
