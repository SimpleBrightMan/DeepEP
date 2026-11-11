# URMA shared expert split experiment

Branch: `megamoe-shared-opt`, based on `megamoe-submit` commit `4d180ea`.

Only S1 with **BS >= 4096** changes. S0 and S1 with BS < 4096 retain their
existing compute/activation assignment and small-BS Dispatch throttle.

Each AIC executes shared GMM1 groups `[0, ceil(groups/2))`, its routed GMM1/GMM2
work, the remaining shared GMM1 groups, then shared GMM2. Groups are M256 and
keep their original global row/tile indices. BS12288 gives 24 + 24 groups;
an odd count places the extra group in the leading half. No new all-core or
cross-rank barrier is introduced. "Routed complete" is local to each paired
AIC/AIV0, rather than a new global join.

All 28 AIV0 workers own the existing shared activation tasks (M128 x N256,
implemented as the same two M64 operations), keeping the original numerical
sequence and completion counts. After startup Pack, they opportunistically
activate ready leading-half tiles until their paired AIC releases shared L1.
Then they run routed W4 dequantization exclusively. They release routed L1
**before** waiting for remaining shared activations; otherwise the AIC cannot
produce the shared GMM1 tail. They complete remaining activation work before
their static routed Reduce partition. AIV1 retains Pack/communication/scale
gather/routed activation/reorder/Reduce and does not scan shared activation.

Shared GMM2 still waits for each M-group's exact activation count. A separate
GMM1 context owns each half; it drains and destructs before L1 is handed off.
Shared AIV0 activation never aliases a live W4 dequant context. Existing GM
workspaces, ABI, weights, quantization and math are unchanged.

Tradeoffs: one additional shared GMM1 context retirement; AIV0 Reduce starts
after its shared activation tasks, while AIV1 can reach Reduce earlier. Moving
shared activation can also change HBM contention. Measure end-to-end
round-max-mean and Combine/Reduce tail, not only the GMM envelope.

Fine trace reuses `SharedActivation` (ID57), now on AIV0 for this path. Coarse
trace emits two nonoverlapping `CoarseActivation` (ID50) intervals around the
routed Dequant interval; the gap is not counted as continuous activation.
The coarse intervals include readiness waits; fine records describe actual
activation tasks. Trace schema/event IDs are unchanged.

Host tests cover the 4096 threshold, odd groups, M tails, global ownership,
exactly-once activation and the GMM2 readiness counts. They do not establish
NPU correctness or a performance improvement.
