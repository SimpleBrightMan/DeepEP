# AIV0 owns routed GMM2 Reorder

This revision replaces the unsuccessful wave6-12-6 commit. Scheduling returns to the pre-wave policy: EP4 with shared experts uses wave12; other cases use expert batches and gmm1_ahead_mgroups. No wave6-12-6 helper is included.

## UB and work ownership

GMM1 remains on AIC with paired AIV0 weight dequantization using4x16KiB FP4 inputs +3x32KiB FP8 outputs (160KiB). FIX output still targets paired AIV1 UB0, and AIV1 performs activation.

For routed GMM2, paired AIV0 changes only its UB layout to3x16KiB inputs +2x32KiB outputs (112KiB). GMM2 FIX targets AIV0 UB[120,248)KiB, which it scatters directly into remote source-major gmm2Output or self returnY. There is8KiB separation between the dequant workspace and RX. L1 B retains four64KiB buffers and its cursor/credits across mode switches. GMM1's inner loops remain specialized4/3; GMM2's are specialized3/2.

The AIV1 completion cursor now skips GMM2 entirely; the AIC issued counter counts only GMM1. AIV1 handles Pack, Dispatch/Combine, its GMM1 activation, shared activation, and incoming completion bookkeeping. Noncommunication AIV1s still handle input-scale gather. Communication owners publish self-CV readiness after AIV0's final-layout group counts reach the required N-tile count.

## AIV0 supply/consume pipeline

Within a run of owned GMM2 tiles:

    shrink/drain old GMM1 UB; grant one GMM2 EMPTY
    supply B(tile0)
    supply B(tile1); wait FIX(tile0); scatter(tile0); return EMPTY
    supply B(tile2); wait FIX(tile1); scatter(tile1); return EMPTY
    ...
    wait FIX(last); scatter(last); do NOT return EMPTY
    restore GMM1 UB, or retire context at invocation end

At most one next tile's weights are supplied ahead. Supplying unlimited future weights could block on B/L0C resources while the old output still owns RX. The final output is consumed before GMM1 expands back into the overlapping UB region; withholding the last EMPTY prevents a future GMM2 FIX from overwriting GMM1's workspace. A subsequent GMM2 stage grants its initial credit only after shrinking safely again. Empty stages create no credit.

B supply channels remain4..11, context retirement2; GMM1/AIV1 uses FULL12/EMPTY14 and GMM2/AIV0 uses FULL13/EMPTY15. No extra cross-rank synchronization or device reset is introduced. Reorder increments gmm2Done only after its MTE3 stores complete.

## Limits and validation

Native marker: unitflag-shared-aiv0-gmm2-v7. Weight layout and arithmetic are unchanged (routed fused gate128/up128, W4 x64 restoration and BF16 output).

This may reduce communication-core pressure, but UB-mode switches need local retirement and Reorder competes with AIV0's B-supply MTE3 traffic. A one-tile lookahead helps overlap; it is not a speedup guarantee. Coarse Dequant now includes AIV0 Reorder, and AIV0 Gmm2Drain means the hardware result wait.

CANN build and isolated native import passed on177.22.25.1. CPU checks include GMM1-only consumer ownership, both UB destinations, direct scatter with delayed DMA, and32 mixed-stage credit/liveness scenarios. No NPU numerical or performance test was run for this revision.
