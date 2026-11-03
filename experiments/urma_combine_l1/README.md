# Shared MegaMoE communication / compute contention

Based on megamoe-commfirst-shared 81377da. Standalone native executable; no wheel
installation. Defaults EP16, BS8192, H5120, I2304, E384, TopK6, one shared expert
Is2304. Supports EP4/8/16 and BS1024..8192 (multiples of128), K1..8.

Run on user-allocated idle devices only. The runner snapshots npu-smi; it does not
reserve cards or automatically prove availability. No remote NPU run was performed.

## What is reproduced

Production balanced routing, routed tile shapes, per-AIC ownership and computation
task order, shared GMM1 head / shared GMM2 tail, contiguous unitflag shared outputs,
W4 dequant, default L2 policy. Communication is one core per remote peer on QP0:
Dispatch X+scale, then Combine BF16. Input operands are all prepared before launch.
Dispatch uses production batch/quota math (without executing Pack). Combine uses
the production merging rules assuming all CVs ready, leaving the last CV separate.
Off/on use exactly the same communication descriptors, allocation sizes and order.

There are NO readiness dependencies between communication and compute, no SO
flags, no shared-GMM1 Dispatch throttle, and no routed/shared activation waits.
GMM2 reads an independent prefilled FP8 activation, not GMM1 output. The paired
AIC/AIV0 dequant and UB/L1/L0 slot handshakes remain required for correctness.
Pack, SwiGLU, Reduce and final shared Add are excluded: this isolates GEMM/dequant
resource contention, not the complete production workload. Actual dynamic PUT
sizes depend on GMM readiness and cannot be replicated without a recorded schedule.
This is the same batch-selection policy with all data ready, not replay of a trace.

FP8 inputs/activations use 0x38, packed FP4 weights 0x11, MX scales 0x7f: nonzero
finite operands. Communication uses a rank/peer/byte-dependent pattern. After all
rounds, check received payload and check each GMM output has finite nonzero values.
This is not a numerical Golden validation or a guarantee about hardware compression.

## Timing

Keep the original device barrier and Exp launch guard, warmup10/iters20/exp50.
Each round has a device barrier before and after the mixed kernel. No per-tile
profiling or inter-stage quiet; only one final peer quiet certifies both streams.
TOTAL bandwidth = summed remote payload / core-envelope(first PUT..last quiet),
per rank then summarized. COMBINE uses its first PUT..that same final quiet and may
include pending Dispatch. DISPATCH_SUBMIT is submission speed, not completed BW.
The final local kernel join and timestamp GM writes are outside these envelopes.
Check reported GMM/communication coverage: short overlap cannot establish absence
of contention over an entire production run.

## Build / run

After activating CANN:

```bash
export SHMEM_ROOT="$PWD/third-party/shmem/install/shmem"
export CATLASS_ROOT="$PWD/third-party/catlass"
bash experiments/urma_combine_l1/build.sh
unset ASCEND_RT_VISIBLE_DEVICES
python3 experiments/urma_combine_l1/run.py \
  --ep-size 16 --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --bs 8192 --hidden 5120 --intermediate 2304 --topk 6 \
  --shared-experts 1 --shared-intermediate 2304 \
  --compute off,on,on,off --warmup 10 --iters 20 --exp-repeat 50
```

Run the same command with --bs1024 (written as `--bs 1024`) for the small case.
EP4/8: change --ep-size and --devices together. Compare TOTAL/CROSS and Combine
bandwidth, GMM envelope, and overlap coverage in summary.json / comparison.csv.
Implementation has not been compiled or run on NPU for this update.
