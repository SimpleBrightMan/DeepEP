# Shared MegaMoE communication / compute contention

Based on megamoe-commfirst-shared 81377da. Standalone native executable; no wheel
installation. Defaults EP16, BS8192, H5120, I2304, E384, TopK6, one shared expert
Is2304. Supports EP4/8/16 and BS1024..16384 (multiples of128), K1..8.

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


## No-supply-wait GMM ceiling and MFU

`--compute on --communication off` runs only production comm-first routed/shared
GMM classes and coupled W4 dequantization inside the timed mixed kernel. All
GMM1 inputs and GMM2 activations remain prefilled/nonzero. The peer AIV1s do no PUT,
Pack, activation, Reduce or Reorder work. Required intra-matmul buffer handshakes
remain. This is a practical ceiling for this GMM implementation, not dense Cube
hardware peak and not a unitflag continuous-context benchmark.

`--communication on` adds the previous data-only communication workload with the
same prepared compute inputs. Compare the MFU of these two runs to isolate resource
contention. Off+off is rejected. In compute-only mode communication columns are N/A.
The original per-round SHMEM device barriers, initial Exp50 guard, warmup and
multiple invocations remain; final local join and report writes are outside the
measured AIC interval. There is no new per-tile timer/fence.

For each rank/round, `T_AIC = max_core(end) - min_core(start)`, including internal
gaps and context drains. The printed primary MFU uses
`T = mean_round(max_rank(T_AIC))`, not the maximum of rank means. Work is balanced,
so per-card FLOPs = `6*BS*H*(TopK*I + S*Is)`; there is no further EP factor.
Default `--peak-tflops 756` is per-card dense FP8 peak. Dequant operations and
activation FLOPs are not added to the numerator. Additional rank MFU statistics
are explicitly labeled and may differ from the primary ratio. Raw intervals,
formula, model and MFU are saved in summary.json; comparison.csv includes MFU.

```bash
python3 experiments/urma_combine_l1/run.py \
  --ep-size 16 --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --bs 16384 --hidden 5120 --intermediate 2304 --topk 6 \
  --shared-experts 1 --shared-intermediate 2304 \
  --compute on --communication off --peak-tflops 756 \
  --warmup 10 --iters 20 --exp-repeat 50
```

Rebuild this demo executable after updating (Args gained a communication switch).
Do not reuse an older executable. No NPU ceiling/MFU result is claimed until this
command is run on the target hardware. Nonzero finite-output checks are retained.

Validation for MFU controls/report: five CPU report checks passed (rank-local
envelopes, round-max then mean, shared width, missing lanes and compute-off),
plus the native CPU planner cases including BS12288/16384. CANN compilation and
NPU performance/accuracy have not been run for this update.
