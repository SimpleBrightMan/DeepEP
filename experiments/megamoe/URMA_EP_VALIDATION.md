# MegaMoE URMA EP4 / EP8 / EP16 validation

This candidate is based on CombineAdvance v2, with the updated upstream main and
SHMEM PR944 SDK. It retains the M256 compute groups and two QPs per remote peer.
It does not include the M128/N512 tile experiment.

| EP | Local experts | Communication AIV1 | Scale / SwiGLU AIV1 | Reduce AIV |
| --- | ---: | ---: | ---: | ---: |
| 4 | 96 | 3 | 25 | 53 |
| 8 | 48 | 7 | 21 | 49 |
| 16 | 24 | 15 | 13 | 41 |

All modes launch 28 AIC / 56 AIV. Every communication worker owns exactly one
remote rank. Scale gathering and activation tasks use the new non-communication
worker count; initial Pack still uses 56 AIV and subsequent Pack uses all 28 AIV1.
Only non-communication AIVs enter Reduce. Runtime size, expert ownership,
completion flags, Pack-ready allocation and readiness counters use the active EP.
The descriptor capacities remain bounded for EP16 and 96 local experts.

The Python runner obtains EP from torchrun's WORLD_SIZE. All ranks must use the
same rebuilt wheel and SDK. The native tail-schedule marker is now
`combine-ready-scan-reduce-world-v1`, so an older fixed-EP16 wheel is rejected.

## Accuracy procedure

The CPU Golden, NPU cascade and unchanged error budgets are reused from the
previous validated double-reference implementation (65a15a2). `--check-tokens 0`
checks every token and every output element, plus local expert counts and the
host plan's received-row count. No profiler or internal timers are used with the
commands below. `--exp-repeat 0` preserves the HCCL block boundary barriers.

Generate a separate CPU Golden for each EP and routing mode. Generation is a CPU
operation and does not initialize or use the NPUs. Example for EP8:

```bash
export EP=8
export ROUTING=balanced
python -m deep_ep_experimental.megamoe.golden \
  --config urma --ep-size "$EP" --ranks "0-$((EP-1))" \
  --bs-values 2048,4096,8192 --input fp8 --routing "$ROUTING" \
  --check-tokens 0 --golden-dir "$PWD/golden/ep${EP}_${ROUTING}"
```

After verifying that the selected devices are idle and allocated to the job:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --standalone --nproc_per_node="$EP" \
  -m deep_ep_experimental.megamoe.urma_benchmark \
  --config urma --bs-values 2048,4096,8192 --routing "$ROUTING" \
  --cv-rows 256 --gmm1-ahead-mgroups 12 --warmup 1 --iters 2 \
  --exp-repeat 0 --accuracy-only --check-tokens 0 \
  --golden-dir "$PWD/golden/ep${EP}_${ROUTING}" \
  --output-dir "$PWD/benchmark-runs/urma_ep${EP}_$(date +%Y%m%d_%H%M%S)"
```

For EP4, set EP=4 and expose exactly the four allocated idle devices. For EP16 on
two hosts, use torchrun's normal two-host rendezvous with eight workers per host
and pass `--allow-multi-host`. The hosts must be in one MTE-accessible UB memory
domain for completion signals, in addition to UDMA data connectivity.

Repeat with random routing. For skewed routing, the default receive allocation
(2 * maximum BS * TopK) may be insufficient; pass
`--max-recv-rows "$((EP*8192*8))"` to cover every legal concentration. That allocation
is intentionally checked and can require substantially more device memory.
`--debug-intermediates` may be added to accuracy-only runs for stage diagnosis;
it copies completed intermediate arrays to CPU and does not change the verdict.

Each rank writes `bs_<BS>/accuracy.json`; rank zero also writes the combined
`summary.json`. A passing source-level test or build does not establish NPU
accuracy. Hardware run results must be reported separately.
