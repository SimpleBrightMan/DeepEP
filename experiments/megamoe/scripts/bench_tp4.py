#!/usr/bin/env python3
# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""TP4 MegaMoE (A8W4) accuracy, zero-sync check, timer-off E2E perf, and swimlane trace runner.

Launch example (4 NPUs on single node):
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\
      experiments/megamoe/scripts/bench_tp4.py --mode all --output-dir /tmp/tp4_run
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

import numpy as np


def make_routing(seed: int, rank: int, bs: int, hidden: int, num_experts: int, topk: int, pattern: str = "random"):
    from deep_ep_experimental.megamoe.reference import bf16

    rng = np.random.default_rng(seed + rank * 1009 + bs)
    x = bf16(rng.normal(0.0, 0.2, (bs, hidden)).astype(np.float32))
    if pattern == "balanced":
        start = rank * bs * topk
        seq = np.arange(start, start + bs * topk, dtype=np.int64)
        ids = (seq % num_experts).reshape(bs, topk)
    elif pattern == "skewed":
        ids = np.broadcast_to(np.arange(topk, dtype=np.int32), (bs, topk)).copy()
    else:
        ids = np.array([rng.choice(num_experts, topk, replace=False) for _ in range(bs)], dtype=np.int32)
    weights = rng.uniform(0.05, 1.0, size=(bs, topk)).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    return x, ids.astype(np.int32), weights


def expert_forward_ref(x_deq: np.ndarray, w1: np.ndarray, w2: np.ndarray, route_w: np.ndarray | None = None):
    from deep_ep_experimental.megamoe.reference import bf16, dequantize_input, quantize_input

    first = bf16(x_deq @ w1.T)
    gate, up = np.split(first, 2, axis=-1)
    act = gate / (1.0 + np.exp(np.clip(-gate, -80.0, 80.0))) * up
    if route_w is not None:
        act = act * route_w[:, None]
    act_bf16 = bf16(act)
    act_q = dequantize_input(*quantize_input(act_bf16))
    return bf16(act_q @ w2.T)


def tp_rank_routed_partial_ref(
    selected_x_deq: np.ndarray,
    selected_ids: np.ndarray,
    selected_weights: np.ndarray,
    raw_routed: tuple,
    pre_quant: bool,
) -> np.ndarray:
    from deep_ep_experimental.megamoe.reference import unpack_weight

    w1_all = unpack_weight(raw_routed[0], weight_bits=4)
    w2_all = unpack_weight(raw_routed[1], weight_bits=4)
    partial = np.zeros_like(selected_x_deq, dtype=np.float32)
    used_experts = np.unique(selected_ids)
    for exp_id in used_experts:
        if exp_id < 0:
            continue
        tok_idx, slot_idx = np.where(selected_ids == exp_id)
        if tok_idx.size == 0:
            continue
        rw = selected_weights[tok_idx, slot_idx]
        out = expert_forward_ref(selected_x_deq[tok_idx], w1_all[exp_id], w2_all[exp_id], rw if pre_quant else None)
        if pre_quant:
            np.add.at(partial, tok_idx, out)
        else:
            np.add.at(partial, tok_idx, out * rw[:, None])
    return np.asarray(partial, dtype=np.float32)


def shared_ref(
    selected_x_bf16: np.ndarray,
    selected_x_fp8_deq: np.ndarray,
    shared_kind: str,
    shared_raw,
) -> np.ndarray:
    from deep_ep_experimental.megamoe.reference import bf16, expert_forward, unpack_weight

    if shared_kind == "none":
        return np.zeros_like(selected_x_bf16, dtype=np.float32)
    if shared_kind == "fp8":
        out = np.zeros_like(selected_x_fp8_deq, dtype=np.float32)
        for w1, w2 in zip(*(unpack_weight(p, weight_bits=8) for p in shared_raw)):
            out += expert_forward(selected_x_fp8_deq, w1, w2)
        return out
    if shared_kind == "bf16":
        sw1_np, sw2_np = shared_raw
        out = np.zeros_like(selected_x_bf16, dtype=np.float32)
        for s in range(sw1_np.shape[0]):
            first = bf16(selected_x_bf16 @ sw1_np[s].T)
            gate, up = np.split(first, 2, axis=-1)
            act = bf16(gate / (1.0 + np.exp(np.clip(-gate, -80.0, 80.0))) * up)
            out += bf16(act @ sw2_np[s].T)
        return out
    raise ValueError(f"Unknown shared_kind={shared_kind}")


@contextmanager
def forbid_host_device_sync(torch):
    """Verify zero D2H readback or host synchronization during operator launch."""
    orig_cpu = torch.Tensor.cpu
    orig_item = torch.Tensor.item
    orig_tolist = torch.Tensor.tolist
    orig_sync = torch.npu.synchronize

    def _forbid(*_a, **_kw):
        raise AssertionError("Forbidden host-device readback / synchronize detected during MegaMoE launch!")

    torch.Tensor.cpu = _forbid
    torch.Tensor.item = _forbid
    torch.Tensor.tolist = _forbid
    torch.npu.synchronize = _forbid
    try:
        yield
    finally:
        torch.Tensor.cpu = orig_cpu
        torch.Tensor.item = orig_item
        torch.Tensor.tolist = orig_tolist
        torch.npu.synchronize = orig_sync


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("accuracy", "perf", "trace", "all"), default="all")
    parser.add_argument("--bs", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=1536, help="Global routed intermediate width")
    parser.add_argument("--shared-intermediate", type=int, default=1536)
    parser.add_argument("--num-experts", type=int, default=352)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--m-groups-per-wave", type=int, default=22)
    parser.add_argument("--ag-cores-per-peer", type=int, default=11)
    parser.add_argument("--rs-tokens-per-chunk", type=int, default=64)
    parser.add_argument("--check-tokens", type=int, default=32, help="Tokens per rank checked against CPU NumPy oracle")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--shmem-host", type=str, default="127.0.0.1")
    parser.add_argument("--shmem-port", type=int, default=29620)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401
    from deep_ep_experimental.megamoe.api import (
        StoreGroup,
        SymmBuffer,
        fp8_fp4_mega_moe,
        mega_moe,
        transform_weights_for_mega_moe,
    )
    from deep_ep_experimental.megamoe.golden import compare
    from deep_ep_experimental.megamoe.reference import (
        bf16,
        dequantize_input,
        error_metrics,
        make_weights,
        quantize_input,
    )

    torch.set_num_threads(1)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    group = StoreGroup.from_env()
    rank, world = group.rank(), group.size()

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    group.barrier("mkdir")

    local_inter = args.intermediate // world
    raw_routed = make_weights(args.num_experts, args.hidden, local_inter, args.seed + rank, weight_bits=4)

    def to_npu_pair(pair, *, fp8=False):
        data, scale = pair
        w = torch.from_numpy(data).npu()
        if fp8:
            w = w.view(torch.float8_e4m3fn)
        return (w, torch.from_numpy(scale).npu().view(torch.float8_e8m0fnu))

    prep_routed = transform_weights_for_mega_moe(*map(to_npu_pair, raw_routed), keep_planar=False)

    # Prepare both FP8 shared weights and BF16 shared weights (replicated across ranks: same seed!)
    shared_fp8_raw = make_weights(1, args.hidden, args.shared_intermediate, args.seed + 100000, weight_bits=8)
    prep_shared_fp8 = transform_weights_for_mega_moe(
        *(to_npu_pair(p, fp8=True) for p in shared_fp8_raw), shared=True
    )

    rng_sh = np.random.default_rng(args.seed + 200000)
    sw1_bf16_np = bf16((rng_sh.standard_normal((1, 2 * args.shared_intermediate, args.hidden)) * 0.02).astype(np.float32))
    sw2_bf16_np = bf16((rng_sh.standard_normal((1, args.hidden, args.shared_intermediate)) * 0.02).astype(np.float32))
    prep_shared_bf16 = transform_weights_for_mega_moe(
        torch.from_numpy(sw1_bf16_np).to(device="npu", dtype=torch.bfloat16),
        torch.from_numpy(sw2_bf16_np).to(device="npu", dtype=torch.bfloat16),
        shared=True,
    )

    summary = {"rank": rank, "world": world, "accuracy_cases": [], "perf": None, "trace": None}
    shmem_port_cursor = args.shmem_port

    def create_tp_buffer(max_tokens: int, num_shared: int, tag: str) -> SymmBuffer:
        nonlocal shmem_port_cursor
        group.barrier(f"pre_shmem_{tag}")
        os.environ["DEEPEP_SHMEM_ENDPOINT"] = f"tcp://{args.shmem_host}:{shmem_port_cursor}"
        shmem_port_cursor += 1
        return SymmBuffer(
            group,
            num_experts=args.num_experts,
            num_max_tokens_per_rank=max_tokens,
            num_topk=args.topk,
            hidden=args.hidden,
            intermediate_hidden=args.intermediate,
            shared_intermediate_hidden=args.shared_intermediate,
            num_shared_experts=num_shared,
            parallel_mode="tp",
            m_groups_per_wave=args.m_groups_per_wave,
            ag_cores_per_peer=args.ag_cores_per_peer,
            rs_tokens_per_chunk=args.rs_tokens_per_chunk,
        )

    shared1_buf = None
    try:
        # ==================== 1. Accuracy Matrix + Zero-Sync Enforcement ====================
        if args.mode in ("accuracy", "all"):
            cases = [
                # Smoke BS=96 + full target BS=4096 covering all required combinations
                (96, "bf16", "none", False),
                (96, "fp8", "bf16", False),
                (96, "bf16", "fp8", True),
                (args.bs, "bf16", "bf16", False),
                (args.bs, "fp8", "bf16", False),
                (args.bs, "bf16", "fp8", False),
                (args.bs, "fp8", "fp8", True),
                (args.bs, "bf16", "bf16", True),
            ]
            for num_shared in (0, 1):
                sub_cases = [c for c in cases if (c[2] != "none") == (num_shared == 1)]
                if not sub_cases:
                    continue
                max_tok = max(max(c[0] for c in sub_cases), args.bs if num_shared == 1 else 0)
                buf = create_tp_buffer(max_tok, num_shared, f"acc_{num_shared}")
                keep_for_perf = (num_shared == 1) and (args.mode in ("perf", "trace", "all"))
                try:
                    for bs_case, x_kind, sh_kind, pre_quant in sub_cases:
                        x_np, ids_np, w_np = make_routing(
                            args.seed, rank, bs_case, args.hidden, args.num_experts, args.topk, "random"
                        )
                        x_q_codes, x_q_scales = quantize_input(x_np)
                        x_fp8_deq = dequantize_input(x_q_codes, x_q_scales)

                        if x_kind == "bf16":
                            x_override = torch.from_numpy(x_np).to(device="npu", dtype=torch.bfloat16)
                        else:
                            buf.x[:bs_case].copy_(torch.from_numpy(x_q_codes).npu().view(torch.float8_e4m3fn))
                            buf.x_sf[:bs_case].copy_(torch.from_numpy(x_q_scales).npu().view(torch.float8_e8m0fnu))
                            x_override = None

                        buf.topk_idx[:bs_case].copy_(torch.from_numpy(ids_np))
                        buf.topk_weights[:bs_case].copy_(torch.from_numpy(w_np))
                        buf.prepare(bs_case)

                        sh_kw = {}
                        if sh_kind == "bf16":
                            sh_kw = dict(shared_l1_weights=prep_shared_bf16[0], shared_l2_weights=prep_shared_bf16[1])
                        elif sh_kind == "fp8":
                            sh_kw = dict(shared_l1_weights=prep_shared_fp8[0], shared_l2_weights=prep_shared_fp8[1])

                        y = torch.empty((bs_case, args.hidden), device="npu", dtype=torch.bfloat16)
                        stats = torch.zeros((args.num_experts,), device="npu", dtype=torch.int32)
                        torch.npu.synchronize()
                        group.barrier(f"before_launch_{bs_case}_{x_kind}_{sh_kind}_{pre_quant}")

                        # Strictly forbid any D2H / host sync during forward!
                        with forbid_host_device_sync(torch):
                            fp8_fp4_mega_moe(
                                y,
                                *prep_routed,
                                buf,
                                cumulative_local_expert_recv_stats=stats,
                                x=x_override,
                                route_weight_before_quant=pre_quant,
                                **sh_kw,
                            )

                        torch.npu.synchronize()

                        # Also test explicit mega_moe() wrapper on the 96-token cases
                        if bs_case == 96:
                            x_arg = x_override if x_kind == "bf16" else (buf.x[:bs_case], buf.x_sf[:bs_case])
                            with forbid_host_device_sync(torch):
                                y2, counts2 = mega_moe(
                                    x_arg,
                                    buf.topk_idx[:bs_case],
                                    buf.topk_weights[:bs_case],
                                    *prep_routed,
                                    buf,
                                    route_weight_before_quant=pre_quant,
                                    **sh_kw,
                                )
                            torch.npu.synchronize()
                            assert torch.equal(y, y2) and torch.equal(stats, counts2)

                        # Build exact multi-rank reference on selected tokens + full expert counts
                        # (Use shared output_dir .npz files + StoreGroup barrier to stay well under StoreGroup's 1 MiB control limit)
                        check_n = bs_case if (args.check_tokens <= 0 or bs_case <= args.check_tokens) else args.check_tokens
                        sel = np.linspace(0, bs_case - 1, num=check_n, dtype=np.int64)
                        case_tag = f"{bs_case}_{x_kind}_{sh_kind}_{pre_quant}"

                        in_path = args.output_dir / f"_tmp_in_{case_tag}_r{rank}.npz"
                        np.savez(
                            in_path,
                            x_deq=x_fp8_deq[sel],
                            ids=ids_np[sel],
                            weights=w_np[sel],
                            hist=np.bincount(ids_np.ravel(), minlength=args.num_experts).astype(np.int32),
                        )
                        group.barrier(f"saved_inputs_{case_tag}")

                        all_sel_inputs = [
                            dict(np.load(args.output_dir / f"_tmp_in_{case_tag}_r{r}.npz")) for r in range(world)
                        ]
                        expected_counts = np.sum([item["hist"].astype(np.int32) for item in all_sel_inputs], axis=0)

                        # Compute our rank's TP partial contribution for EVERY target rank's selected tokens
                        my_partials_for_dst = np.stack(
                            [
                                bf16(
                                    tp_rank_routed_partial_ref(
                                        all_sel_inputs[dst_r]["x_deq"].astype(np.float32),
                                        all_sel_inputs[dst_r]["ids"].astype(np.int32),
                                        all_sel_inputs[dst_r]["weights"].astype(np.float32),
                                        raw_routed,
                                        pre_quant,
                                    )
                                )
                                for dst_r in range(world)
                            ],
                            axis=0,
                        )
                        part_path = args.output_dir / f"_tmp_part_{case_tag}_r{rank}.npy"
                        np.save(part_path, my_partials_for_dst)
                        group.barrier(f"saved_partials_{case_tag}")

                        routed_sum = np.sum(
                            [
                                np.load(args.output_dir / f"_tmp_part_{case_tag}_r{src_r}.npy")[rank].astype(np.float32)
                                for src_r in range(world)
                            ],
                            axis=0,
                        )
                        group.barrier(f"loaded_partials_{case_tag}")
                        in_path.unlink(missing_ok=True)
                        part_path.unlink(missing_ok=True)
                        sh_raw_ref = (
                            (sw1_bf16_np, sw2_bf16_np)
                            if sh_kind == "bf16"
                            else (shared_fp8_raw if sh_kind == "fp8" else None)
                        )
                        sh_x_ref = x_np[sel] if x_kind == "bf16" else bf16(x_fp8_deq[sel])
                        sh_out = shared_ref(sh_x_ref, x_fp8_deq[sel], sh_kind, sh_raw_ref)
                        expected_y = bf16(routed_sum + sh_out)

                        actual_y = y.float().cpu().numpy()[sel]
                        actual_counts = stats.cpu().numpy()
                        counts_ok = bool(np.array_equal(actual_counts, expected_counts))
                        cmp_res = compare(actual_y, expected_y)
                        metrics = error_metrics(actual_y, expected_y)
                        passed = bool(cmp_res["passed"] and counts_ok and metrics["finite"])

                        case_rec = dict(
                            bs=bs_case,
                            x=x_kind,
                            shared=sh_kind,
                            route_weight_before_quant=pre_quant,
                            passed=passed,
                            counts_equal=counts_ok,
                            relative_rmse=metrics["relative_rmse"],
                            max_abs=metrics["max_abs"],
                            mismatch_ratio=cmp_res.get("mismatch_ratio"),
                        )
                        all_rec = group.all_gather(case_rec, f"report_{bs_case}_{x_kind}_{sh_kind}_{pre_quant}")
                        if rank == 0:
                            summary["accuracy_cases"].append(all_rec)
                            max_rmse = max(r["relative_rmse"] for r in all_rec)
                            max_abs = max(r["max_abs"] for r in all_rec)
                            ok_all = all(r["passed"] for r in all_rec)
                            print(
                                f"[TP4-ACCURACY] BS={bs_case:<4} x={x_kind:<4} shared={sh_kind:<4} "
                                f"pre_quant={str(pre_quant):<5} -> {'PASS' if ok_all else 'FAIL'} "
                                f"(max_rel_rmse={max_rmse:.5f}, max_abs={max_abs:.5f}, counts_exact={all(r['counts_equal'] for r in all_rec)})",
                                flush=True,
                            )
                        if not all(r["passed"] for r in all_rec):
                            raise AssertionError(f"TP4 accuracy failed: {all_rec}")
                    if keep_for_perf:
                        shared1_buf = buf
                        buf = None
                    else:
                        group.barrier(f"destroy_acc_buf_{num_shared}")
                        buf.destroy()
                        buf = None
                finally:
                    pass

        # ==================== 2. Timer-Off E2E Performance & Timer-On Swimlane Trace ====================
        if args.mode in ("perf", "trace", "all"):
            buf = shared1_buf if shared1_buf is not None else create_tp_buffer(args.bs, 1, "perf_trace")
            shared1_buf = None
            try:
                x_np, ids_np, w_np = make_routing(
                    args.seed + 77, rank, args.bs, args.hidden, args.num_experts, args.topk, "balanced"
                )
                x_bf16 = torch.from_numpy(x_np).to(device="npu", dtype=torch.bfloat16)
                buf.topk_idx[: args.bs].copy_(torch.from_numpy(ids_np))
                buf.topk_weights[: args.bs].copy_(torch.from_numpy(w_np))
                buf.prepare(args.bs)
                y = torch.empty((args.bs, args.hidden), device="npu", dtype=torch.bfloat16)
                sh_kw = dict(shared_l1_weights=prep_shared_bf16[0], shared_l2_weights=prep_shared_bf16[1])

                def run_once(timer_tensor=None):
                    fp8_fp4_mega_moe(y, *prep_routed, buf, x=x_bf16, timer=timer_tensor, **sh_kw)

                for _ in range(args.warmup):
                    run_once(None)
                torch.npu.synchronize()
                group.barrier("perf_warmup_done")

                if args.mode in ("perf", "all"):
                    start_ev = [torch.npu.Event(enable_timing=True) for _ in range(args.iters)]
                    end_ev = [torch.npu.Event(enable_timing=True) for _ in range(args.iters)]
                    t0 = time.perf_counter()
                    for i in range(args.iters):
                        start_ev[i].record()
                        run_once(None)
                        end_ev[i].record()
                    torch.npu.synchronize()
                    wall_us = (time.perf_counter() - t0) * 1e6 / args.iters
                    event_us = [start_ev[i].elapsed_time(end_ev[i]) * 1000.0 for i in range(args.iters)]
                    half = event_us[len(event_us) // 2 :]
                    perf_rec = dict(
                        rank=rank,
                        mean_event_us=float(np.mean(half)),
                        median_event_us=float(np.median(half)),
                        min_event_us=float(np.min(half)),
                        wall_avg_us=float(wall_us),
                        workspace_mib=round(buf.workspace_bytes / (1024**2), 2),
                        symm_mib=round(buf.symmetric_buffer_bytes / (1024**2), 2),
                    )
                    all_perf = group.all_gather(perf_rec, "perf_gather")
                    if rank == 0:
                        summary["perf"] = all_perf
                        rank_max_med = max(p["median_event_us"] for p in all_perf)
                        rank_min_med = min(p["median_event_us"] for p in all_perf)
                        print(
                            f"[TP4-PERF] BS={args.bs} H={args.hidden} I={args.intermediate} E={args.num_experts} "
                            f"topK={args.topk} wave={args.m_groups_per_wave} (timer=OFF): "
                            f"median={rank_min_med:.2f}..{rank_max_med:.2f} us "
                            f"(workspace={perf_rec['workspace_mib']} MiB, symm={perf_rec['symm_mib']} MiB)",
                            flush=True,
                        )

                if args.mode in ("trace", "all") and buf.timer_enabled:
                    y_ref_notimer = y.float().cpu().numpy().copy()
                    timer_buf = buf.allocate_timer()
                    torch.npu.synchronize()
                    group.barrier("before_trace")
                    run_once(timer_buf)
                    torch.npu.synchronize()
                    group.barrier("after_trace")

                    # Verify timer-on output matches timer-off output
                    t_metrics = error_metrics(y.float().cpu().numpy(), y_ref_notimer)
                    assert t_metrics["finite"] and t_metrics["relative_rmse"] < 0.02, f"Timer output drift: {t_metrics}"

                    timer_dir = args.output_dir / "timer"
                    timer_dir.mkdir(parents=True, exist_ok=True)
                    raw_bin = timer_dir / f"rank_{rank}_npu_time.bin"
                    raw_arr = timer_buf.cpu().numpy().astype("<i8", copy=False).reshape(-1)
                    raw_arr.tofile(raw_bin)

                    from deep_ep_experimental.megamoe.timer.breakdown import (
                        compact_events,
                        rank_breakdown,
                        write_breakdown,
                    )
                    from deep_ep_experimental.megamoe.timer.parse_timer_csv import decode
                    from deep_ep_experimental.megamoe.timer.trace_parser import (
                        apply_wait_mode,
                        raw_events_for_rank,
                        synchronized_rank_bases,
                        trace_events,
                        write_summary,
                    )

                    parsed_csv = timer_dir / f"rank_{rank}_npu_time_parsed.csv"
                    decode(raw_bin, parsed_csv)
                    bd = rank_breakdown(compact_events(raw_arr, rank), rank)
                    all_bd = group.all_gather(bd, "gather_breakdown")
                    group.barrier("parsed_csv_ready")

                    if rank == 0:
                        write_breakdown(all_bd, timer_dir, f"TP4_BS={args.bs}_wave={args.m_groups_per_wave}")
                        raw_by_rank = {r: raw_events_for_rank(timer_dir / f"rank_{r}_npu_time_parsed.csv", r) for r in range(world)}
                        sync_bases, _ = synchronized_rank_bases(raw_by_rank)
                        trace_dir = args.output_dir / "timer_json"
                        trace_dir.mkdir(parents=True, exist_ok=True)
                        merged_trace = []
                        merged_raw = []
                        for r, raw_ev in raw_by_rank.items():
                            # Default include_wait=False: soft-sync waits appear as clean blank bubbles!
                            tr = trace_events(raw_ev, "keep", sync_bases[r], include_wait=False, time_origin="sync")
                            (trace_dir / f"rank_{r}_npu_time.json").write_text(
                                json.dumps(tr, indent=2, ensure_ascii=False), encoding="utf-8"
                            )
                            merged_trace.extend(tr)
                            merged_raw.extend(apply_wait_mode(raw_ev, False))
                        all_json = trace_dir / "all_rank_trace.json"
                        all_json.write_text(json.dumps(merged_trace, indent=2, ensure_ascii=False), encoding="utf-8")
                        write_summary(merged_raw, trace_dir / "timer_summary.csv")
                        families_present = sorted({ev["args"].get("event_family", ev["name"]) for ev in merged_trace})
                        wait_in_trace = [ev["name"] for ev in merged_trace if "WAIT" in ev["name"]]
                        assert not wait_in_trace, f"WAIT bars leaked into default swimlane trace: {wait_in_trace[:5]}"
                        summary["trace"] = dict(
                            trace_json=str(all_json),
                            total_events=len(merged_trace),
                            families=families_present,
                        )
                        print(f"[TP4-TRACE] Swimlane JSON ({len(merged_trace)} events, WAIT bars=0) -> {all_json}", flush=True)
                        print(f"[TP4-TRACE] Event families recorded: {families_present}", flush=True)
                group.barrier("destroy_perf_buf")
                buf.destroy()
                buf = None
            finally:
                pass

        if rank == 0:
            (args.output_dir / "tp4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        group.barrier("done")
        group.close()
    except BaseException as exc:
        group.abort(f"bench_tp4 failed: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
