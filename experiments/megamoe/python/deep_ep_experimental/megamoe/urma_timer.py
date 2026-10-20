"""Decode URMA schema 1; aggregate same-rank elapsed envelopes, never core averages."""

import math
import statistics

from .urma_routing import remote_peer_for_worker

PHASES = ("Pack", "Dispatch", "Gmm1", "Activation", "Gmm2", "Combine", "Barrier", "Reduce")
TIMER_NUMEL = 28 * 3 * 8 * 16
TIMER_SCHEMA = 1


def communication_bandwidth(raw, rank, *, clock_mhz=1000.0):
    """Effective aggregate payload divided by the class's first-to-last span.

    1000 ticks/us follows the existing MegaMoE timer protocol. It is an explicit
    assumption, not a device-frequency measurement. Pass None for raw ticks.
    Older records may include self-copy in INTRA. Detect it from recorded
    payload; direct self Pack/GMM2 writes are not communication payload.
    """
    import numpy as np

    if not 0 <= rank < 16:
        raise ValueError("rank must be in [0,16)")
    if clock_mhz is not None and (not math.isfinite(clock_mhz) or clock_mhz <= 0):
        raise ValueError("clock_mhz must be positive and finite, or None")
    values = np.asarray(raw)
    if values.size != TIMER_NUMEL or values.dtype.kind not in "iu":
        raise ValueError(f"URMA timer requires {TIMER_NUMEL} integer values")
    values = values.reshape(28, 3, 8, 16)
    active = values[..., 4] > 0
    if np.any(values[..., 6][active] != TIMER_SCHEMA):
        raise ValueError("URMA timer schema is missing or incompatible")
    if np.any(values[..., 1][active] < values[..., 0][active]):
        raise ValueError("URMA timer end precedes start")
    result = dict(
        rank=rank,
        schema=TIMER_SCHEMA,
        clock_mhz=clock_mhz,
        clock_source="caller supplied; default 1000 follows legacy MegaMoE protocol",
        metric="single_direction_aggregate_effective_payload_gbps",
        denominator="same-rank class max(last_tick) - min(first_tick)",
        includes_self_copy_in_intra=False,
        classes=[],
    )
    for phase in ("Dispatch", "Combine"):
        phase_idx = PHASES.index(phase)
        for kind in ("INTRA", "CROSS"):
            selected = []
            for owner in range(16):
                peer = remote_peer_for_worker(rank, owner) if owner < 15 else rank
                owner_kind = "INTRA" if peer // 4 == rank // 4 else "CROSS"
                record = values[owner, 2, phase_idx]
                if owner_kind == kind and record[3] > 0:
                    if record[4] <= 0 or record[6] != TIMER_SCHEMA:
                        raise ValueError("payload without a completed timer record")
                    selected.append((owner, peer, record))
            payload = sum(int(record[3]) for _, _, record in selected)
            includes_self_copy = any(peer == rank for _, peer, _ in selected)
            result["includes_self_copy_in_intra"] |= includes_self_copy
            if selected:
                first = min(int(record[0]) for _, _, record in selected)
                last = max(int(record[1]) for _, _, record in selected)
                span = last - first
                if span <= 0:
                    raise ValueError("nonempty payload requires a positive timer span")
            else:
                first = last = span = 0
            result["classes"].append(
                dict(
                    phase=phase,
                    communication_class=kind,
                    payload_bytes=payload,
                    first_tick=first,
                    last_tick=last,
                    envelope_ticks=span,
                    envelope_us=span / clock_mhz if clock_mhz is not None else None,
                    bandwidth_gbps=payload * clock_mhz / (span * 1000) if span and clock_mhz is not None else None,
                    workers=[owner for owner, _, _ in selected],
                    peers=[peer for _, peer, _ in selected],
                    includes_self_copy=includes_self_copy,
                )
            )
    return result


def summarize_bandwidth(records):
    """Average already-local bandwidths, without mixing clocks across ranks."""
    if len(records) != 16 or {row["rank"] for row in records} != set(range(16)):
        raise ValueError("bandwidth summary requires exactly one report for every EP16 rank")
    result = []
    for phase in ("Dispatch", "Combine"):
        for kind in ("INTRA", "CROSS"):
            rows = [
                next(row for row in record["classes"] if (row["phase"], row["communication_class"]) == (phase, kind))
                for record in records
            ]
            rates = [row["bandwidth_gbps"] for row in rows if row["bandwidth_gbps"] is not None]
            result.append(
                dict(
                    phase=phase,
                    communication_class=kind,
                    mean_gbps=statistics.mean(rates) if rates else None,
                    rank_min_gbps=min(rates) if rates else None,
                    rank_max_gbps=max(rates) if rates else None,
                    nonempty_ranks=len(rates),
                    bytes_per_rank=[row["payload_bytes"] for row in rows],
                    includes_self_copy=any(row.get("includes_self_copy", False) for row in rows),
                )
            )
    return result
