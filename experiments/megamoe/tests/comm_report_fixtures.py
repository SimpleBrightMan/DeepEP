"""Synthetic device captures for report validation; no NPU/extension needed."""

from collections import defaultdict

import numpy as np

from deep_ep_experimental.megamoe import wave_oracle
from deep_ep_experimental.megamoe.timer import protocol as p


class SparseNumpyTimer:
    def __init__(self, audit=False):
        self.values = {}
        self.audit = audit

    def __len__(self):
        return p.COMM_AUDIT_BUFFER_SIZE if self.audit else p.TOTAL_BUFFER_SIZE

    def __getitem__(self, index):
        if isinstance(index, slice):
            return np.array([self.values.get(i, 0) for i in range(index.start, index.stop)], dtype=np.int64)
        return np.int64(self.values.get(index, 0))


def capture(bs=16, rank=0, dispatch_intra=0, combine_intra=0, hidden=3584, intermediate=1280, experts=384, topk=8):
    spec = dict(
        bs=bs,
        hidden=hidden,
        intermediate=intermediate,
        experts=experts,
        topk=topk,
        rank=rank,
        dispatch_intra=dispatch_intra,
        combine_intra=combine_intra,
    )
    plan = wave_oracle.plan(**spec)
    measured, audit = SparseNumpyTimer(), SparseNumpyTimer(audit=True)
    audit.values[p.COMM_AUDIT_ENABLE_OFFSET] = 1
    for raw in (measured, audit):
        raw.values[p.COMM_M_GROUPS_PER_WAVE_OFFSET] = plan["m_groups_per_wave"]
    grouped = defaultdict(list)
    for transfer in plan["transfers"]:
        grouped[(transfer.operation, transfer.wave, transfer.core, transfer.expert)].append(transfer)
        hardware_core = transfer.core * 3 + 2
        count_offset = p.COMM_AUDIT_COUNT_OFFSET + hardware_core * p.COMM_AUDIT_COUNT_STRIDE
        index = audit.values.get(count_offset, 0)
        offset = p.COMM_AUDIT_RECORD_OFFSET + (hardware_core * p.COMM_AUDIT_MAX_RECORDS_PER_CORE + index) * 3
        audit.values[offset] = transfer.wave
        audit.values[offset + 1] = ((transfer.operation == "COMBINE") << 56) | (transfer.expert << 32) | transfer.row
        audit.values[offset + 2] = (transfer.source_rank << 48) | (transfer.topk_slot << 32) | transfer.source_token
        audit.values[count_offset] = index + 1
    slices = {(s["wave"], s["expert"]): s for s in plan["slices"]}
    for sequence, ((operation, wave, core, expert), transfers) in enumerate(sorted(grouped.items())):
        hardware_core = core * 3 + 2
        family = p.DYNAMIC_NAMES.index(f"AIV_{operation}_COMM")
        count_offset = p.N_TIMING_COUNTER + hardware_core * p.DYNAMIC_ITER_PER_CORE + family
        index = measured.values.get(count_offset, 0)
        logical = len(p.FIXED_NAMES) + family * p.MAX_DYNAMIC_ITER + index
        time_offset = hardware_core * p.COUNTERS_PER_CORE + logical * 2
        measured.values[time_offset] = sequence * 100000 + 1000
        measured.values[time_offset + 1] = sequence * 100000 + 1000 + (core + 1) * 1000
        split = dispatch_intra if operation == "DISPATCH" else combine_intra
        first_row = slices[(wave, expert)]["row_start"] if split else min(t.row for t in transfers)
        metadata = (
            p.TASK_METADATA_OFFSET
            + (hardware_core * len(p.DYNAMIC_NAMES) * p.MAX_DYNAMIC_ITER + logical - len(p.FIXED_NAMES)) * 2
        )
        measured.values[metadata] = ((expert + 1) << 32) | first_row
        measured.values[metadata + 1] = len(transfers)
        measured.values[p.WAVE_ID_OFFSET + hardware_core * p.WAVE_ID_PER_CORE + logical] = wave
        measured.values[count_offset] = index + 1
    return spec, measured, audit, plan
