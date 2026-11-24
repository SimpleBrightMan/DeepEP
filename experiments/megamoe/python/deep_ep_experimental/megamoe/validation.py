# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Explicit host-side preflight, never a hidden scan in the kernel hot path."""

import math
from deep_ep._store import StoreGroup


def check_routing(indices, weights, num_experts, num_topk, *, allow_invalid=False):
    """Validate CPU lists obtained from the actual input tensors."""
    if len(indices) != len(weights):
        raise ValueError("Routing IDs and weights have different BS")
    for token, (ids, probs) in enumerate(zip(indices, weights)):
        if len(ids) != num_topk or len(probs) != num_topk:
            raise ValueError(f"token={token}: routing width must equal top-k={num_topk}")
        if any(type(expert) is not int or not (-1 if allow_invalid else 0) <= expert < num_experts for expert in ids):
            raise ValueError(f"token={token}: expert IDs must be in [0,{num_experts}); -1 masking is unsupported")
        valid=[expert for expert in ids if expert>=0]
        if len(set(valid)) != len(valid):
            raise ValueError(f"token={token}: duplicate expert IDs are unsupported")
        if any(not math.isfinite(value) for value in probs):
            raise ValueError(f"token={token}: route weights must be finite")
        if allow_invalid and any(expert<0 and value!=0 for expert,value in zip(ids,probs)):
            raise ValueError(f"token={token}: invalid routes require zero weight")


def collective_check(group, stage, error, signature=None):
    if isinstance(group, StoreGroup):
        peers = group.all_gather((error, signature), f"check:{stage}")
    else:
        import torch.distributed as dist

        peers = [None] * dist.get_world_size(group)
        dist.all_gather_object(peers, (error, signature), group=group)
    failures = [f"rank={rank}: {value[0]}" for rank, value in enumerate(peers) if value[0]]
    if failures:
        raise ValueError(f"MegaMoE {stage} failed: " + "; ".join(failures))
    if any(value[1] != peers[0][1] for value in peers):
        raise ValueError(
            f"MegaMoE {stage}: ranks disagree: "
            + "; ".join(f"rank={rank}: {value[1]}" for rank, value in enumerate(peers))
        )
