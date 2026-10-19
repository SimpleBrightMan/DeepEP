# Copyright (c) 2026, SimpleBrightMan

"""Pure host-side planning helpers for the MTE All-to-All benchmark."""

from __future__ import annotations

from typing import Iterable, Sequence

MODES = ("local", "intra", "inter", "full")
MODE_LABELS = {
    "local": "本卡内",
    "intra": "组内下行",
    "inter": "组间下行",
    "full": "全量",
}
MAX_TOKENS = 16384
MAX_TOKEN_BYTES = 64 * 1024
MAX_WORLD = 128


def parse_token_counts(value: str | int | Iterable[int]) -> list[int]:
    """Parse ``1024,2048,4096`` into an ordered list of positive token counts."""
    if isinstance(value, int):
        raw: Sequence[str] = [str(value)]
    elif isinstance(value, str):
        raw = value.split(",")
    else:
        raw = [str(item) for item in value]
    counts: list[int] = []
    for item in raw:
        text = item.strip()
        if not text:
            raise ValueError("token counts must not contain empty entries")
        if not text.isdecimal():
            raise ValueError(f"token count must be a positive integer: {text!r}")
        count = int(text)
        if count < 1 or count > MAX_TOKENS:
            raise ValueError(f"token count must be in [1, {MAX_TOKENS}]: {count}")
        if count not in counts:
            counts.append(count)
    if not counts:
        raise ValueError("at least one token count is required")
    return counts


def validate_shape(world: int, group_size: int, token_bytes: int) -> None:
    if not 2 <= world <= MAX_WORLD:
        raise ValueError(f"world size must be in [2, {MAX_WORLD}]: {world}")
    if group_size < 1 or world % group_size:
        raise ValueError(f"group size must divide the world size: {group_size} vs {world}")
    if token_bytes < 32 or token_bytes > MAX_TOKEN_BYTES or token_bytes % 32:
        raise ValueError(f"token bytes must be a 32-byte multiple in [32, {MAX_TOKEN_BYTES}]: {token_bytes}")


def peers_for(rank: int, world: int, group_size: int, mode: str) -> list[int]:
    """Return the destination ranks selected by one measurement scope."""
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode!r}")
    if not 0 <= rank < world:
        raise ValueError(f"rank must be in [0, {world}): {rank}")
    if mode == "local":
        return [rank]
    own_group = rank // group_size
    if mode == "intra":
        return [peer for peer in range(world) if peer // group_size == own_group and peer != rank]
    if mode == "inter":
        return [peer for peer in range(world) if peer // group_size != own_group]
    return list(range(world))


def payload_bytes(tokens: int, token_bytes: int, peers: int) -> int:
    return int(tokens) * int(token_bytes) * int(peers)


def workspace_bytes(world: int, tokens: int, token_bytes: int) -> int:
    return int(world) * int(tokens) * int(token_bytes)


def bandwidth_gbps(payload: int, milliseconds: float) -> float:
    """Decimal GB/s from payload bytes and a millisecond duration."""
    if milliseconds <= 0:
        return float("nan")
    return float(payload) / (float(milliseconds) / 1000.0) / 1e9


def format_gbps(value: float) -> str:
    if value != value:  # NaN
        return "n/a"
    return f"{value:.2f}"


def build_table(cases: Sequence[dict]) -> str:
    """Render one Markdown table from aggregated case dictionaries."""
    by_tokens: dict[int, dict[str, dict]] = {}
    for case in cases:
        by_tokens.setdefault(int(case["tokens"]), {})[str(case["mode"])] = case
    header = "| tokens/pair | 本卡内 (GB/s) | 组内下行 (GB/s) | 组间下行 (GB/s) | 全量 (GB/s) |"
    divider = "| ---: | ---: | ---: | ---: | ---: |"
    lines = [header, divider]
    for tokens in sorted(by_tokens):
        row = by_tokens[tokens]
        cells = [str(tokens)]
        for mode in MODES:
            case = row.get(mode)
            cells.append(format_gbps(float(case["bandwidth_gbps"])) if case else "n/a")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_per_link_table(cases: Sequence[dict]) -> str:
    """Render per-destination bandwidth (aggregate divided by selected peers)."""
    by_tokens: dict[int, dict[str, dict]] = {}
    for case in cases:
        by_tokens.setdefault(int(case["tokens"]), {})[str(case["mode"])] = case
    header = "| tokens/pair | 组内/目的地 (GB/s) | 组间/目的地 (GB/s) |"
    divider = "| ---: | ---: | ---: |"
    lines = [header, divider]
    for tokens in sorted(by_tokens):
        row = by_tokens[tokens]
        cells = [str(tokens)]
        for mode in ("intra", "inter"):
            case = row.get(mode)
            if not case or not case.get("peers"):
                cells.append("n/a")
            else:
                cells.append(format_gbps(float(case["bandwidth_gbps"]) / int(case["peers"])))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
