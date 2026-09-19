"""Failure boundary for interfaces whose Ascend backend is not implemented.

This module deliberately does not load Torch, torch_npu, a native extension,
or a communication runtime. Native loading belongs to a later implementation.
"""

from typing import NoReturn


def _unavailable(operation: str) -> NoReturn:
    raise NotImplementedError(
        f"{operation} is not implemented in DeepEP for Ascend. "
        "This checkout provides API declarations only; no communication, "
        "memory allocation, or completion event has been performed. "
        "See docs/api.md for implementation status."
    )
