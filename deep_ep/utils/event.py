# API signatures adapted from DeepSeek DeepEP; see third-party/deepseek-deepep.LICENSE.
"""Public EP interface declarations.

These declarations do not provide an Ascend implementation. Every operation
raises NotImplementedError before inspecting tensors or acquiring resources.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

from .._native import _unavailable

if TYPE_CHECKING:
    import torch


class EventHandle:
    """Native completion declaration; cannot record or wait for an event yet."""

    def __init__(self) -> None:
        _unavailable("EventHandle.__init__")

    def current_stream_wait(self) -> None:
        _unavailable("EventHandle.current_stream_wait")


class EventOverlap:
    """EventOverlap API declaration. No usable runtime object can be constructed yet."""

    def __init__(
        self,
        event: Optional[EventHandle] = None,
        extra_tensors: Optional[Tuple[torch.Tensor]] = None,
    ) -> None:
        """Declare EventOverlap.__init__; execution is not implemented."""
        _unavailable("EventOverlap.__init__")

    def current_stream_wait(self, release_handle: bool = False) -> None:
        """Declare EventOverlap.current_stream_wait; execution is not implemented."""
        _unavailable("EventOverlap.current_stream_wait")

    def register_hook_after_wait(self, hook_after_wait: Callable) -> None:
        """Declare EventOverlap.register_hook_after_wait; execution is not implemented."""
        _unavailable("EventOverlap.register_hook_after_wait")

    def __call__(self, release_handle: bool = False) -> "EventOverlap":
        """Declare EventOverlap.__call__; execution is not implemented."""
        _unavailable("EventOverlap.__call__")

    def __enter__(self) -> Any:
        """Declare EventOverlap.__enter__; execution is not implemented."""
        _unavailable("EventOverlap.__enter__")

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Declare EventOverlap.__exit__; execution is not implemented."""
        _unavailable("EventOverlap.__exit__")
