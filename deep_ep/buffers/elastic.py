# API signatures adapted from DeepSeek DeepEP; see third-party/deepseek-deepep.LICENSE.
"""V2 API with native resource primitives; EP algorithms remain unavailable."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

from .._native import _unavailable, require_native
from .._runtime import Runtime, runtime_of, warn_unclosed

if TYPE_CHECKING:
    import torch
    import torch.distributed as dist
    from ..utils.event import EventHandle, EventOverlap


class EPHandle:
    """EPHandle API declaration. No usable runtime object can be constructed yet."""

    def __init__(
        self,
        do_expand: bool,
        num_experts: int,
        expert_alignment: int,
        num_max_tokens_per_rank: int,
        num_compute_units: int,
        topk_idx: torch.Tensor,
        num_recv_tokens: int,
        num_expanded_tokens: int,
        num_recv_tokens_per_expert_list: list,
        psum_num_recv_tokens_per_scaleup_rank: torch.Tensor,
        psum_num_recv_tokens_per_expert: torch.Tensor,
        num_unaligned_recv_tokens_per_expert: torch.Tensor,
        recv_src_metadata: torch.Tensor,
        dst_buffer_slot_idx: torch.Tensor,
        token_metadata_at_forward: Optional[torch.Tensor],
        channel_linked_list: Optional[torch.Tensor],
    ):
        """Declare EPHandle.__init__; execution is not implemented."""
        _unavailable("EPHandle.__init__")


class ElasticBuffer:
    """Own a restricted native resource session; see docs/runtime-design.md."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_bytes: Optional[int] = None,
        num_cpu_bytes: int = 0,
        num_max_tokens_per_rank: int = 0,
        hidden: int = 0,
        num_topk: int = 0,
        use_fp8_dispatch: bool = False,
        deterministic: bool = False,
        allow_hybrid_mode: bool = True,
        allow_multiple_reduction: bool = True,
        prefer_overlap_with_compute: bool = True,
        sl_idx: int = 3,
        num_allocated_qps: int = 0,
        num_cpu_timeout_secs: int = 300,
        num_device_timeout_secs: int = 100,
        explicitly_destroy: bool = False,
    ):
        """Create explicit-size MTE resources, not an EP algorithm backend.

        num_allocated_qps is reserved; MTE does not consume a QP budget.
        A nondefault value is rejected, not mapped to DMA queue/channel counts.
        """
        require_native("ElasticBuffer.__init__")
        if (
            num_cpu_bytes != 0
            or num_max_tokens_per_rank != 0
            or hidden != 0
            or num_topk != 0
            or use_fp8_dispatch is not False
            or deterministic is not False
            or allow_hybrid_mode is not True
            or allow_multiple_reduction is not True
            or prefer_overlap_with_compute is not True
            or sl_idx != 3
            or num_allocated_qps != 0
            or num_device_timeout_secs != 100
        ):
            raise NotImplementedError("ElasticBuffer supports explicit bytes and default algorithm options only")
        if explicitly_destroy is not True:
            raise NotImplementedError("Runtime currently requires explicitly_destroy=True")
        if num_bytes is None:
            raise NotImplementedError("Pass num_bytes explicitly; algorithm sizing is not implemented")
        self._runtime = Runtime(group, num_bytes, num_cpu_timeout_secs)

    def __del__(self):
        warn_unclosed(self)

    def destroy(self) -> None:
        """Collectively release owned resources; successful repetition is safe."""
        require_native("ElasticBuffer.destroy")
        runtime_of(self).close()

    @staticmethod
    def get_buffer_size_hint(
        group: dist.ProcessGroup,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_topk: int = 0,
        use_fp8_dispatch: bool = False,
        allow_hybrid_mode: bool = True,
        allow_multiple_reduction: bool = True,
    ) -> int:
        """Declare ElasticBuffer.get_buffer_size_hint; execution is not implemented."""
        _unavailable("ElasticBuffer.get_buffer_size_hint")

    def barrier(
        self,
        use_comm_stream: bool = True,
        with_cpu_sync: bool = False,
        sequential: bool = True,
    ) -> None:
        """Run the explicitly selected host-synchronous sequential barrier."""
        require_native("ElasticBuffer.barrier")
        if type(use_comm_stream) is not bool or with_cpu_sync is not True or sequential is not True:
            raise NotImplementedError("Only the host-synchronous sequential barrier is implemented")
        runtime_of(self).barrier()

    @staticmethod
    def capture() -> EventHandle:
        """Capture a real event on the current framework stream."""
        require_native("ElasticBuffer.capture")
        from ..utils.event import EventHandle

        return EventHandle()

    def get_comm_stream(self) -> torch.Stream:
        """Return the live framework communication stream."""
        require_native("ElasticBuffer.get_comm_stream")
        runtime = runtime_of(self)
        runtime.ensure_alive()
        return runtime.stream

    def get_physical_domain_size(self) -> Tuple[int, int]:
        """Declare ElasticBuffer.get_physical_domain_size; execution is not implemented."""
        _unavailable("ElasticBuffer.get_physical_domain_size")

    def get_logical_domain_size(self) -> Tuple[int, int]:
        """Declare ElasticBuffer.get_logical_domain_size; execution is not implemented."""
        _unavailable("ElasticBuffer.get_logical_domain_size")

    def get_theoretical_num_compute_units(
        self,
        num_experts: int,
        num_topk: int,
        num_scaleout_topk: int = 0,
        scaleout_gbs: float = 0,
        scaleup_gbs: float = 0,
        compute_read_gbs: float = 200,
        compute_write_gbs: float = 50,
    ) -> int:
        """Declare ElasticBuffer.get_theoretical_num_compute_units; execution is not implemented."""
        _unavailable("ElasticBuffer.get_theoretical_num_compute_units")

    def get_theoretical_num_qps(self, num_compute_units: int) -> int:
        """Declare a reserved QP sizing query, not a generic DMA resource count."""
        _unavailable("ElasticBuffer.get_theoretical_num_qps")

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        num_max_tokens_per_rank: Optional[int] = None,
        expert_alignment: Optional[int] = None,
        num_compute_units: int = 0,
        num_qps: int = 0,
        previous_event: Optional[EventHandle] = None,
        previous_event_before_epilogue: Optional[EventHandle] = None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
        handle: Optional[EPHandle] = None,
        do_handle_copy: bool = True,
        do_cpu_sync: Optional[bool] = None,
        do_expand: Optional[bool] = None,
        do_zero_padding: bool = False,
        use_aligned_col_major_sf: bool = False,
    ) -> Tuple[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        EPHandle,
        EventOverlap,
    ]:
        """Declare ElasticBuffer.dispatch; execution is not implemented.

        num_qps is reserved for a QP-capable backend, not a resource budget
        shared by all transport engines; see docs/api-contracts.md.
        """
        _unavailable("ElasticBuffer.dispatch")

    def combine(
        self,
        x: torch.Tensor,
        handle: EPHandle,
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        num_compute_units: int = 0,
        num_qps: int = 0,
        previous_event: EventHandle = None,
        previous_event_before_epilogue: Optional[EventHandle] = None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """Declare ElasticBuffer.combine; execution is not implemented.

        num_qps has the same reserved, backend-specific meaning as in dispatch.
        """
        _unavailable("ElasticBuffer.combine")
