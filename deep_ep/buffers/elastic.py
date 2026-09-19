# API signatures adapted from DeepSeek DeepEP; see third-party/deepseek-deepep.LICENSE.
"""Public EP interface declarations.

These declarations do not provide an Ascend implementation. Every operation
raises NotImplementedError before inspecting tensors or acquiring resources.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

from .._native import _unavailable

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
        num_sms: int,
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
    """ElasticBuffer API declaration. No usable runtime object can be constructed yet."""

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
        num_gpu_timeout_secs: int = 100,
        explicitly_destroy: bool = False,
    ):
        """Declare ElasticBuffer.__init__; execution is not implemented."""
        _unavailable("ElasticBuffer.__init__")

    def destroy(self) -> None:
        """Declare ElasticBuffer.destroy; execution is not implemented."""
        _unavailable("ElasticBuffer.destroy")

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
        self, use_comm_stream: bool = True, with_cpu_sync: bool = False, sequential: bool = True
    ) -> None:
        """Declare ElasticBuffer.barrier; execution is not implemented."""
        _unavailable("ElasticBuffer.barrier")

    @staticmethod
    def capture() -> EventHandle:
        """Declare ElasticBuffer.capture; execution is not implemented."""
        _unavailable("ElasticBuffer.capture")

    def get_comm_stream(self) -> torch.Stream:
        """Declare ElasticBuffer.get_comm_stream; execution is not implemented."""
        _unavailable("ElasticBuffer.get_comm_stream")

    def get_physical_domain_size(self) -> Tuple[int, int]:
        """Declare ElasticBuffer.get_physical_domain_size; execution is not implemented."""
        _unavailable("ElasticBuffer.get_physical_domain_size")

    def get_logical_domain_size(self) -> Tuple[int, int]:
        """Declare ElasticBuffer.get_logical_domain_size; execution is not implemented."""
        _unavailable("ElasticBuffer.get_logical_domain_size")

    def get_theoretical_num_sms(
        self,
        num_experts: int,
        num_topk: int,
        num_scaleout_topk: int = 0,
        rdma_gbs: float = 0,
        nvlink_gbs: float = 0,
        sm_read_gbs: float = 200,
        sm_write_gbs: float = 50,
    ) -> int:
        """Declare ElasticBuffer.get_theoretical_num_sms; execution is not implemented."""
        _unavailable("ElasticBuffer.get_theoretical_num_sms")

    def get_theoretical_num_qps(self, num_sms: int) -> int:
        """Declare ElasticBuffer.get_theoretical_num_qps; execution is not implemented."""
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
        num_sms: int = 0,
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
        use_tma_aligned_col_major_sf: bool = False,
    ) -> Tuple[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        EPHandle,
        EventOverlap,
    ]:
        """Declare ElasticBuffer.dispatch; execution is not implemented."""
        _unavailable("ElasticBuffer.dispatch")

    def combine(
        self,
        x: torch.Tensor,
        handle: EPHandle,
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        num_sms: int = 0,
        num_qps: int = 0,
        previous_event: EventHandle = None,
        previous_event_before_epilogue: Optional[EventHandle] = None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """Declare ElasticBuffer.combine; execution is not implemented."""
        _unavailable("ElasticBuffer.combine")
