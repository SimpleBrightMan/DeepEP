# API signatures adapted from DeepSeek DeepEP; see third-party/deepseek-deepep.LICENSE.
"""Public EP interface declarations.

These declarations do not provide an Ascend implementation. Every operation
raises NotImplementedError before inspecting tensors or acquiring resources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

from .._native import _unavailable

if TYPE_CHECKING:
    import mpi4py.MPI
    import torch
    import torch.distributed as dist
    from .._config import Config
    from ..utils.event import EventOverlap


class Buffer:
    """Buffer API declaration. No usable runtime object can be constructed yet."""

    def __init__(
        self,
        group: Optional[dist.ProcessGroup],
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        low_latency_mode: bool = False,
        num_qps_per_rank: int = 24,
        allow_nvlink_for_low_latency_mode: bool = True,
        allow_mnnvl: bool = False,
        explicitly_destroy: bool = False,
        enable_shrink: bool = False,
        comm: Optional["mpi4py.MPI.Comm"] = None,
    ) -> None:
        """Declare Buffer.__init__; execution is not implemented."""
        _unavailable("Buffer.__init__")

    def destroy(self):
        """Declare Buffer.destroy; execution is not implemented."""
        _unavailable("Buffer.destroy")

    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        """Declare Buffer.set_num_sms; execution is not implemented."""
        _unavailable("Buffer.set_num_sms")

    @staticmethod
    def capture() -> EventOverlap:
        """Declare Buffer.capture; execution is not implemented."""
        _unavailable("Buffer.capture")

    @staticmethod
    def get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank: int, hidden: int, num_ranks: int, num_experts: int
    ) -> int:
        """Declare Buffer.get_low_latency_rdma_size_hint; execution is not implemented."""
        _unavailable("Buffer.get_low_latency_rdma_size_hint")

    def get_comm_stream(self) -> torch.Stream:
        """Declare Buffer.get_comm_stream; execution is not implemented."""
        _unavailable("Buffer.get_comm_stream")

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        """Declare Buffer.get_dispatch_config; execution is not implemented."""
        _unavailable("Buffer.get_dispatch_config")

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        """Declare Buffer.get_combine_config; execution is not implemented."""
        _unavailable("Buffer.get_combine_config")

    def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        num_experts: int,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, EventOverlap]:
        """Declare Buffer.get_dispatch_layout; execution is not implemented."""
        _unavailable("Buffer.get_dispatch_layout")

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[Tuple] = None,
        num_tokens_per_rank: Optional[torch.Tensor] = None,
        num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
        is_token_in_rank: Optional[torch.Tensor] = None,
        num_tokens_per_expert: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        expert_alignment: int = 1,
        num_worst_tokens: int = 0,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        Tuple,
        EventOverlap,
    ]:
        """Declare Buffer.dispatch; execution is not implemented."""
        _unavailable("Buffer.dispatch")

    def combine(
        self,
        x: torch.Tensor,
        handle: Tuple,
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """Declare Buffer.combine; execution is not implemented."""
        _unavailable("Buffer.combine")

    def clean_low_latency_buffer(self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int) -> None:
        """Declare Buffer.clean_low_latency_buffer; execution is not implemented."""
        _unavailable("Buffer.clean_low_latency_buffer")

    def low_latency_dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,
        use_fp8: bool = True,
        round_scale: bool = False,
        use_ue8m0: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable]:
        """Declare Buffer.low_latency_dispatch; execution is not implemented."""
        _unavailable("Buffer.low_latency_dispatch")

    def low_latency_combine(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        use_logfmt: bool = False,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        out: Optional[torch.Tensor] = None,
        combine_wait_recv_cost_stats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, EventOverlap, Callable]:
        """Declare Buffer.low_latency_combine; execution is not implemented."""
        _unavailable("Buffer.low_latency_combine")

    def get_next_low_latency_combine_buffer(self, handle: object):
        """Declare Buffer.get_next_low_latency_combine_buffer; execution is not implemented."""
        _unavailable("Buffer.get_next_low_latency_combine_buffer")
