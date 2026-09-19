"""Legacy Config declarations; no Ascend sizing policy is implemented."""

from ._native import _unavailable


class Config:
    """Names and defaults follow the pinned upstream binding, not Ascend tuning."""

    def __init__(
        self,
        num_sms: int = 20,
        num_max_nvl_chunked_send_tokens: int = 6,
        num_max_nvl_chunked_recv_tokens: int = 256,
        num_max_rdma_chunked_send_tokens: int = 6,
        num_max_rdma_chunked_recv_tokens: int = 256,
    ) -> None:
        _unavailable("Config.__init__")

    def get_nvl_buffer_size_hint(self, hidden_bytes: int, num_ranks: int, /) -> int:
        _unavailable("Config.get_nvl_buffer_size_hint")

    def get_rdma_buffer_size_hint(self, hidden_bytes: int, num_ranks: int, /) -> int:
        _unavailable("Config.get_rdma_buffer_size_hint")
