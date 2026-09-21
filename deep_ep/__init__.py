"""DeepEP resource/event interfaces; imports do not require Torch or an NPU SDK.

Installed native wheels provide resource/event primitives, not EP algorithms.
Without the extension, runtime calls fail explicitly. See docs/runtime-design.md.
"""

from ._config import Config
from .buffers import Buffer, ElasticBuffer, EPHandle
from .utils import EventHandle, EventOverlap

__all__ = ["Buffer", "ElasticBuffer", "EPHandle", "Config", "EventHandle", "EventOverlap"]
