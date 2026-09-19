"""DeepEP for Ascend API declarations, without a communication backend.

Imports work without Torch or an NPU installation. Creating any public runtime
object or invoking an operation raises NotImplementedError. Importability is
not evidence of device support; see docs/api.md.
"""

from ._config import Config
from .buffers import Buffer, ElasticBuffer, EPHandle
from .utils import EventHandle, EventOverlap

__all__ = ["Buffer", "ElasticBuffer", "EPHandle", "Config", "EventHandle", "EventOverlap"]
