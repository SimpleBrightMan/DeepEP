"""Public buffer declarations; no native resources are initialized on import."""

from .elastic import ElasticBuffer, EPHandle
from .legacy import Buffer

__all__ = ["Buffer", "ElasticBuffer", "EPHandle"]
