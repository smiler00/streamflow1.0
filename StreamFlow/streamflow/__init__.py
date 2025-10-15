# StreamFlow - Real-time data stream processing library

from .core import StreamFlow, ConnectionConfig, DataSourceType
from .transform.operations import WindowConfig, WindowProcessor

__version__ = "0.1.0"
__all__ = ["StreamFlow", "ConnectionConfig", "DataSourceType", "WindowConfig", "WindowProcessor"]
