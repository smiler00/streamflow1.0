"""
Core module for StreamFlow - handles data stream connections and management.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Callable
from dataclasses import dataclass
from enum import Enum

# Import concrete data source implementations (lazy imports)
# from .mqtt_source import MQTTDataSource
# from .api_source import APIDataSource
# from .websocket_source import WebSocketDataSource
# from .redis_source import RedisDataSource
# from .file_source import FileDataSource

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Supported data source types."""

    MQTT = "mqtt"
    API = "api"
    WEBSOCKET = "websocket"
    REDIS = "redis"
    FILE = "file"
    MOCK = "mock"


@dataclass
class ConnectionConfig:
    """Configuration for data source connections."""

    source_type: DataSourceType
    url: str
    options: Optional[Dict[str, Any]] = None


class DataSource(ABC):
    """Abstract base class for data sources."""

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self._connected = False
        self._buffer: asyncio.Queue = asyncio.Queue(maxsize=1000)

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""
        pass

    @abstractmethod
    async def read_data(self) -> AsyncIterator[Dict[str, Any]]:
        """Read data from the source."""
        pass

    async def is_connected(self) -> bool:
        """Check if the source is connected."""
        return self._connected

    async def get_buffered_data(self) -> List[Dict[str, Any]]:
        """Get all buffered data."""
        data = []
        while not self._buffer.empty():
            try:
                item = self._buffer.get_nowait()
                data.append(item)
            except asyncio.QueueEmpty:
                break
        return data


class StreamFlow:
    """
    Main StreamFlow class for data stream processing.

    This class provides the main interface for connecting to data sources,
    processing streams, and managing the data flow.
    """

    def __init__(self, source: Union[str, ConnectionConfig, DataSource]):
        """
        Initialize StreamFlow with a data source.

        Args:
            source: Data source configuration (URL string, config object, or DataSource instance)
        """
        if isinstance(source, str):
            # Parse URL to determine source type and create config
            self.config = self._parse_source_url(source)
        elif isinstance(source, ConnectionConfig):
            self.config = source
        elif isinstance(source, DataSource):
            self.data_source = source
            self.config = source.config
        else:
            raise ValueError(
                "Source must be a URL string, ConnectionConfig, or DataSource instance"
            )

        self._processors: List[Callable] = []
        self._running = False
        self._background_task: Optional[asyncio.Task] = None

    def _parse_source_url(self, url: str) -> ConnectionConfig:
        """Parse source URL to create ConnectionConfig."""
        if url.startswith("mock://"):
            return ConnectionConfig(source_type=DataSourceType.MOCK, url=url)
        elif (
            url.startswith("http://")
            or url.startswith("https://")
            or url.startswith("api://")
        ):
            source_type = DataSourceType.API
        elif url.startswith("ws://") or url.startswith("wss://"):
            source_type = DataSourceType.WEBSOCKET
        elif url.startswith("redis://"):
            source_type = DataSourceType.REDIS
        elif url.startswith("file://"):
            source_type = DataSourceType.FILE
        elif url.startswith("mqtt://"):
            source_type = DataSourceType.MQTT
        else:
            raise ValueError(f"Unsupported source URL format: {url}")

        return ConnectionConfig(source_type=source_type, url=url)

    def _create_data_source(self) -> DataSource:
        """Create appropriate DataSource instance based on config."""
        if self.config.source_type == DataSourceType.MQTT:
            # Lazy import to avoid dependency issues during testing
            try:
                from .mqtt_source import MQTTDataSource

                return MQTTDataSource(self.config)
            except ImportError as e:
                raise ImportError(
                    "MQTT support not available. Install with: pip install paho-mqtt"
                ) from e
        elif self.config.source_type == DataSourceType.API:
            try:
                from .api_source import APIDataSource

                return APIDataSource(self.config)
            except ImportError as e:
                raise ImportError(
                    "API support not available. Install with: pip install requests aiohttp"
                ) from e
        elif self.config.source_type == DataSourceType.WEBSOCKET:
            try:
                from .websocket_source import WebSocketDataSource

                return WebSocketDataSource(self.config)
            except ImportError as e:
                raise ImportError(
                    "WebSocket support not available. Install with: pip install websockets"
                ) from e
        elif self.config.source_type == DataSourceType.REDIS:
            try:
                from .redis_source import RedisDataSource

                return RedisDataSource(self.config)
            except ImportError as e:
                raise ImportError(
                    "Redis support not available. Install with: pip install redis"
                ) from e
        elif self.config.source_type == DataSourceType.FILE:
            try:
                from .file_source import FileDataSource

                return FileDataSource(self.config)
            except ImportError as e:
                raise ImportError("File support not available") from e
        else:
            raise ValueError(f"Unsupported data source type: {self.config.source_type}")

    async def connect(self) -> None:
        """Connect to the data source."""
        if self.data_source is None:
            self.data_source = self._create_data_source()

        await self.data_source.connect()
        logger.info(
            f"Connected to {self.config.source_type.value} source: {self.config.url}"
        )

    async def disconnect(self) -> None:
        """Disconnect from the data source."""
        if self.data_source:
            await self.data_source.disconnect()
            logger.info(
                f"Disconnected from {self.config.source_type.value} source: {self.config.url}"
            )

    def filter(self, condition: Callable[[Dict[str, Any]], bool]):
        """Add a filter processor to the pipeline."""
        self._processors.append(
            lambda data, condition=condition: data if condition(data) else None
        )
        return self

    def map(self, transform: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Add a map processor to the pipeline."""
        self._processors.append(transform)
        return self

    def process_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process data through the pipeline."""
        current_data = data

        for processor in self._processors:
            try:
                result = processor(current_data)
                if result is None:
                    return None
                current_data = result
            except Exception as e:
                logger.error(f"Error in processor: {e}")
                return None

        return current_data

    async def _run_stream(self) -> None:
        """Internal method to run the data stream."""
        try:
            async for data in self.data_source.read_data():
                if self._running:
                    processed_data = self.process_data(data)
                    if processed_data is not None:
                        # Put processed data back to buffer for downstream consumers
                        try:
                            await self.data_source._buffer.put(processed_data)
                        except asyncio.QueueFull:
                            logger.warning("Buffer full, dropping data")
        except Exception as e:
            logger.error(f"Error in stream: {e}")
        finally:
            self._running = False

    async def run(self) -> None:
        """Start the data stream processing."""
        if self._running:
            logger.warning("Stream is already running")
            return

        await self.connect()
        self._running = True
        self._background_task = asyncio.create_task(self._run_stream())
        logger.info("StreamFlow started")

    async def stop(self) -> None:
        """Stop the data stream processing."""
        if not self._running:
            return

        self._running = False

        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

        await self.disconnect()
        logger.info("StreamFlow stopped")

    def __enter__(self):
        """Context manager entry."""
        return self

    def plot(
        self,
        chart_type: str = "line",
        title: str = "StreamFlow Chart",
        x_field: str = "_timestamp",
        y_field: str = None,
        **kwargs,
    ):
        """Add visualization to the stream pipeline."""
        from ..viz.charts import StreamFlowVisualizer, ChartConfig, create_line_chart

        # Create visualizer if not exists
        if not hasattr(self, "_visualizer"):
            self._visualizer = StreamFlowVisualizer()

        # Determine y_field if not provided
        if y_field is None:
            # Try to find a numeric field in the data
            sample_data = self._get_sample_data()
            if sample_data:
                for key, value in sample_data.items():
                    if isinstance(value, (int, float)) and key != "_timestamp":
                        y_field = key
                        break
                if y_field is None:
                    y_field = "value"  # Default fallback

        # Create chart configuration
        if chart_type == "line":
            chart_config = create_line_chart(title, x_field, y_field, **kwargs)
        else:
            chart_config = ChartConfig(
                chart_type=chart_type,
                title=title,
                x_field=x_field,
                y_field=y_field,
                **kwargs,
            )

        # Create and add chart
        chart = self._visualizer.create_chart(f"{title}_{id(self)}", chart_config)

        # Add data handler to feed the chart
        def data_handler(data):
            chart.add_data(data)

        self._visualizer.add_data_handler(data_handler)

        return self

    def _get_sample_data(self) -> Optional[Dict[str, Any]]:
        """Get sample data to determine field types for visualization."""
        # This is a simplified implementation
        # In a full implementation, would buffer recent data
        return None

    async def start_dashboard(self, host: str = "localhost", port: int = 8080) -> None:
        """Start the web dashboard server."""
        from ..viz.dashboard import StreamFlowDashboard

        if not hasattr(self, "_dashboard"):
            self._dashboard = StreamFlowDashboard(host, port)

            # Connect dashboard to stream data
            def dashboard_data_handler(data):
                self._dashboard.process_data(data)

            if hasattr(self, "_visualizer"):
                self._visualizer.add_data_handler(dashboard_data_handler)

        await self._dashboard.start()
        logger.info(f"Dashboard started at http://{host}:{port}")

    async def stop_dashboard(self) -> None:
        """Stop the web dashboard server."""
        if hasattr(self, "_dashboard"):
            await self._dashboard.stop()

    async def predict(
        self, model_name: str, output_field: str = "prediction"
    ) -> "StreamFlow":
        """Add AI model prediction to the pipeline."""
        from ..ai.models import StreamFlowAI, AIPipelineIntegration

        # Create AI module if not exists
        if not hasattr(self, "_ai_module"):
            self._ai_module = StreamFlowAI()

        # Create prediction transform
        prediction_transform = AIPipelineIntegration.create_prediction_transform(
            model_name, self._ai_module, output_field
        )

        # Add as async transform
        async def async_transform(data: Dict[str, Any]) -> Dict[str, Any]:
            return await prediction_transform(data)

        # Add to pipeline (simplified - in full implementation would integrate with async pipeline)
        self._processors.append(async_transform)

        return self

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.disconnect()
    
    async def detect_anomalies(
        self, model_name: str, threshold: float = 0.5
    ) -> "StreamFlow":
        """Add anomaly detection to the pipeline."""
        from ..ai.models import StreamFlowAI, AIPipelineIntegration

        # Create AI module if not exists
        if not hasattr(self, "_ai_module"):
            self._ai_module = StreamFlowAI()

        # Create anomaly detection transform
        anomaly_transform = AIPipelineIntegration.create_anomaly_detector(
            model_name, self._ai_module, threshold
        )

        # Add to pipeline
        async def async_transform(data: Dict[str, Any]) -> Dict[str, Any]:
            return await anomaly_transform(data)

        self._processors.append(async_transform)

        return self

    def add_ai_model(
        self,
        name: str,
        model_type: str,
        model_path: str,
        input_features: List[str],
        output_features: List[str],
    ) -> "StreamFlow":
        """Add an AI model for use in the pipeline."""
        from ..ai.models import StreamFlowAI, ModelConfig

        # Create AI module if not exists
        if not hasattr(self, "_ai_module"):
            self._ai_module = StreamFlowAI()

        # Create model configuration
        config = ModelConfig(
            model_type=model_type,
            model_path=model_path,
            input_features=input_features,
            output_features=output_features,
        )

        # Add model to AI module
        self._ai_module.add_model(name, config)

        return self

    def use_plugin_transform(
        self, plugin_type: str, config: Dict[str, Any] = None
    ) -> "StreamFlow":
        """Add a plugin-based transformation to the pipeline."""
        if not hasattr(self, "_plugin_manager"):
            from ..plugins import StreamFlowPluginManager

            self._plugin_manager = StreamFlowPluginManager()
            self._plugin_manager.initialize_plugins()

        if config is None:
            config = {}

        transform = self._plugin_manager.create_extended_transform(plugin_type, config)
        if transform:
            self._processors.append(transform)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._running:
            asyncio.run(self.stop())
        if hasattr(self, "_dashboard") and self._dashboard.is_running():
            asyncio.run(self.stop_dashboard())
