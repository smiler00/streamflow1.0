"""
Unit tests for StreamFlow core module.
"""

import asyncio
import pytest

from streamflow.core.stream import (
    StreamFlow,
    DataSourceType,
    ConnectionConfig,
    DataSource,
)


class MockDataSource(DataSource):
    """Mock data source for testing."""

    def __init__(self, test_data=None):
        config = ConnectionConfig(DataSourceType.API, "mock://test")
        super().__init__(config)
        self.test_data = test_data or []
        self.data_index = 0

    async def connect(self):
        self._connected = True

    async def disconnect(self):
        self._connected = False

    async def read_data(self):
        while self._connected and self.data_index < len(self.test_data):
            yield self.test_data[self.data_index]
            self.data_index += 1
            await asyncio.sleep(0.01)


class TestStreamFlow:
    """Test cases for StreamFlow core functionality."""

    def test_streamflow_initialization_with_url(self):
        """Test StreamFlow initialization with URL string."""
        stream = StreamFlow("mqtt://broker:1883/test")
        assert stream.config.source_type == DataSourceType.MQTT
        assert stream.config.url == "mqtt://broker:1883/test"
        assert not stream._running

    def test_streamflow_initialization_with_config(self):
        """Test StreamFlow initialization with ConnectionConfig."""
        config = ConnectionConfig(DataSourceType.API, "https://api.example.com")
        stream = StreamFlow(config)
        assert stream.config == config

    def test_streamflow_initialization_with_datasource(self):
        """Test StreamFlow initialization with DataSource instance."""
        datasource = MockDataSource()
        stream = StreamFlow(datasource)
        assert stream.data_source == datasource

    def test_invalid_source_raises_error(self):
        """Test that invalid source raises ValueError."""
        # The error is printed but not raised in the current implementation
        # This test documents the expected behavior
        try:
            StreamFlow(123)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Source must be a URL string, ConnectionConfig, or DataSource instance" in str(e)

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test connection and disconnection."""
        datasource = MockDataSource()
        stream = StreamFlow(datasource)

        await stream.connect()
        assert stream.data_source._connected

        await stream.disconnect()
        assert not stream.data_source._connected

    @pytest.mark.asyncio
    async def test_filter_transform(self):
        """Test filter transformation."""
        stream = StreamFlow("mock://test")
        stream = stream.filter(lambda x: x.get("value", 0) > 10)
        if stream.config.url.startswith("mock://"):
            stream.config.source_type = DataSourceType.API
            stream.config.host = "test"
            stream.config.port = None


        # Test the processor
        test_data = {"value": 15}
        result = stream.process_data(test_data)
        assert result == test_data

        test_data_low = {"value": 5}
        result_low = stream.process_data(test_data_low)
        assert result_low is None

    @pytest.mark.asyncio
    async def test_map_transform(self):
        """Test map transformation."""
        stream = StreamFlow("mock://test")
        if stream.config.url.startswith("mock://"):
            stream.config.source_type = DataSourceType.API
            stream.config.host = "test"
            stream.config.port = None
        stream = stream.map(lambda x: {**x, "doubled": x.get("value", 0) * 2})

        test_data = {"value": 10}
        result = stream.process_data(test_data)
        assert result["value"] == 10
        assert result["doubled"] == 20

    @pytest.mark.asyncio
    async def test_multiple_transforms(self):
        """Test chaining multiple transformations."""
        stream = StreamFlow("mock://test")
        if stream.config.url.startswith("mock://"):
            stream.config.source_type = DataSourceType.API
            stream.config.host = "test"
            stream.config.port = None
        stream = stream.filter(lambda x: x.get("value", 0) > 0).map(
            lambda x: {**x, "processed": True}
        )

        test_data = {"value": 5}
        result = stream.process_data(test_data)
        assert result["value"] == 5
        assert result["processed"] is True

    @pytest.mark.asyncio
    async def test_run_stop_cycle(self):
        """Test basic run/stop functionality."""
        test_data = [{"value": i} for i in range(5)]
        datasource = MockDataSource(test_data)
        stream = StreamFlow(datasource)
        if stream.config.url.startswith("mock://"):
            stream.config.source_type = DataSourceType.API
            stream.config.host = "test"
            stream.config.port = None

        await stream.connect()

        # Start the stream
        await stream.run()
        assert stream._running

        # Wait a bit for data processing
        await asyncio.sleep(0.1)

        # Stop the stream
        await stream.stop()
        assert not stream._running

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test context manager functionality."""
        test_data = [{"value": i} for i in range(3)]
        datasource = MockDataSource(test_data)
        if datasource.config.url.startswith("mock://"):
            datasource.config.source_type = DataSourceType.API
            datasource.config.host = "test"
            datasource.config.port = None

        async with StreamFlow(datasource) as stream:
            await stream.connect()
            assert stream.data_source._connected

        # Should be disconnected after context
        assert not stream.data_source._connected


class TestDataSourceTypes:
    """Test different data source type parsing."""

    def test_mqtt_url_parsing(self):
        """Test MQTT URL parsing."""
        stream = StreamFlow("mqtt://broker:1883/topic")
        assert stream.config.source_type == DataSourceType.MQTT

    def test_api_url_parsing(self):
        """Test API URL parsing."""
        stream = StreamFlow("https://api.example.com/data")
        assert stream.config.source_type == DataSourceType.API

    def test_websocket_url_parsing(self):
        """Test WebSocket URL parsing."""
        stream = StreamFlow("ws://localhost:8080/stream")
        assert stream.config.source_type == DataSourceType.WEBSOCKET

    def test_redis_url_parsing(self):
        """Test Redis URL parsing."""
        stream = StreamFlow("redis://localhost:6379")
        assert stream.config.source_type == DataSourceType.REDIS

    def test_file_url_parsing(self):
        """Test file URL parsing."""
        stream = StreamFlow("file:///path/to/data.json")
        assert stream.config.source_type == DataSourceType.FILE

    def test_invalid_url_raises_error(self):
        """Test invalid URL raises error."""
        with pytest.raises(ValueError, match="Unsupported source URL format"):
            StreamFlow("invalid://format")


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_processor_error_handling(self):
        """Test that processor errors are handled gracefully."""

        def failing_processor(data):
            raise ValueError("Test error")

        stream = StreamFlow("mock://test")
        stream._processors.append(failing_processor)

        test_data = {"value": 10}
        result = stream.process_data(test_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test connection error handling."""

        class FailingDataSource(DataSource):
            async def connect(self):
                raise ConnectionError("Connection failed")

            async def disconnect(self):
                pass

            async def read_data(self):
                yield {}

        config = ConnectionConfig(DataSourceType.API, "fail://test")
        datasource = FailingDataSource(config)
        stream = StreamFlow(datasource)

        with pytest.raises(ConnectionError):
            await stream.connect()


if __name__ == "__main__":
    pytest.main([__file__])
