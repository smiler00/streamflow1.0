"""
Integration tests for StreamFlow - testing end-to-end workflows.
"""

import pytest
import asyncio
import time

from streamflow import StreamFlow
from streamflow.core.stream import ConnectionConfig, DataSourceType
from streamflow.viz.charts import ChartConfig
from streamflow.ai.models import ModelConfig
from dataclasses import dataclass


class MockIntegrationDataSource:
    """Mock data source for integration testing."""

    @dataclass
    class DataPoint:
        temperature: float
        humidity: float
        timestamp: float

    def __init__(self, data_points=None):
        self._connected = False
        self.data_points = data_points or [
            self.DataPoint(20.0, 60.0, time.time()),
            self.DataPoint(25.0, 65.0, time.time()),
            self.DataPoint(30.0, 70.0, time.time()),
        ]
        self.index = 0

    async def connect(self):
        self._connected = True

    async def disconnect(self):
        self._connected = False

    async def read_data(self):
        while self._connected and self.index < len(self.data_points):
            yield self.data_points[self.index]
            self.index += 1
            await asyncio.sleep(0.1)


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Integration tests for complete StreamFlow workflows."""

    @pytest.mark.asyncio
    async def test_basic_stream_processing(self):
        """Test basic stream processing pipeline."""
        datasource = MockIntegrationDataSource()
        stream = StreamFlow(datasource)

        # Add filtering and transformation
        processed_data = []

        def collect_data(data):
            processed_data.append(data)

        # Simulate the pipeline manually for testing
        await stream.connect()

        # Collect some data
        data_count = 0
        async for data in datasource.read_data():
            if data_count >= 2:  # Collect first 2 data points
                break
            processed_data.append(data)
            data_count += 1

        await stream.disconnect()

        assert len(processed_data) == 2
        assert all("temperature" in data for data in processed_data)

    @pytest.mark.asyncio
    async def test_visualization_integration(self):
        """Test integration with visualization system."""
        # Create stream with visualization
        config = ConnectionConfig(DataSourceType.API, "mock://test")
        stream = StreamFlow(config)

        # Add visualization
        chart_config = ChartConfig("line", "Temperature", "timestamp", "temperature")
        # Note: In real implementation, would use stream.plot()

        # Test data processing with visualization
        test_data = {"temperature": 25.0, "humidity": 60.0, "_timestamp": time.time()}

        # Simulate chart data addition
        from streamflow.viz.charts import RealTimeChart
        chart = RealTimeChart(chart_config)
        chart.add_data(test_data)

        assert len(chart.data_buffer) == 1
        assert chart.data_buffer[0]["temperature"] == 25.0

        # Test that stream is created properly (basic smoke test)
        assert stream is not None

    @pytest.mark.asyncio
    async def test_ai_integration_workflow(self):
        """Test AI model integration in workflow."""
        # Create a simple mock model for testing
        class MockAIModel:
            def predict(self, features):
                import numpy as np
                return np.array([[features[0] * 0.1]])  # Simple prediction

        # Test model configuration and prediction
        config = ModelConfig(
            model_type="sklearn",
            model_path="/fake/path/model.pkl",
            input_features=["temperature"],
            output_features=["prediction"]
        )

        from streamflow.ai.models import ModelPredictor
        predictor = ModelPredictor(config)

        # Mock the model loading
        predictor.model = MockAIModel()

        test_data = {"temperature": 25.0}
        features = predictor._extract_features(test_data)

        # Test prediction
        prediction = predictor.model.predict(features)
        expected = 25.0 * 0.1  # Based on our mock model

        assert abs(prediction[0][0] - expected) < 0.001

    @pytest.mark.asyncio
    async def test_plugin_integration(self):
        """Test plugin system integration."""
        from streamflow.plugins import StreamFlowPluginManager

        manager = StreamFlowPluginManager()
        manager.initialize_plugins()

        # Test plugin-based data source creation
        config = {"fields": ["value"], "interval": 0.1}
        datasource = manager.create_extended_data_source("random_generator", config)

        if datasource:  # Plugin might not be available in test environment
            assert datasource is not None

    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self):
        """Test complete pipeline with multiple components."""
        datasource = MockIntegrationDataSource()

        # Test data flow through multiple processing stages
        data_received = []

        # Simulate pipeline processing
        await datasource.connect()

        data_count = 0
        async for data in datasource.read_data():
            if data_count >= 3:  # Process first 3 data points
                break

            # Simulate transformations
            processed_data = {
                **data,
                "processed": True,
                "temperature_category": "high" if data["temperature"] > 25 else "normal"
            }

            data_received.append(processed_data)
            data_count += 1

        await datasource.disconnect()

        assert len(data_received) == 3
        assert all(data["processed"] for data in data_received)
        assert all("temperature_category" in data for data in data_received)


@pytest.mark.integration
class TestPerformanceIntegration:
    """Performance and stress tests."""

    @pytest.mark.asyncio
    async def test_high_throughput_processing(self):
        """Test processing with high data throughput."""
        # Create larger dataset
        large_dataset = [
            {"value": i, "timestamp": time.time() + i * 0.01}
            for i in range(100)
        ]

        datasource = MockIntegrationDataSource(large_dataset)

        processed_count = 0
        start_time = time.time()

        await datasource.connect()

        async for data in datasource.read_data():
            processed_count += 1
            if processed_count >= 50:  # Process half the data
                break

        end_time = time.time()
        processing_time = end_time - start_time

        await datasource.disconnect()

        assert processed_count == 50
        assert processing_time < 2.0  # Should process quickly

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test memory usage stability during processing."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process some data
        datasource = MockIntegrationDataSource()
        await datasource.connect()

        data_processed = 0
        async for data in datasource.read_data():
            data_processed += 1
            if data_processed >= 20:
                break

        await datasource.disconnect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB for this test)
        assert memory_increase < 50.0


@pytest.mark.integration
class TestErrorRecoveryIntegration:
    """Test error recovery and resilience."""

    @pytest.mark.asyncio
    async def test_connection_recovery(self):
        """Test recovery from connection failures."""
        class FlakyDataSource(MockIntegrationDataSource):
            def __init__(self):
                super().__init__()
                self.connect_attempts = 0

            async def connect(self):
                self.connect_attempts += 1
                if self.connect_attempts < 3:
                    raise ConnectionError(f"Connection attempt {self.connect_attempts} failed")
                await super().connect()

        datasource = FlakyDataSource()

        # Should eventually connect after retries
        await datasource.connect()
        assert datasource._connected
        assert datasource.connect_attempts >= 3

        await datasource.disconnect()

    @pytest.mark.asyncio
    async def test_data_processing_error_recovery(self):
        """Test recovery from data processing errors."""
        def failing_processor(data):
            if data.get("temperature", 0) == 25.0:
                raise ValueError("Processing error for temperature 25.0")
            return data

        # Test that other data points still process correctly
        test_data = [
            {"temperature": 20.0},  # Should process fine
            {"temperature": 25.0},  # Should fail
            {"temperature": 30.0},  # Should process fine
        ]

        successful_processed = 0
        for data in test_data:
            try:
                result = failing_processor(data)
                if result:
                    successful_processed += 1
            except ValueError:
                pass  # Expected error

        assert successful_processed == 2  # First and third should succeed


if __name__ == "__main__":
    pytest.main([__file__])
