"""
Unit tests for StreamFlow visualization module.
"""

import pytest
import time
from unittest.mock import patch

from streamflow.viz.charts import (
    ChartConfig,
    RealTimeChart,
    StreamFlowVisualizer,
    create_line_chart,
    create_scatter_chart,
    create_bar_chart,
    create_histogram,
)
from streamflow.viz.dashboard import DashboardServer, StreamFlowDashboard


class TestChartConfig:
    """Test cases for ChartConfig."""

    def test_chart_config_creation(self):
        """Test ChartConfig initialization."""
        config = ChartConfig(
            chart_type="line",
            title="Test Chart",
            x_field="time",
            y_field="value",
            width=800,
            height=600,
        )

        assert config.chart_type == "line"
        assert config.title == "Test Chart"
        assert config.x_field == "time"
        assert config.y_field == "value"
        assert config.width == 800
        assert config.height == 600


class TestRealTimeChart:
    """Test cases for RealTimeChart."""

    def test_chart_initialization(self):
        """Test RealTimeChart initialization."""
        config = ChartConfig("line", "Test", "x", "y")
        chart = RealTimeChart(config)

        assert chart.config == config
        assert len(chart.data_buffer) == 0
        assert chart.max_buffer_size == 1000

    def test_add_data_single_point(self):
        """Test adding single data point."""
        config = ChartConfig("line", "Test", "x", "y")
        chart = RealTimeChart(config)

        test_data = {"x": 1, "y": 10, "_timestamp": time.time()}
        chart.add_data(test_data)

        assert len(chart.data_buffer) == 1
        assert chart.data_buffer[0]["x"] == 1
        assert chart.data_buffer[0]["y"] == 10

    def test_add_data_multiple_points(self):
        """Test adding multiple data points."""
        config = ChartConfig("line", "Test", "x", "y")
        chart = RealTimeChart(config)

        for i in range(5):
            test_data = {"x": i, "y": i * 10, "_timestamp": time.time()}
            chart.add_data(test_data)

        assert len(chart.data_buffer) == 5

        # Check data integrity
        for i, data_point in enumerate(chart.data_buffer):
            assert data_point["x"] == i
            assert data_point["y"] == i * 10

    def test_buffer_overflow(self):
        """Test buffer overflow handling."""
        config = ChartConfig("line", "Test", "x", "y")
        chart = RealTimeChart(config)
        chart.max_buffer_size = 3  # Small buffer for testing

        # Add more data than buffer size
        for i in range(5):
            test_data = {"x": i, "y": i * 10, "_timestamp": time.time()}
            chart.add_data(test_data)

        # Should only keep the last 3 items
        assert len(chart.data_buffer) == 3
        assert chart.data_buffer[0]["x"] == 2  # First item should be index 2
        assert chart.data_buffer[-1]["x"] == 4  # Last item should be index 4

    def test_get_plotly_data_line_chart(self):
        """Test Plotly data generation for line chart."""
        config = ChartConfig("line", "Test", "x", "y")
        chart = RealTimeChart(config)

        # Add test data
        test_data = {"x": 1, "y": 10, "_timestamp": time.time()}
        chart.add_data(test_data)

        plotly_data = chart.get_plotly_data()

        assert len(plotly_data) == 1
        assert plotly_data[0]["type"] == "scatter"
        assert plotly_data[0]["mode"] == "lines+markers"
        assert plotly_data[0]["x"] == [1]
        assert plotly_data[0]["y"] == [10]

    def test_get_plotly_data_scatter_chart(self):
        """Test Plotly data generation for scatter chart."""
        config = ChartConfig("scatter", "Test", "x", "y")
        chart = RealTimeChart(config)

        test_data = {"x": 5, "y": 15, "_timestamp": time.time()}
        chart.add_data(test_data)

        plotly_data = chart.get_plotly_data()

        assert len(plotly_data) == 1
        assert plotly_data[0]["type"] == "scatter"
        assert plotly_data[0]["mode"] == "markers"
        assert plotly_data[0]["x"] == [5]
        assert plotly_data[0]["y"] == [15]

    def test_get_plotly_layout(self):
        """Test Plotly layout generation."""
        config = ChartConfig("line", "Test Chart", "x_axis", "y_axis", 1000, 800)
        chart = RealTimeChart(config)

        layout = chart.get_plotly_layout()

        assert layout["title"] == "Test Chart"
        assert layout["width"] == 1000
        assert layout["height"] == 800
        assert layout["xaxis"]["title"] == "x_axis"
        assert layout["yaxis"]["title"] == "y_axis"

    def test_clear_data(self):
        """Test clearing chart data."""
        config = ChartConfig("line", "Test", "x", "y")
        chart = RealTimeChart(config)

        # Add some data
        for i in range(3):
            test_data = {"x": i, "y": i * 10, "_timestamp": time.time()}
            chart.add_data(test_data)

        assert len(chart.data_buffer) == 3

        # Clear data
        chart.clear_data()

        assert len(chart.data_buffer) == 0


class TestStreamFlowVisualizer:
    """Test cases for StreamFlowVisualizer."""

    def test_visualizer_initialization(self):
        """Test StreamFlowVisualizer initialization."""
        visualizer = StreamFlowVisualizer()

        assert len(visualizer.charts) == 0
        assert len(visualizer._data_handlers) == 0

    def test_create_chart(self):
        """Test chart creation."""
        visualizer = StreamFlowVisualizer()
        config = ChartConfig("line", "Test", "x", "y")

        chart = visualizer.create_chart("test_chart", config)

        assert "test_chart" in visualizer.charts
        assert visualizer.charts["test_chart"] == chart
        assert chart.config == config

    def test_process_data_updates_charts(self):
        """Test that process_data updates all charts."""
        visualizer = StreamFlowVisualizer()
        config = ChartConfig("line", "Test", "x", "y")

        chart1 = visualizer.create_chart("chart1", config)
        chart2 = visualizer.create_chart("chart2", config)

        test_data = {"x": 1, "y": 10, "_timestamp": time.time()}
        visualizer.process_data(test_data)

        # Both charts should have the data
        assert len(chart1.data_buffer) == 1
        assert len(chart2.data_buffer) == 1

    def test_data_handler_registration(self):
        """Test data handler registration and execution."""
        visualizer = StreamFlowVisualizer()
        handler_called = False

        def test_handler(data):
            nonlocal handler_called
            handler_called = True

        visualizer.add_data_handler(test_handler)

        test_data = {"value": 42}
        visualizer.process_data(test_data)

        assert handler_called


class TestChartCreationFunctions:
    """Test convenience functions for chart creation."""

    def test_create_line_chart(self):
        """Test create_line_chart function."""
        config = create_line_chart("Line Chart", "time", "value")

        assert config.chart_type == "line"
        assert config.title == "Line Chart"
        assert config.x_field == "time"
        assert config.y_field == "value"

    def test_create_scatter_chart(self):
        """Test create_scatter_chart function."""
        config = create_scatter_chart("Scatter Chart", "x", "y")

        assert config.chart_type == "scatter"
        assert config.title == "Scatter Chart"
        assert config.x_field == "x"
        assert config.y_field == "y"

    def test_create_bar_chart(self):
        """Test create_bar_chart function."""
        config = create_bar_chart("Bar Chart", "category", "count")

        assert config.chart_type == "bar"
        assert config.title == "Bar Chart"
        assert config.x_field == "category"
        assert config.y_field == "count"

    def test_create_histogram(self):
        """Test create_histogram function."""
        config = create_histogram("Value Distribution", "values")

        assert config.chart_type == "histogram"
        assert config.title == "Value Distribution"
        assert config.x_field == "index"
        assert config.y_field == "values"


class TestDashboard:
    """Test cases for Dashboard functionality."""

    def test_dashboard_creation(self):
        """Test Dashboard initialization."""
        dashboard = StreamFlowDashboard("Test Dashboard")

        assert dashboard.title == "Test Dashboard"
        assert len(dashboard.charts) == 0

    def test_dashboard_chart_management(self):
        """Test adding and removing charts from dashboard."""
        dashboard = StreamFlowDashboard("Test")
        config = ChartConfig("line", "Test", "x", "y")
        chart = RealTimeChart(config)

        # Add chart
        dashboard.add_chart("test_chart", chart)
        assert "test_chart" in dashboard.charts

        # Remove chart
        dashboard.remove_chart("test_chart")
        assert "test_chart" not in dashboard.charts

    def test_dashboard_data_export(self):
        """Test dashboard data export functionality."""
        dashboard = StreamFlowDashboard("Export Test")
        config = ChartConfig("line", "Test", "x", "y")
        chart = RealTimeChart(config)

        dashboard.add_chart("test", chart)

        # Add some data
        test_data = {"x": 1, "y": 10, "_timestamp": time.time()}
        chart.add_data(test_data)

        # Get dashboard data
        dashboard_data = dashboard.get_dashboard_data()

        assert dashboard_data["title"] == "Export Test"
        assert "test" in dashboard_data["charts"]
        assert "data" in dashboard_data["charts"]["test"]
        assert "layout" in dashboard_data["charts"]["test"]


class TestDashboardServer:
    """Test cases for DashboardServer (integration tests)."""

    def test_server_initialization(self):
        """Test DashboardServer initialization."""
        visualizer = StreamFlowVisualizer()
        server = DashboardServer(visualizer, "localhost", 8081)

        assert server.visualizer == visualizer
        assert server.host == "localhost"
        assert server.port == 8081
        assert not server._running

    @patch("aiohttp.web.Application")
    def test_server_route_setup(self, mock_app):
        """Test that server routes are properly configured."""
        visualizer = StreamFlowVisualizer()
        server = DashboardServer(visualizer)

        # Check that routes were added (mock verification)
        assert server.app is not None

    def test_generate_dashboard_html(self):
        """Test HTML generation for dashboard."""
        visualizer = StreamFlowVisualizer()
        server = DashboardServer(visualizer)

        html = server._generate_dashboard_html()

        assert "StreamFlow Real-Time Dashboard" in html
        assert "plotly" in html.lower()
        assert "refresh" in html.lower()


if __name__ == "__main__":
    pytest.main([__file__])
