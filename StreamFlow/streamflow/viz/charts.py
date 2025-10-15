"""
Visualization module for StreamFlow - provides real-time charting and dashboard capabilities.
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class ChartConfig:
    """Configuration for chart visualization."""
    chart_type: str  # "line", "scatter", "bar", "histogram", "heatmap"
    title: str
    x_field: str
    y_field: str
    width: int = 800
    height: int = 600
    update_interval: float = 1.0  # seconds


class RealTimeChart:
    """Real-time chart for streaming data visualization."""

    def __init__(self, config: ChartConfig):
        self.config = config
        self.data_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = 1000
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Initialize Plotly chart data
        self.x_data = []
        self.y_data = []
        self.timestamps = []

    def add_data(self, data: Dict[str, Any]) -> None:
        """Add new data point to the chart."""
        with self._lock:
            # Extract data based on configuration
            x_val = data.get(self.config.x_field)
            y_val = data.get(self.config.y_field)

            if x_val is not None and y_val is not None:
                timestamp = data.get("_timestamp", time.time())

                self.data_buffer.append({
                    "x": x_val,
                    "y": y_val,
                    "timestamp": timestamp,
                    "data": data
                })

                # Maintain buffer size
                if len(self.data_buffer) > self.max_buffer_size:
                    self.data_buffer.pop(0)

    def get_plotly_data(self) -> List[Dict[str, Any]]:
        """Get data formatted for Plotly."""
        with self._lock:
            if not self.data_buffer:
                return []

            # Update chart data
            self.x_data = [item["x"] for item in self.data_buffer]
            self.y_data = [item["y"] for item in self.data_buffer]
            self.timestamps = [item["timestamp"] for item in self.data_buffer]

            if self.config.chart_type == "line":
                return [{
                    "x": self.x_data,
                    "y": self.y_data,
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": self.config.y_field,
                    "line": {"width": 2}
                }]
            elif self.config.chart_type == "scatter":
                return [{
                    "x": self.x_data,
                    "y": self.y_data,
                    "type": "scatter",
                    "mode": "markers",
                    "name": self.config.y_field,
                    "marker": {"size": 8}
                }]
            elif self.config.chart_type == "bar":
                return [{
                    "x": self.x_data,
                    "y": self.y_data,
                    "type": "bar",
                    "name": self.config.y_field
                }]
            elif self.config.chart_type == "histogram":
                return [{
                    "x": self.y_data,
                    "type": "histogram",
                    "name": self.config.y_field,
                    "nbinsx": 30
                }]
            else:
                # Default to scatter plot
                return [{
                    "x": self.x_data,
                    "y": self.y_data,
                    "type": "scatter",
                    "mode": "markers",
                    "name": self.config.y_field
                }]


    def get_plotly_layout(self) -> Dict[str, Any]:
        """Get Plotly layout configuration."""
        return {
            "title": self.config.title,
            "width": self.config.width,
            "height": self.config.height,
            "xaxis": {
                "title": self.config.x_field,
                "showgrid": True
            },
            "yaxis": {
                "title": self.config.y_field,
                "showgrid": True
            },
            "hovermode": "closest"
        }

    @property
    def title(self) -> str:
        """Get chart title."""
        return self.config.title

    @property
    def width(self) -> int:
        """Get chart width."""
        return self.config.width

    @property
    def height(self) -> int:
        """Get chart height."""
        return self.config.height

    def clear_data(self) -> None:
        """Clear all chart data."""
        with self._lock:
            self.data_buffer.clear()
            self.x_data.clear()
            self.y_data.clear()
            self.timestamps.clear()


class Dashboard:
    """Interactive dashboard for multiple charts."""

    def __init__(self, title: str = "StreamFlow Dashboard"):
        self.title = title
        self.charts: Dict[str, RealTimeChart] = {}
        self._running = False
        self._server_thread: Optional[threading.Thread] = None

    def add_chart(self, name: str, chart: RealTimeChart) -> None:
        """Add a chart to the dashboard."""
        self.charts[name] = chart

    def remove_chart(self, name: str) -> None:
        """Remove a chart from the dashboard."""
        if name in self.charts:
            del self.charts[name]

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for the entire dashboard."""
        dashboard_data = {
            "title": self.title,
            "timestamp": time.time(),
            "charts": {}
        }

        for name, chart in self.charts.items():
            dashboard_data["charts"][name] = {
                "config": {
                    "chart_type": chart.config.chart_type,
                    "title": chart.config.title,
                    "x_field": chart.config.x_field,
                    "y_field": chart.config.y_field
                },
                "data": chart.get_plotly_data(),
                "layout": chart.get_plotly_layout()
            }

        return dashboard_data

    def export_html(self, filename: str) -> None:
        """Export dashboard as HTML file."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly

            # Create subplots for multiple charts
            if len(self.charts) == 1:
                chart_name = list(self.charts.keys())[0]
                chart = self.charts[chart_name]
                fig = go.Figure(data=chart.get_plotly_data(), layout=chart.get_plotly_layout())
            else:
                # Create subplot layout
                rows = (len(self.charts) + 1) // 2  # 2 charts per row
                cols = min(2, len(self.charts))

                subplot_titles = [chart.config.title for chart in self.charts.values()]
                fig = make_subplots(
                    rows=rows, cols=cols,
                    subplot_titles=subplot_titles,
                    specs=[[{"type": "scatter"}] * cols] * rows
                )

                for i, (name, chart) in enumerate(self.charts.items()):
                    row = i // cols + 1
                    col = i % cols + 1

                    for trace in chart.get_plotly_data():
                        fig.add_trace(trace, row=row, col=col)

                    # Update layout for each subplot
                    fig.update_xaxes(title_text=chart.config.x_field, row=row, col=col)
                    fig.update_yaxes(title_text=chart.config.y_field, row=row, col=col)

            # Update overall layout
            fig.update_layout(
                title=self.title,
                height=max(600, 300 * len(self.charts)),
                showlegend=True
            )

            # Save as HTML
            plotly.offline.plot(fig, filename=filename, auto_open=False)
            logger.info(f"Dashboard exported to {filename}")

        except ImportError:
            logger.error("Plotly not installed. Install with: pip install plotly")
        except Exception as e:
            logger.error(f"Error exporting dashboard: {e}")


class StreamFlowVisualizer:
    """Main visualizer class for StreamFlow."""

    def __init__(self):
        self.charts: Dict[str, RealTimeChart] = {}
        self.dashboards: Dict[str, Dashboard] = {}
        self._data_handlers: List[Callable] = []

    def create_chart(self, name: str, config: ChartConfig) -> RealTimeChart:
        """Create a new real-time chart."""
        chart = RealTimeChart(config)
        self.charts[name] = chart
        return chart

    def create_dashboard(self, name: str, title: str = "StreamFlow Dashboard") -> Dashboard:
        """Create a new dashboard."""
        dashboard = Dashboard(title)
        self.dashboards[name] = dashboard
        return dashboard

    def add_data_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Add a data handler that will be called when new data arrives."""
        self._data_handlers.append(handler)

    def process_data(self, data: Dict[str, Any]) -> None:
        """Process incoming data and update all charts."""
        # Update all charts with new data
        for chart in self.charts.values():
            chart.add_data(data)

        # Call data handlers
        for handler in self._data_handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Error in data handler: {e}")

    def get_chart_data(self, chart_name: str) -> Optional[Dict[str, Any]]:
        """Get current data for a specific chart."""
        if chart_name in self.charts:
            chart = self.charts[chart_name]
            return {
                "data": chart.get_plotly_data(),
                "layout": chart.get_plotly_layout(),
                "config": {
                    "chart_type": chart.config.chart_type,
                    "title": chart.config.title,
                    "x_field": chart.config.x_field,
                    "y_field": chart.config.y_field
                }
            }
        return None

    def export_dashboard(self, dashboard_name: str, filename: str) -> None:
        """Export a dashboard as HTML."""
        if dashboard_name in self.dashboards:
            self.dashboards[dashboard_name].export_html(filename)

    def clear_all_data(self) -> None:
        """Clear all chart data."""
        for chart in self.charts.values():
            chart.clear_data()


# Convenience functions for quick chart creation
def create_line_chart(title: str, x_field: str, y_field: str, **kwargs) -> ChartConfig:
    """Create a line chart configuration."""
    return ChartConfig(
        chart_type="line",
        title=title,
        x_field=x_field,
        y_field=y_field,
        **kwargs
    )


def create_scatter_chart(title: str, x_field: str, y_field: str, **kwargs) -> ChartConfig:
    """Create a scatter chart configuration."""
    return ChartConfig(
        chart_type="scatter",
        title=title,
        x_field=x_field,
        y_field=y_field,
        **kwargs
    )


def create_bar_chart(title: str, x_field: str, y_field: str, **kwargs) -> ChartConfig:
    """Create a bar chart configuration."""
    return ChartConfig(
        chart_type="bar",
        title=title,
        x_field=x_field,
        y_field=y_field,
        **kwargs
    )


def create_histogram(title: str, y_field: str, **kwargs) -> ChartConfig:
    """Create a histogram configuration."""
    return ChartConfig(
        chart_type="histogram",
        title=title,
        x_field="index",
        y_field=y_field,
        **kwargs
    )
