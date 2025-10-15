"""
Web dashboard server for StreamFlow real-time visualization.
"""

import logging
import time
from typing import Any, Dict, Optional
import aiohttp
from aiohttp import web

# Import chart types
from .charts import StreamFlowVisualizer

logger = logging.getLogger(__name__)


class DashboardServer:
    """Web server for serving real-time dashboards."""

    def __init__(self, visualizer, host: str = "localhost", port: int = 8080):
        self.visualizer = visualizer
        self.host = host
        self.port = port
        self.app = web.Application()
        self.runner: Optional[aiohttp.AppRunner] = None
        self.site: Optional[aiohttp.TCPSite] = None
        self._running = False

        # Set up routes
        self.app.router.add_get("/", self.dashboard_page)
        self.app.router.add_get("/api/charts", self.get_charts_data)
        self.app.router.add_get("/api/charts/{chart_name}", self.get_chart_data)
        self.app.router.add_get(
            "/api/dashboard/{dashboard_name}", self.get_dashboard_data
        )
        self.app.router.add_static("/static", "streamflow/viz/static")

    async def start(self) -> None:
        """Start the dashboard server."""
        if self._running:
            return

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

        self._running = True
        logger.info(f"Dashboard server started at http://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the dashboard server."""
        if not self._running:
            return

        if self.site:
            await self.site.stop()

        if self.runner:
            await self.runner.cleanup()

        self._running = False
        logger.info("Dashboard server stopped")

    async def dashboard_page(self, request: web.Request) -> web.Response:
        """Serve the main dashboard page."""
        html = self._generate_dashboard_html()
        return web.Response(text=html, content_type="text/html")

    async def get_charts_data(self, request: web.Request) -> web.Response:
        """Get data for all charts."""
        data = {"charts": {}, "timestamp": time.time()}

        for name, chart in self.visualizer.charts.items():
            data["charts"][name] = self.visualizer.get_chart_data(name)

        return web.json_response(data)

    async def get_chart_data(self, request: web.Request) -> web.Response:
        """Get data for a specific chart."""
        chart_name = request.match_info["chart_name"]

        chart_data = self.visualizer.get_chart_data(chart_name)
        if chart_data:
            return web.json_response(chart_data)
        else:
            return web.json_response({"error": "Chart not found"}, status=404)

    async def get_dashboard_data(self, request: web.Request) -> web.Response:
        """Get data for a specific dashboard."""
        dashboard_name = request.match_info["dashboard_name"]

        if dashboard_name in self.visualizer.dashboards:
            dashboard = self.visualizer.dashboards[dashboard_name]
            data = dashboard.get_dashboard_data()
            return web.json_response(data)
        else:
            return web.json_response({"error": "Dashboard not found"}, status=404)

    def _generate_dashboard_html(self) -> str:
        """Generate HTML for the dashboard."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StreamFlow Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chart-title {
            margin: 0 0 15px 0;
            color: #333;
            font-size: 18px;
        }
        .chart {
            width: 100%;
            height: 400px;
        }
        .controls {
            margin-bottom: 20px;
            text-align: center;
        }
        .btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin: 0 5px;
        }
        .btn:hover {
            background: #0056b3;
        }
        .status {
            margin-top: 10px;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }
        .status.connected {
            background: #d4edda;
            color: #155724;
        }
        .status.disconnected {
            background: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>StreamFlow Real-Time Dashboard</h1>
            <div class="controls">
                <button class="btn" onclick="refreshData()">Refresh Data</button>
                <button class="btn" onclick="clearAllData()">Clear All</button>
                <button class="btn" onclick="exportDashboard()">Export</button>
            </div>
            <div id="status" class="status disconnected">
                Disconnected - Waiting for data...
            </div>
        </div>

        <div class="charts-grid" id="chartsContainer">
            <!-- Charts will be inserted here dynamically -->
        </div>
    </div>

    <script>
        let charts = {};
        let isConnected = false;

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            loadCharts();
            setInterval(updateAllCharts, 1000);
        });

        async function loadCharts() {
            try {
                const response = await fetch('/api/charts');
                const data = await response.json();

                const container = document.getElementById('chartsContainer');
                container.innerHTML = '';

                for (const [chartName, chartInfo] of Object.entries(data.charts)) {
                    if (chartInfo) {
                        createChart(chartName, chartInfo);
                    }
                }

                updateConnectionStatus(true);
            } catch (error) {
                console.error('Error loading charts:', error);
                updateConnectionStatus(false);
            }
        }

        function createChart(chartName, chartInfo) {
            const container = document.getElementById('chartsContainer');

            const chartDiv = document.createElement('div');
            chartDiv.className = 'chart-container';

            const title = document.createElement('h3');
            title.className = 'chart-title';
            title.textContent = chartInfo.config.title;

            const chartElement = document.createElement('div');
            chartElement.className = 'chart';
            chartElement.id = `chart-${chartName}`;

            chartDiv.appendChild(title);
            chartDiv.appendChild(chartElement);
            container.appendChild(chartDiv);

            // Create Plotly chart
            Plotly.newPlot(chartElement.id, chartInfo.data, chartInfo.layout, {responsive: true});
            charts[chartName] = chartElement.id;
        }

        async function updateAllCharts() {
            if (Object.keys(charts).length === 0) {
                await loadCharts();
                return;
            }

            try {
                const response = await fetch('/api/charts');
                const data = await response.json();

                for (const [chartName, chartInfo] of Object.entries(data.charts)) {
                    if (chartInfo && charts[chartName]) {
                        Plotly.update(charts[chartName], chartInfo.data, chartInfo.layout);
                    }
                }

                updateConnectionStatus(true);
            } catch (error) {
                console.error('Error updating charts:', error);
                updateConnectionStatus(false);
            }
        }

        function updateConnectionStatus(connected) {
            const statusDiv = document.getElementById('status');
            if (connected) {
                statusDiv.textContent = 'Connected - Receiving real-time data...';
                statusDiv.className = 'status connected';
                isConnected = true;
            } else {
                statusDiv.textContent = 'Disconnected - Attempting to reconnect...';
                statusDiv.className = 'status disconnected';
                isConnected = false;
            }
        }

        function refreshData() {
            updateAllCharts();
        }

        function clearAllData() {
            for (const chartId of Object.values(charts)) {
                Plotly.purge(chartId);
            }
            charts = {};
            loadCharts();
        }

        function exportDashboard() {
            // Simple export - in a full implementation would generate downloadable file
            alert('Export functionality would generate a downloadable HTML file with current charts.');
        }

        // Handle page visibility changes
        document.addEventListener('visibilitychange', function() {
            if (!document.hidden && !isConnected) {
                updateAllCharts();
            }
        });
    </script>
</body>
</html>
        """

class StreamFlowDashboard:
    """Integrated dashboard for StreamFlow with web server."""

    def __init__(self, host: str = "localhost", port: int = 8080, title: str = "StreamFlow Dashboard"):
        self.title = title  # Use the passed title parameter correctly
        self._charts = {}
        self.visualizer = StreamFlowVisualizer()
        self.server = DashboardServer(self.visualizer, host, port)
        self._running = False

    async def start(self) -> None:
        """Start the dashboard server."""
        self._running = True

    async def stop(self) -> None:
        """Stop the dashboard server."""
        await self.server.stop()
        self._running = False

    def add_dashboard(self, name: str, title: str = "StreamFlow Dashboard"):
        """Add a dashboard."""
        return self.visualizer.create_dashboard(name, title)

    def process_data(self, data: Dict[str, Any]) -> None:
        """Process incoming data for visualization."""
        self.visualizer.process_data(data)

    def export_dashboard_html(self, dashboard_name: str, filename: str) -> None:
        """Export a dashboard as HTML file."""
        self.visualizer.export_dashboard(dashboard_name, filename)

    def is_running(self) -> bool:
        """Check if the dashboard server is running."""
        return self._running

    @property
    def charts(self):
        """Get all charts in the dashboard."""
        return self._charts

    @charts.setter
    def charts(self, value):
        """Set charts in the dashboard."""
        self._charts = value

    def add_chart(self, chart_id: str, chart):
        """Add a chart to the dashboard."""
        self._charts[chart_id] = chart

    def remove_chart(self, chart_id: str):
        """Remove a chart from the dashboard."""
        if chart_id in self._charts:
            del self._charts[chart_id]

    def export_data(self):
        """Export dashboard data."""
        return {"title": self.title, "charts": list(self._charts.keys())}

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for export."""
        return {
            "title": self.title,
            "charts": {
                name: {
                    "data": chart.get_plotly_data(),
                    "layout": chart.get_plotly_layout()
                }
                for name, chart in self._charts.items()
            }
        }
