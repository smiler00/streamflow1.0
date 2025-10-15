"""
Example usage of StreamFlow plugin system.
"""

import asyncio
import time
import random
from streamflow import StreamFlow
from streamflow.plugins import StreamFlowPluginManager


async def demo_plugin_system():
    """Demonstrate StreamFlow plugin system capabilities."""

    # Create a simple data source for testing
    class SimpleDataSource:
        def __init__(self):
            self._connected = False

        async def connect(self):
            self._connected = True
            print("Connected to simple data source")

        async def disconnect(self):
            self._connected = False
            print("Disconnected from simple data source")

        async def read_data(self):
            """Generate simple test data."""
            while self._connected:
                data = {
                    "value": random.uniform(0, 100),
                    "category": random.choice(["A", "B", "C"]),
                    "timestamp": time.time(),
                    "_source": "test_data"
                }
                yield data
                await asyncio.sleep(1)

    # Initialize plugin system
    plugin_manager = StreamFlowPluginManager()
    plugin_manager.initialize_plugins()

    # List available plugin extensions
    extensions = plugin_manager.plugin_manager.list_available_extensions()
    print("Available plugin extensions:")
    for ext_type, plugins in extensions.items():
        print(f"  {ext_type}: {plugins}")

    # Create StreamFlow instance
    data_source = SimpleDataSource()
    stream = StreamFlow(data_source)

    # Add built-in visualization
    stream = (stream
              .plot(chart_type="line", title="Plugin Demo Data", y_field="value"))

    # Use plugin-based transformation (moving average)
    stream = stream.use_plugin_transform("moving_average", {
        "window_size": 5,
        "field": "value"
    })

    # Start dashboard
    await stream.start_dashboard(port=8080)

    # Run the stream
    await stream.run()

    try:
        await asyncio.sleep(30)  # Let it run for 30 seconds
    finally:
        await stream.stop()
        await stream.stop_dashboard()


async def demo_custom_plugin():
    """Demonstrate creating and using a custom plugin."""

    # Initialize plugin system
    plugin_manager = StreamFlowPluginManager()
    plugin_manager.initialize_plugins()

    # Register example plugins
    from streamflow.plugins.examples import register_example_plugins
    register_example_plugins(plugin_manager.plugin_manager.registry)

    # Create a data source that generates 2D data for heatmap
    class HeatmapDataSource:
        def __init__(self):
            self._connected = False

        async def connect(self):
            self._connected = True
            print("Connected to heatmap data source")

        async def disconnect(self):
            self._connected = False
            print("Disconnected from heatmap data source")

        async def read_data(self):
            """Generate 2D data for heatmap."""
            while self._connected:
                # Generate data points for a heatmap
                for x in range(10):
                    for y in range(10):
                        value = (x * y) + random.uniform(-5, 5)
                        data = {
                            "x": x,
                            "y": y,
                            "value": value,
                            "timestamp": time.time(),
                            "_source": "heatmap_data"
                        }
                        yield data

                await asyncio.sleep(2)  # Update every 2 seconds

    # Create StreamFlow instance
    data_source = HeatmapDataSource()
    stream = StreamFlow(data_source)

    # Use plugin-based visualization (heatmap)
    heatmap_config = {
        "title": "2D Data Heatmap",
        "x_field": "x",
        "y_field": "y",
        "value_field": "value"
    }

    # Note: In a full implementation, would integrate heatmap plugin with visualization system
    print("Heatmap plugin would create visualization with config:", heatmap_config)

    # Add basic line chart for demonstration
    stream = stream.plot(chart_type="scatter", title="2D Data Points", x_field="x", y_field="y")

    # Start dashboard
    await stream.start_dashboard(port=8080)

    # Run the stream
    await stream.run()

    try:
        await asyncio.sleep(20)  # Let it run for 20 seconds
    finally:
        await stream.stop()
        await stream.stop_dashboard()


if __name__ == "__main__":
    print("StreamFlow Plugin System Demo")
    print("This demo shows plugin system capabilities")
    print("Starting demo in 3 seconds...")
    print("Open http://localhost:8080 in your browser to see the dashboard")

    time.sleep(3)

    try:
        # Run basic plugin demo
        asyncio.run(demo_plugin_system())

        print("\n" + "="*50)
        print("Custom Plugin Demo")
        # Run custom plugin demo
        asyncio.run(demo_custom_plugin())

    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"Demo error: {e}")
