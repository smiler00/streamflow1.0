"""
Example usage of StreamFlow visualization capabilities.
"""

import asyncio
import time
import random
from streamflow import StreamFlow


async def demo_visualization():
    """Demonstrate StreamFlow visualization features."""

    # Create a data source that generates sample data
    # In a real scenario, this would be connected to a real data source
    class DemoDataSource:
        def __init__(self):
            self._connected = False

        async def connect(self):
            self._connected = True
            print("Connected to demo data source")

        async def disconnect(self):
            self._connected = False
            print("Disconnected from demo data source")

        async def read_data(self):
            """Generate sample temperature and humidity data."""
            while self._connected:
                # Simulate sensor data
                data = {
                    "temperature": random.uniform(20, 30),
                    "humidity": random.uniform(40, 80),
                    "timestamp": time.time(),
                    "_source": "demo_sensor"
                }
                yield data
                await asyncio.sleep(1)  # Generate data every second

    # Create StreamFlow instance
    demo_source = DemoDataSource()
    stream = StreamFlow(demo_source)

    # Add filtering and transformation
    stream = (stream
              .filter(lambda x: x["temperature"] > 25)  # Only high temperatures
              .map(lambda x: {**x, "temp_category": "high" if x["temperature"] > 28 else "normal"}))

    # Add real-time visualization
    stream = (stream
              .plot(chart_type="line", title="Temperature Over Time", x_field="timestamp", y_field="temperature")
              .plot(chart_type="scatter", title="Temperature vs Humidity", x_field="temperature", y_field="humidity"))

    # Start the web dashboard
    await stream.start_dashboard(port=8080)

    # Run the stream for 30 seconds
    await stream.run()

    try:
        await asyncio.sleep(30)  # Let it run for 30 seconds
    finally:
        await stream.stop()
        await stream.stop_dashboard()


if __name__ == "__main__":
    print("StreamFlow Visualization Demo")
    print("Starting demo in 3 seconds...")
    print("Open http://localhost:8080 in your browser to see the dashboard")

    time.sleep(3)

    try:
        asyncio.run(demo_visualization())
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
