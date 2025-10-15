"""
Example plugins for StreamFlow plugin system.
"""

import asyncio
import logging
import random
import time
from typing import Any, Dict, List, Callable
from .manager import (
    DataSourcePlugin,
    TransformPlugin,
    VisualizationPlugin,
    create_plugin_metadata,
    register_plugin_with_metadata,
)

logger = logging.getLogger(__name__)


# Example Data Source Plugin
@register_plugin_with_metadata(
    create_plugin_metadata(
        name="random_generator",
        version="1.0.0",
        description="Generates random data for testing",
        author="StreamFlow Team",
        plugin_type="source",
        data_generator="RandomDataSource",
    )
)
class RandomDataSourcePlugin(DataSourcePlugin):
    """Plugin that generates random data for testing purposes."""

    def create_data_source(self, config: Dict[str, Any]) -> Any:
        """Create a random data source."""
        return RandomDataSource(config)


class RandomDataSource:
    """Data source that generates random values."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._connected = False
        self._fields = config.get("fields", ["value"])
        self._interval = config.get("interval", 1.0)
        self._min_value = config.get("min_value", 0)
        self._max_value = config.get("max_value", 100)

    async def connect(self) -> None:
        """Connect to the random data source."""
        self._connected = True
        logger.info("Connected to random data source")

    async def disconnect(self) -> None:
        """Disconnect from the random data source."""
        self._connected = False
        logger.info("Disconnected from random data source")

    async def read_data(self):
        """Generate random data."""
        while self._connected:
            data = {"_timestamp": time.time(), "_source": "random_generator"}

            # Generate random values for each field
            for field in self._fields:
                data[field] = random.uniform(self._min_value, self._max_value)

            yield data
            await asyncio.sleep(self._interval)

    async def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


# Example Transform Plugin
@register_plugin_with_metadata(
    create_plugin_metadata(
        name="moving_average",
        version="1.0.0",
        description="Calculates moving average of numeric fields",
        author="StreamFlow Team",
        plugin_type="transform",
        moving_average_transform="MovingAverageTransform",
    )
)
class MovingAverageTransformPlugin(TransformPlugin):
    """Plugin that calculates moving averages."""

    def create_transform(self, config: Dict[str, Any]) -> Callable:
        """Create a moving average transform."""
        window_size = config.get("window_size", 10)
        field = config.get("field", "value")

        def moving_average_transform(
            data_list: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            """Apply moving average transformation."""
            if len(data_list) < window_size:
                return data_list

            result = []
            values = [item.get(field, 0) for item in data_list]

            for i in range(len(data_list)):
                if i < window_size - 1:
                    # Not enough data for full window
                    result.append(data_list[i])
                else:
                    # Calculate moving average
                    window_values = values[i - window_size + 1 : i + 1]
                    avg_value = sum(window_values) / window_size

                    # Create new data point with moving average
                    new_data = data_list[i].copy()
                    new_data[f"{field}_moving_avg"] = avg_value
                    result.append(new_data)

            return result

        return moving_average_transform


# Example Visualization Plugin
@register_plugin_with_metadata(
    create_plugin_metadata(
        name="heatmap_generator",
        version="1.0.0",
        description="Creates heatmap visualizations",
        author="StreamFlow Team",
        plugin_type="visualization",
        heatmap_viz="HeatmapVisualization",
    )
)
class HeatmapVisualizationPlugin(VisualizationPlugin):
    """Plugin that creates heatmap visualizations."""

    def create_visualization(self, config: Dict[str, Any]) -> Any:
        """Create a heatmap visualization."""
        return HeatmapVisualization(config)


class HeatmapVisualization:
    """Heatmap visualization for 2D data."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.title = config.get("title", "Heatmap")
        self.x_field = config.get("x_field", "x")
        self.y_field = config.get("y_field", "y")
        self.value_field = config.get("value_field", "value")
        self.data_points = []

    def add_data(self, data: Dict[str, Any]) -> None:
        """Add data point to heatmap."""
        x = data.get(self.x_field)
        y = data.get(self.y_field)
        value = data.get(self.value_field)

        if x is not None and y is not None and value is not None:
            self.data_points.append(
                {
                    "x": x,
                    "y": y,
                    "value": value,
                    "timestamp": data.get("_timestamp", time.time()),
                }
            )

    def get_plotly_data(self) -> List[Dict[str, Any]]:
        """Get data formatted for Plotly heatmap."""
        if not self.data_points:
            return []

        # Extract unique x and y values
        x_values = sorted(list(set(point["x"] for point in self.data_points)))
        y_values = sorted(list(set(point["y"] for point in self.data_points)))

        # Create value matrix
        value_matrix = [[0 for _ in x_values] for _ in y_values]

        for point in self.data_points:
            try:
                x_idx = x_values.index(point["x"])
                y_idx = y_values.index(point["y"])
                value_matrix[y_idx][x_idx] = point["value"]
            except (ValueError, IndexError):
                continue

        return [
            {
                "x": x_values,
                "y": y_values,
                "z": value_matrix,
                "type": "heatmap",
                "colorscale": "Viridis",
            }
        ]

    def get_plotly_layout(self) -> Dict[str, Any]:
        """Get Plotly layout for heatmap."""
        return {
            "title": self.title,
            "xaxis": {"title": self.x_field},
            "yaxis": {"title": self.y_field},
        }


# Example AI Plugin
@register_plugin_with_metadata(
    create_plugin_metadata(
        name="simple_classifier",
        version="1.0.0",
        description="Simple rule-based classifier",
        author="StreamFlow Team",
        plugin_type="ai",
        classifier="SimpleClassifier",
    )
)
class SimpleClassifierPlugin:
    """Plugin that provides simple classification capabilities."""

    def create_model_loader(self, config: Dict[str, Any]) -> Callable:
        """Create a simple classifier."""
        return SimpleClassifier(config)


class SimpleClassifier:
    """Simple rule-based classifier for demonstration."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules = config.get("rules", [])
        self.field = config.get("field", "value")
        self.default_class = config.get("default_class", "unknown")

    def predict(self, features: List[float]) -> str:
        """Make prediction based on rules."""
        value = features[0] if features else 0

        for rule in self.rules:
            if rule["min"] <= value <= rule["max"]:
                return rule["class"]

        return self.default_class

    def predict_proba(self, features: List[float]) -> List[float]:
        """Return prediction probabilities (simplified)."""
        prediction = self.predict(features)
        # Simple probability assignment
        return [1.0 if prediction == self.default_class else 0.8]


# Plugin registration function for manual registration
def register_example_plugins(registry) -> None:
    """Register example plugins with the registry."""
    registry.register_plugin(
        RandomDataSourcePlugin().metadata, RandomDataSourcePlugin()
    )

    registry.register_plugin(
        MovingAverageTransformPlugin().metadata, MovingAverageTransformPlugin()
    )

    registry.register_plugin(
        HeatmapVisualizationPlugin().metadata, HeatmapVisualizationPlugin()
    )

    registry.register_plugin(
        SimpleClassifierPlugin().metadata, SimpleClassifierPlugin()
    )
