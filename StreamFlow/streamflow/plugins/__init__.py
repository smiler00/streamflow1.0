# Plugin system for extensibility

from .manager import (
    PluginRegistry,
    PluginManager,
    StreamFlowPluginManager,
    PluginBase,
    DataSourcePlugin,
    TransformPlugin,
    VisualizationPlugin,
    AIModelPlugin,
    create_plugin_metadata,
    register_plugin_with_metadata
)

from .examples import (
    RandomDataSourcePlugin,
    MovingAverageTransformPlugin,
    HeatmapVisualizationPlugin,
    SimpleClassifierPlugin,
    register_example_plugins
)

__all__ = [
    "PluginRegistry",
    "PluginManager",
    "StreamFlowPluginManager",
    "PluginBase",
    "DataSourcePlugin",
    "TransformPlugin",
    "VisualizationPlugin",
    "AIModelPlugin",
    "create_plugin_metadata",
    "register_plugin_with_metadata",
    "RandomDataSourcePlugin",
    "MovingAverageTransformPlugin",
    "HeatmapVisualizationPlugin",
    "SimpleClassifierPlugin",
    "register_example_plugins"
]
