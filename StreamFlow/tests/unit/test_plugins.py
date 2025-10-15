"""
Unit tests for StreamFlow plugin system.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from streamflow.plugins.manager import (
    PluginRegistry, PluginManager, StreamFlowPluginManager,
    PluginBase, TransformPlugin,
    create_plugin_metadata, register_plugin_with_metadata
)
from streamflow.plugins.examples import (
    RandomDataSourcePlugin, MovingAverageTransformPlugin,
    HeatmapVisualizationPlugin, SimpleClassifierPlugin
)


class TestPluginMetadata:
    """Test cases for PluginMetadata."""

    def test_plugin_metadata_creation(self):
        """Test PluginMetadata initialization."""
        metadata = create_plugin_metadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            plugin_type="source",
            custom_field="custom_value"
        )

        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test plugin"
        assert metadata.author == "Test Author"
        assert metadata.plugin_type == "source"
        assert metadata.entry_points["custom_field"] == "custom_value"


class TestPluginRegistry:
    """Test cases for PluginRegistry."""

    def test_registry_initialization(self):
        """Test PluginRegistry initialization."""
        registry = PluginRegistry()

        assert len(registry.plugins) == 0
        assert len(registry.plugin_instances) == 0
        assert len(registry._plugin_paths) == 0

    def test_plugin_registration(self):
        """Test plugin registration."""
        registry = PluginRegistry()

        metadata = create_plugin_metadata(
            name="test_plugin",
            version="1.0.0",
            description="Test",
            author="Test",
            plugin_type="source"
        )

        plugin_instance = Mock()
        registry.register_plugin(metadata, plugin_instance)

        assert "test_plugin" in registry.plugins
        assert registry.plugins["test_plugin"] == metadata
        assert registry.plugin_instances["test_plugin"] == plugin_instance

    def test_plugin_unregistration(self):
        """Test plugin unregistration."""
        registry = PluginRegistry()

        metadata = create_plugin_metadata(
            name="test_plugin",
            version="1.0.0",
            description="Test",
            author="Test",
            plugin_type="source"
        )

        registry.register_plugin(metadata)

        removed = registry.unregister_plugin("test_plugin")
        assert removed
        assert "test_plugin" not in registry.plugins

    def test_get_plugin(self):
        """Test getting plugin instance."""
        registry = PluginRegistry()

        metadata = create_plugin_metadata(
            name="test_plugin",
            version="1.0.0",
            description="Test",
            author="Test",
            plugin_type="source"
        )

        plugin_instance = Mock()
        registry.register_plugin(metadata, plugin_instance)

        retrieved = registry.get_plugin("test_plugin")
        assert retrieved == plugin_instance

        # Test non-existent plugin
        not_found = registry.get_plugin("non_existent")
        assert not_found is None

    def test_list_plugins_by_type(self):
        """Test listing plugins by type."""
        registry = PluginRegistry()

        # Register plugins of different types
        source_metadata = create_plugin_metadata(
            name="source_plugin",
            version="1.0.0",
            description="Source",
            author="Test",
            plugin_type="source"
        )

        transform_metadata = create_plugin_metadata(
            name="transform_plugin",
            version="1.0.0",
            description="Transform",
            author="Test",
            plugin_type="transform"
        )

        registry.register_plugin(source_metadata)
        registry.register_plugin(transform_metadata)

        source_plugins = registry.list_plugins_by_type("source")
        transform_plugins = registry.list_plugins_by_type("transform")

        assert "source_plugin" in source_plugins
        assert "transform_plugin" in transform_plugins
        assert len(source_plugins) == 1
        assert len(transform_plugins) == 1


class TestPluginBase:
    """Test cases for PluginBase."""

    def test_plugin_base_metadata(self):
        """Test PluginBase metadata handling."""
        class TestPlugin(PluginBase):
            def _get_metadata(self):
                return create_plugin_metadata(
                    name="test",
                    version="1.0.0",
                    description="Test plugin",
                    author="Test",
                    plugin_type="base"
                )

        plugin = TestPlugin()
        assert plugin.metadata.name == "test"
        assert plugin.metadata.plugin_type == "base"


class TestExamplePlugins:
    """Test cases for example plugins."""

    def test_random_data_source_plugin(self):
        """Test RandomDataSourcePlugin."""
        plugin = RandomDataSourcePlugin()

        assert plugin.metadata.name == "random_generator"
        assert plugin.metadata.plugin_type == "source"

        # Test data source creation
        config = {"fields": ["value"], "interval": 0.1}
        datasource = plugin.create_data_source(config)

        assert datasource is not None
        assert hasattr(datasource, 'connect')
        assert hasattr(datasource, 'read_data')

    def test_moving_average_transform_plugin(self):
        """Test MovingAverageTransformPlugin."""
        plugin = MovingAverageTransformPlugin()

        assert plugin.metadata.name == "moving_average"
        assert plugin.metadata.plugin_type == "transform"

        # Test transform creation
        config = {"window_size": 5, "field": "value"}
        transform = plugin.create_transform(config)

        assert callable(transform)

    def test_heatmap_visualization_plugin(self):
        """Test HeatmapVisualizationPlugin."""
        plugin = HeatmapVisualizationPlugin()

        assert plugin.metadata.name == "heatmap_generator"
        assert plugin.metadata.plugin_type == "visualization"

        # Test visualization creation
        config = {"title": "Test Heatmap"}
        visualization = plugin.create_visualization(config)

        assert visualization is not None
        assert hasattr(visualization, 'add_data')
        assert hasattr(visualization, 'get_plotly_data')

    def test_simple_classifier_plugin(self):
        """Test SimpleClassifierPlugin."""
        plugin = SimpleClassifierPlugin()

        assert plugin.metadata.name == "simple_classifier"
        assert plugin.metadata.plugin_type == "ai"

        # Test model loader creation
        config = {"rules": [{"min": 0, "max": 50, "class": "low"}]}
        classifier = plugin.create_model_loader(config)

        assert classifier is not None
        assert hasattr(classifier, 'predict')


class TestPluginManager:
    """Test cases for PluginManager."""

    def test_plugin_manager_initialization(self):
        """Test PluginManager initialization."""
        manager = PluginManager()

        assert manager.registry is not None
        assert not manager._initialized

    def test_plugin_initialization(self):
        """Test plugin system initialization."""
        manager = PluginManager()

        # Test with custom paths
        with tempfile.TemporaryDirectory() as temp_dir:
            manager.initialize([temp_dir])

            assert manager._initialized
            # Should have added default paths
            assert len(manager.registry._plugin_paths) > 0

    def test_get_data_source_plugin(self):
        """Test getting data source plugin."""
        manager = PluginManager()
        manager.initialize()

        # Register a test plugin
        plugin = RandomDataSourcePlugin()
        manager.registry.register_plugin(plugin.metadata, plugin)

        config = {"test": "config"}
        datasource = manager.get_data_source("random_generator", config)

        assert datasource is not None

    def test_get_transform_plugin(self):
        """Test getting transform plugin."""
        manager = PluginManager()
        manager.initialize()

        # Register a test plugin
        plugin = MovingAverageTransformPlugin()
        manager.registry.register_plugin(plugin.metadata, plugin)

        config = {"window_size": 5}
        transform = manager.get_transform("moving_average", config)

        assert callable(transform)

    def test_plugin_discovery(self):
        """Test plugin discovery functionality."""
        manager = PluginManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake plugin file
            plugin_file = Path(temp_dir) / "fake_plugin.py"
            plugin_file.write_text("""
def register_plugin(registry):
    from streamflow.plugins.manager import create_plugin_metadata
    metadata = create_plugin_metadata(
        name="fake_plugin",
        version="1.0.0",
        description="Fake plugin for testing",
        author="Test",
        plugin_type="source"
    )
    registry.register_plugin(metadata)
""")

            manager.registry.add_plugin_path(temp_dir)
            discovered = manager.registry.discover_plugins([temp_dir])

            # Should discover the fake plugin
            assert discovered >= 0


class TestStreamFlowPluginManager:
    """Test cases for StreamFlowPluginManager."""

    def test_plugin_manager_integration(self):
        """Test integration with StreamFlow."""
        manager = StreamFlowPluginManager()

        assert manager.plugin_manager is not None
        assert manager.registry is not None

    def test_extended_data_source_creation(self):
        """Test creating extended data sources."""
        manager = StreamFlowPluginManager()
        manager.initialize_plugins()

        # Register a plugin
        plugin = RandomDataSourcePlugin()
        manager.plugin_manager.registry.register_plugin(plugin.metadata, plugin)

        config = {"fields": ["value"]}
        datasource = manager.create_extended_data_source("random_generator", config)

        assert datasource is not None

    def test_extended_transform_creation(self):
        """Test creating extended transforms."""
        manager = StreamFlowPluginManager()
        manager.initialize_plugins()

        # Register a plugin
        plugin = MovingAverageTransformPlugin()
        manager.plugin_manager.registry.register_plugin(plugin.metadata, plugin)

        config = {"window_size": 5}
        transform = manager.create_extended_transform("moving_average", config)

        assert callable(transform)


class TestPluginDecorator:
    """Test cases for plugin decorator."""

    def test_plugin_decorator_registration(self):
        """Test plugin decorator functionality."""
        @register_plugin_with_metadata(
            create_plugin_metadata(
                name="decorated_plugin",
                version="1.0.0",
                description="Decorated test plugin",
                author="Test",
                plugin_type="transform"
            )
        )
        class DecoratedPlugin(TransformPlugin):
            def create_transform(self, config):
                return lambda x: x

        plugin = DecoratedPlugin()

        assert plugin.metadata.name == "decorated_plugin"
        assert plugin.metadata.plugin_type == "transform"


if __name__ == "__main__":
    pytest.main([__file__])
