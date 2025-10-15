"""
Plugin system for StreamFlow - provides extensibility for adding new data sources, transformations, and visualizations.
"""

import importlib
import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import sys

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for plugin registration."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: str  # "source", "transform", "visualization", "ai"
    entry_points: Dict[str, Any]  # Plugin entry points


class PluginRegistry:
    """Registry for managing StreamFlow plugins."""

    def __init__(self):
        self.plugins: Dict[str, PluginMetadata] = {}
        self.plugin_instances: Dict[str, Any] = {}
        self._plugin_paths: List[str] = []

    def register_plugin(self, metadata: PluginMetadata, plugin_instance: Any = None) -> None:
        """Register a plugin with the system."""
        if metadata.name in self.plugins:
            logger.warning(f"Plugin '{metadata.name}' is already registered. Overwriting.")

        self.plugins[metadata.name] = metadata
        if plugin_instance:
            self.plugin_instances[metadata.name] = plugin_instance

        logger.info(f"Registered plugin: {metadata.name} v{metadata.version}")

    def unregister_plugin(self, name: str) -> bool:
        """Unregister a plugin from the system."""
        if name in self.plugins:
            del self.plugins[name]
            if name in self.plugin_instances:
                del self.plugin_instances[name]
            logger.info(f"Unregistered plugin: {name}")
            return True
        return False

    def get_plugin(self, name: str) -> Optional[Any]:
        """Get a plugin instance by name."""
        return self.plugin_instances.get(name)

    def get_plugin_metadata(self, name: str) -> Optional[PluginMetadata]:
        """Get plugin metadata by name."""
        return self.plugins.get(name)

    def list_plugins(self) -> List[str]:
        """List all registered plugin names."""
        return list(self.plugins.keys())

    def list_plugins_by_type(self, plugin_type: str) -> List[str]:
        """List plugins of a specific type."""
        return [name for name, metadata in self.plugins.items()
                if metadata.plugin_type == plugin_type]

    def add_plugin_path(self, path: str) -> None:
        """Add a path to search for plugins."""
        if path not in self._plugin_paths:
            self._plugin_paths.append(path)
            # Add to Python path for imports
            if path not in sys.path:
                sys.path.insert(0, path)

    def discover_plugins(self, search_paths: Optional[List[str]] = None) -> int:
        """Discover and load plugins from specified paths."""
        paths_to_search = search_paths or self._plugin_paths
        discovered_count = 0

        for path in paths_to_search:
            path_obj = Path(path)
            if not path_obj.exists():
                continue

            # Look for plugin entry points
            for plugin_file in path_obj.rglob("*.py"):
                if plugin_file.name.startswith("__"):
                    continue

                try:
                    # Import the module
                    module_name = plugin_file.stem
                    spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # Look for plugin registration functions
                        if hasattr(module, 'register_plugin'):
                            module.register_plugin(self)
                            discovered_count += 1

                except Exception as e:
                    logger.warning(f"Failed to load plugin from {plugin_file}: {e}")

        logger.info(f"Discovered {discovered_count} plugins")
        return discovered_count


class PluginBase:
    """Base class for StreamFlow plugins."""

    def __init__(self):
        self.metadata = self._get_metadata()

    def _get_metadata(self) -> PluginMetadata:
        """Get plugin metadata. Override in subclasses."""
        return PluginMetadata(
            name="base_plugin",
            version="1.0.0",
            description="Base plugin class",
            author="StreamFlow",
            plugin_type="base",
            entry_points={}
        )

    def register(self, registry: PluginRegistry) -> None:
        """Register this plugin with the registry."""
        registry.register_plugin(self.metadata, self)


class DataSourcePlugin(PluginBase):
    """Plugin for adding new data sources."""

    def __init__(self):
        super().__init__()
        self.metadata.plugin_type = "source"

    def create_data_source(self, config: Dict[str, Any]) -> Any:
        """Create a data source instance. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement create_data_source")


class TransformPlugin(PluginBase):
    """Plugin for adding new data transformations."""

    def __init__(self):
        super().__init__()
        self.metadata.plugin_type = "transform"

    def create_transform(self, config: Dict[str, Any]) -> Callable:
        """Create a transform function. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement create_transform")


class VisualizationPlugin(PluginBase):
    """Plugin for adding new visualization types."""

    def __init__(self):
        super().__init__()
        self.metadata.plugin_type = "visualization"

    def create_visualization(self, config: Dict[str, Any]) -> Any:
        """Create a visualization instance. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement create_visualization")


class AIModelPlugin(PluginBase):
    """Plugin for adding new AI model types."""

    def __init__(self):
        super().__init__()
        self.metadata.plugin_type = "ai"

    def create_model_loader(self, config: Dict[str, Any]) -> Callable:
        """Create a model loader function. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement create_model_loader")


class PluginManager:
    """High-level manager for StreamFlow plugins."""

    def __init__(self):
        self.registry = PluginRegistry()
        self._initialized = False

    def initialize(self, plugin_paths: Optional[List[str]] = None) -> None:
        """Initialize the plugin system."""
        if self._initialized:
            return

        # Add default plugin paths
        default_paths = [
            str(Path(__file__).parent / "plugins"),
            str(Path.cwd() / "plugins"),
            str(Path.home() / ".streamflow" / "plugins")
        ]

        for path in default_paths:
            self.registry.add_plugin_path(path)

        # Add user-specified paths
        if plugin_paths:
            for path in plugin_paths:
                self.registry.add_plugin_path(path)

        # Discover plugins
        self.registry.discover_plugins()

        # Register example plugins for development/testing
        try:
            from .examples import register_example_plugins
            register_example_plugins(self.registry)
            logger.info("Registered example plugins")
        except ImportError:
            logger.debug("Example plugins not available")

        self._initialized = True
        logger.info("Plugin system initialized")

    def get_data_source(self, source_type: str, config: Dict[str, Any]) -> Optional[Any]:
        """Get a data source plugin by type."""
        # Try the expected naming pattern first
        plugin_name = f"{source_type}_source"
        plugin = self.registry.get_plugin(plugin_name)

        if plugin and hasattr(plugin, 'create_data_source'):
            return plugin.create_data_source(config)

        # Try the source_type directly as fallback
        plugin = self.registry.get_plugin(source_type)
        if plugin and hasattr(plugin, 'create_data_source'):
            return plugin.create_data_source(config)

        return None

    def get_transform(self, transform_type: str, config: Dict[str, Any]) -> Optional[Callable]:
        """Get a transform plugin by type."""
        # Try the expected naming pattern first
        plugin_name = f"{transform_type}_transform"
        plugin = self.registry.get_plugin(plugin_name)

        if plugin and hasattr(plugin, 'create_transform'):
            return plugin.create_transform(config)

        # Try the transform_type directly as fallback
        plugin = self.registry.get_plugin(transform_type)
        if plugin and hasattr(plugin, 'create_transform'):
            return plugin.create_transform(config)

        return None

    def get_visualization(self, viz_type: str, config: Dict[str, Any]) -> Optional[Any]:
        """Get a visualization plugin by type."""
        plugin_name = f"{viz_type}_visualization"
        plugin = self.registry.get_plugin(plugin_name)

        if plugin and hasattr(plugin, 'create_visualization'):
            return plugin.create_visualization(config)

        return None

    def get_ai_model_loader(self, model_type: str, config: Dict[str, Any]) -> Optional[Callable]:
        """Get an AI model loader plugin by type."""
        plugin_name = f"{model_type}_ai"
        plugin = self.registry.get_plugin(plugin_name)

        if plugin and hasattr(plugin, 'create_model_loader'):
            return plugin.create_model_loader(config)

        return None

    def list_available_extensions(self) -> Dict[str, List[str]]:
        """List all available plugin extensions by type."""
        extensions = {
            "sources": self.registry.list_plugins_by_type("source"),
            "transforms": self.registry.list_plugins_by_type("transform"),
            "visualizations": self.registry.list_plugins_by_type("visualization"),
            "ai_models": self.registry.list_plugins_by_type("ai")
        }

        return extensions


class StreamFlowPluginManager:
    """Integration of plugin system with StreamFlow core."""

    def __init__(self):
        self.plugin_manager = PluginManager()

    @property
    def registry(self):
        """Access to the plugin registry."""
        return self.plugin_manager.registry

    def initialize_plugins(self, plugin_paths: Optional[List[str]] = None) -> None:
        """Initialize the plugin system for StreamFlow."""
        self.plugin_manager.initialize(plugin_paths)

    def create_extended_data_source(self, source_type: str, config: Dict[str, Any]) -> Optional[Any]:
        """Create a data source using plugins if available."""
        # First try plugins
        plugin_source = self.plugin_manager.get_data_source(source_type, config)
        if plugin_source:
            return plugin_source

        # Fallback to built-in sources
        from ..core.stream import DataSourceType, ConnectionConfig

        try:
            source_type_enum = DataSourceType(source_type)
            connection_config = ConnectionConfig(
                source_type=source_type_enum,
                url=config.get("url", ""),
                options=config
            )
            return self._create_builtin_data_source(connection_config)
        except ValueError:
            return None

    def create_extended_transform(self, transform_type: str, config: Dict[str, Any]) -> Optional[Callable]:
        """Create a transform using plugins if available."""
        # First try plugins
        plugin_transform = self.plugin_manager.get_transform(transform_type, config)
        if plugin_transform:
            return plugin_transform

        # Fallback to built-in transforms
        return self._create_builtin_transform(transform_type, config)

    def _create_builtin_data_source(self, config) -> Optional[Any]:
        """Create built-in data source."""
        from ..core.stream import StreamFlow

        # Create a minimal StreamFlow instance to get the data source creation logic
        temp_stream = StreamFlow.__new__(StreamFlow)
        temp_stream.config = config
        return temp_stream._create_data_source()

    def _create_builtin_transform(self, transform_type: str, config: Dict[str, Any]) -> Optional[Callable]:
        """Create built-in transform."""
        from ..transform.operations import TransformOperations

        transforms = TransformOperations()

        if hasattr(transforms, transform_type):
            transform_method = getattr(transforms, transform_type)
            if callable(transform_method):
                return transform_method(**config)

        return None


# Convenience functions for plugin development
def create_plugin_metadata(name: str, version: str, description: str,
                          author: str, plugin_type: str, **entry_points) -> PluginMetadata:
    """Create plugin metadata for easy plugin registration."""
    return PluginMetadata(
        name=name,
        version=version,
        description=description,
        author=author,
        plugin_type=plugin_type,
        entry_points=entry_points
    )


def register_plugin_with_metadata(metadata: PluginMetadata) -> Callable:
    """Decorator for registering plugins with metadata."""
    def decorator(cls):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.metadata = metadata

        cls.__init__ = new_init
        cls.metadata = metadata
        return cls

    return decorator
