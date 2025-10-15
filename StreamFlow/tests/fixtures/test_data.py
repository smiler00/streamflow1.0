"""
Test fixtures for StreamFlow testing.
"""

import pytest
import time

from streamflow.core.stream import ConnectionConfig, DataSourceType


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return [
        {"temperature": 20.0, "humidity": 60.0, "timestamp": time.time()},
        {"temperature": 25.0, "humidity": 65.0, "timestamp": time.time()},
        {"temperature": 30.0, "humidity": 70.0, "timestamp": time.time()},
    ]


@pytest.fixture
def large_dataset():
    """Large dataset for performance testing."""
    return [
        {
            "value": i,
            "category": "A" if i % 2 == 0 else "B",
            "timestamp": time.time() + i * 0.01,
        }
        for i in range(1000)
    ]


@pytest.fixture
def mock_streamflow_config():
    """Mock StreamFlow configuration."""
    return ConnectionConfig(
        source_type=DataSourceType.API,
        url="https://api.example.com/data",
        options={"timeout": 30, "retries": 3},
    )


@pytest.fixture
def categorical_data():
    """Categorical data for testing encoding."""
    return [
        {"weather": "sunny", "temperature": 25.0},
        {"weather": "cloudy", "temperature": 20.0},
        {"weather": "rainy", "temperature": 15.0},
        {"weather": "sunny", "temperature": 30.0},
    ]


@pytest.fixture
def model_config_pytorch():
    """PyTorch model configuration for testing."""
    return {
        "model_type": "pytorch",
        "model_path": "/test/model.pt",
        "input_features": ["temperature", "humidity"],
        "output_features": ["prediction"],
        "preprocessing": {"normalization": "zscore"},
    }


@pytest.fixture
def model_config_sklearn():
    """Scikit-learn model configuration for testing."""
    return {
        "model_type": "sklearn",
        "model_path": "/test/model.pkl",
        "input_features": ["feature1", "feature2"],
        "output_features": ["target"],
        "preprocessing": {
            "encoding": {
                "type": "onehot",
                "categorical_features": ["category"],
                "category_mappings": {"category": ["A", "B", "C"]},
            }
        },
    }


@pytest.fixture
def chart_config():
    """Chart configuration for testing."""
    from streamflow.viz.charts import ChartConfig

    return ChartConfig(
        chart_type="line",
        title="Test Chart",
        x_field="timestamp",
        y_field="temperature",
    )


@pytest.fixture
def plugin_config():
    """Plugin configuration for testing."""
    return {"plugin_type": "transform", "window_size": 10, "field": "value"}
