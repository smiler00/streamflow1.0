"""
Unit tests for StreamFlow AI module.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
import pytest
from unittest.mock import Mock, patch

from streamflow.ai.models import (
    ModelConfig, ModelPredictor, StreamFlowAI,
    create_pytorch_model_config, create_tensorflow_model_config,
    create_sklearn_model_config
)
from streamflow.ai.pipeline import AIPipelineProcessor


def setup_module(module):
    """Set up test data before running tests."""
    os.makedirs("test", exist_ok=True)
    # Create a simple linear model that always returns 0.8
    X = np.array([[1], [2], [3]])
    y = np.array([0.8, 0.8, 0.8])
    model = LinearRegression().fit(X, y)
    joblib.dump(model, "test/model.pkl")


class MockModel:
    """Mock ML model for testing."""

    def __init__(self, prediction_value=0.5):
        self.prediction_value = prediction_value
        self.predict = Mock()
        self.predict.return_value = np.array([[prediction_value]])

    def eval(self):
        pass


class TestModelConfig:
    """Test cases for ModelConfig."""

    def test_model_config_creation(self):
        """Test ModelConfig initialization."""
        config = ModelConfig(
            model_type="pytorch",
            model_path="/path/to/model.pt",
            input_features=["feature1", "feature2"],
            output_features=["prediction"]
        )

        assert config.model_type == "pytorch"
        assert config.model_path == "/path/to/model.pt"
        assert config.input_features == ["feature1", "feature2"]
        assert config.output_features == ["prediction"]

    def test_model_config_with_preprocessing(self):
        """Test ModelConfig with preprocessing configuration."""
        config = ModelConfig(
            model_type="tensorflow",
            model_path="/path/to/model.h5",
            input_features=["temp", "humidity"],
            output_features=["prediction"],
            preprocessing={
                "normalization": "zscore",
                "scaling": "log"
            }
        )

        assert "normalization" in config.preprocessing
        assert "scaling" in config.preprocessing


class TestModelPredictor:
    """Test cases for ModelPredictor."""

    def test_predictor_initialization(self):
        """Test ModelPredictor initialization."""
        config = ModelConfig(
            model_type="sklearn",
            model_path="/path/to/model.pkl",
            input_features=["x"],
            output_features=["y"]
        )

        predictor = ModelPredictor(config)
        assert predictor.config == config
        assert not predictor._loaded

    @pytest.mark.asyncio
    async def test_feature_extraction_numeric(self):
        """Test feature extraction for numeric data."""
        config = ModelConfig(
            model_type="sklearn",
            model_path="test/model.pkl",
            input_features=["temp", "humidity", "pressure"],
            output_features=["prediction"]
        )

        predictor = ModelPredictor(config)

        test_data = {
            "temp": 25.0,
            "humidity": 60.0,
            "pressure": 1013.0,
            "extra": "ignored"
        }

        features = predictor._extract_features(test_data)

        assert features.shape == (1, 3)
        assert features[0, 0] == 25.0  # temp
        assert features[0, 1] == 60.0  # humidity
        assert features[0, 2] == 1013.0  # pressure

    @pytest.mark.asyncio
    async def test_feature_extraction_categorical(self):
        """Test feature extraction for categorical data."""
        config = ModelConfig(
            model_type="sklearn",
            model_path="test/model.pkl",
            input_features=["weather"],
            output_features=["prediction"],
            preprocessing={
                "encoding": {
                    "type": "onehot",
                    "categorical_features": ["weather"],
                    "category_mappings": {
                        "weather": ["sunny", "cloudy", "rainy"]
                    }
                }
            }
        )

        predictor = ModelPredictor(config)

        test_data = {"weather": "sunny"}
        features = predictor._extract_features(test_data)

        # Should have 3 features for one-hot encoding
        assert features.shape == (1, 3)
        assert features[0, 0] == 1.0  # sunny
        assert features[0, 1] == 0.0  # cloudy
        assert features[0, 2] == 0.0  # rainy

    @pytest.mark.asyncio
    async def test_preprocessing_normalization(self):
        """Test preprocessing normalization."""
        config = ModelConfig(
            model_type="sklearn",
            model_path="test/model.pkl",
            input_features=["value"],
            output_features=["prediction"],
            preprocessing={
                "normalization": "minmax",
                "min_vals": [0.0],
                "max_vals": [100.0]
            }
        )

        predictor = ModelPredictor(config)
        test_data = np.array([[50.0]])  # Should normalize to 0.5

        processed = predictor._apply_preprocessing(test_data)
        assert np.allclose(processed, [[0.5]])

    @pytest.mark.asyncio
    async def test_postprocessing_denormalization(self):
        """Test postprocessing denormalization."""
        config = ModelConfig(
            model_type="sklearn",
            model_path="test/model.pkl",
            input_features=["value"],
            output_features=["prediction"],
            postprocessing={
                "denormalization": "minmax",
                "min_vals": [0.0],
                "max_vals": [100.0]
            }
        )

        predictor = ModelPredictor(config)
        prediction = np.array([[0.5]])  # Should denormalize to 50.0

        processed = predictor._apply_postprocessing(prediction)
        assert np.allclose(processed, [[50.0]])

    @patch('torch.load')
    @pytest.mark.asyncio
    async def test_pytorch_model_loading(self, mock_torch_load):
        """Test PyTorch model loading."""
        mock_model = Mock()
        mock_torch_load.return_value = mock_model

        config = ModelConfig(
            model_type="pytorch",
            model_path="test/model.pt",
            input_features=["x"],
            output_features=["y"]
        )

        predictor = ModelPredictor(config)
        await predictor.load_model()

        assert predictor._loaded
        mock_torch_load.assert_called_once()

    @patch('tensorflow.keras.models.load_model')
    @pytest.mark.asyncio
    async def test_tensorflow_model_loading(self, mock_tf_load):
        """Test TensorFlow model loading."""
        mock_model = Mock()
        mock_tf_load.return_value = mock_model

        config = ModelConfig(
            model_type="tensorflow",
            model_path="test/model.h5",
            input_features=["x"],
            output_features=["y"]
        )

        predictor = ModelPredictor(config)
        await predictor.load_model()

        assert predictor._loaded
        mock_tf_load.assert_called_once()

    @patch('pickle.load')
    @pytest.mark.asyncio
    async def test_sklearn_model_loading(self, mock_pickle_load):
        """Test scikit-learn model loading."""
        mock_model = Mock()
        mock_pickle_load.return_value = mock_model

        config = ModelConfig(
            model_type="sklearn",
            model_path="test/model.pkl",
            input_features=["x"],
            output_features=["y"]
        )

        predictor = ModelPredictor(config)
        await predictor.load_model()

        assert predictor._loaded

    @pytest.mark.asyncio
    async def test_prediction_success(self):
        """Test successful prediction."""
        with patch('pickle.load') as mock_pickle:
            mock_model = MockModel(0.8)
            mock_pickle.return_value = mock_model

            config = ModelConfig(
                model_type="sklearn",
                model_path="test/model.pkl",
                input_features=["x"],
                output_features=["y"]
            )

            predictor = ModelPredictor(config)
            await predictor.load_model()

            test_data = {"x": 10.0}
            result = await predictor.predict(test_data)

            assert result["success"]
            assert "prediction" in result
            assert result["prediction"] == [0.8]

    @pytest.mark.asyncio
    async def test_prediction_error(self):
        """Test prediction error handling."""
        with patch('pickle.load') as mock_pickle:
            mock_model = MockModel()
            mock_model.predict.side_effect = Exception("Model error")
            mock_pickle.return_value = mock_model

            config = ModelConfig(
                model_type="sklearn",
                model_path="test/model.pkl",
                input_features=["x"],
                output_features=["y"]
            )

            predictor = ModelPredictor(config)
            await predictor.load_model()

            test_data = {"x": 10.0}
            result = await predictor.predict(test_data)

            assert not result["success"]
            assert "error" in result


class TestStreamFlowAI:
    """Test cases for StreamFlowAI."""

    def test_ai_module_initialization(self):
        """Test StreamFlowAI initialization."""
        ai_module = StreamFlowAI()

        assert len(ai_module.predictors) == 0
        assert len(ai_module._running_predictions) == 0

    def test_add_model(self):
        """Test adding a model."""
        ai_module = StreamFlowAI()

        config = ModelConfig(
            model_type="sklearn",
            model_path="test/model.pkl",
            input_features=["x"],
            output_features=["y"]
        )

        predictor = ai_module.add_model("test_model", config)

        assert "test_model" in ai_module.predictors
        assert ai_module.predictors["test_model"] == predictor

    def test_remove_model(self):
        """Test removing a model."""
        ai_module = StreamFlowAI()

        config = ModelConfig(
            model_type="sklearn",
            model_path="test\model.pkl",
            input_features=["x"],
            output_features=["y"]
        )

        ai_module.add_model("test_model", config)
        assert "test_model" in ai_module.predictors

        removed = ai_module.remove_model("test_model")
        assert removed
        assert "test_model" not in ai_module.predictors

    def test_list_models(self):
        """Test listing models."""
        ai_module = StreamFlowAI()

        # Add some models
        config1 = ModelConfig("sklearn", "../test/test1.pkl", ["x"], ["y"])
        config2 = ModelConfig("pytorch", "../test/test2.pt", ["a"], ["b"])

        ai_module.add_model("model1", config1)
        ai_module.add_model("model2", config2)

        models = ai_module.list_models()
        assert "model1" in models
        assert "model2" in models
        assert len(models) == 2

    def test_get_model_info(self):
        """Test getting model information."""
        ai_module = StreamFlowAI()

        config = ModelConfig(
            model_type="tensorflow",
            model_path="test/model.h5",
            input_features=["temp", "humidity"],
            output_features=["prediction"]
        )

        ai_module.add_model("weather_model", config)

        info = ai_module.get_model_info("weather_model")

        assert info is not None
        assert info["model_type"] == "tensorflow"
        assert info["input_features"] == ["temp", "humidity"]
        assert info["output_features"] == ["prediction"]


class TestAIPipelineProcessor:
    """Test cases for AIPipelineProcessor."""

    @pytest.mark.asyncio
    async def test_predict_transform_creation(self):
        """Test prediction transform creation."""
        ai_module = StreamFlowAI()
        processor = AIPipelineProcessor(ai_module)

        predict_func = processor.predict("test_model", "output")

        # Test that it returns a function
        assert callable(predict_func)

        # The function should be async
        import inspect
        assert inspect.iscoroutinefunction(predict_func)

    @pytest.mark.asyncio
    async def test_anomaly_detection_transform(self):
        """Test anomaly detection transform creation."""
        ai_module = StreamFlowAI()
        processor = AIPipelineProcessor(ai_module)

        anomaly_func = processor.detect_anomalies("test_model", threshold=0.5)

        assert callable(anomaly_func)
        import inspect
        assert inspect.iscoroutinefunction(anomaly_func)

    @pytest.mark.asyncio
    async def test_classification_transform(self):
        """Test classification transform creation."""
        ai_module = StreamFlowAI()
        processor = AIPipelineProcessor(ai_module)

        classify_func = processor.classify("test_model", target_class=1)

        assert callable(classify_func)
        import inspect
        assert inspect.iscoroutinefunction(classify_func)


class TestModelConfigCreationFunctions:
    """Test convenience functions for model config creation."""

    def test_create_pytorch_config(self):
        """Test PyTorch model config creation."""
        config = create_pytorch_model_config(
            "/path/to/model.pt",
            ["input1", "input2"],
            ["output1"]
        )

        assert config.model_type == "pytorch"
        assert config.model_path == "/path/to/model.pt"
        assert config.input_features == ["input1", "input2"]
        assert config.output_features == ["output1"]

    def test_create_tensorflow_config(self):
        """Test TensorFlow model config creation."""
        config = create_tensorflow_model_config(
            "/path/to/model.h5",
            ["feature1"],
            ["prediction"]
        )

        assert config.model_type == "tensorflow"
        assert config.model_path == "/path/to/model.h5"

    def test_create_sklearn_config(self):
        """Test scikit-learn model config creation."""
        config = create_sklearn_model_config(
            "/path/to/model.pkl",
            ["x", "y"],
            ["result"]
        )

        assert config.model_type == "sklearn"
        assert config.model_path == "/path/to/model.pkl"


if __name__ == "__main__":
    pytest.main([__file__])
