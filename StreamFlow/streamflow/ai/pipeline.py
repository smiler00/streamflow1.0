"""
AI pipeline integration utilities for StreamFlow.
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Union

from .models import ModelConfig

logger = logging.getLogger(__name__)


class AIPipelineProcessor:
    """Processes AI operations within StreamFlow pipelines."""

    def __init__(self, ai_module):
        self.ai_module = ai_module

    def predict(self, model_name: str, output_field: str = "prediction"):
        """Create a prediction transform for the pipeline."""
        async def prediction_transform(data: Dict[str, Any]) -> Dict[str, Any]:
            prediction_result = await self.ai_module.predict(model_name, data)

            if prediction_result.get("success", False):
                # Add prediction to data
                data[output_field] = prediction_result.get("prediction", [])
                data["model_prediction"] = prediction_result
            else:
                # Add error information
                data[f"{output_field}_error"] = prediction_result.get("error", "Unknown error")

            return data

        return prediction_transform

    def detect_anomalies(self, model_name: str, threshold: float = 0.5,
                        output_field: str = "anomaly_score"):
        """Create an anomaly detection transform for the pipeline."""
        async def anomaly_transform(data: Dict[str, Any]) -> Dict[str, Any]:
            prediction_result = await self.ai_module.predict(model_name, data)

            if prediction_result.get("success", False):
                # Simple anomaly detection based on prediction confidence
                prediction = prediction_result.get("prediction", [0])[0]
                anomaly_score = abs(prediction - 0.5)  # Distance from neutral prediction

                data[output_field] = anomaly_score
                data["is_anomaly"] = anomaly_score > threshold
                data["anomaly_prediction"] = prediction_result
            else:
                data[output_field] = 0.0
                data["is_anomaly"] = False
                data["anomaly_error"] = prediction_result.get("error", "Prediction failed")

            return data

        return anomaly_transform

    def classify(self, model_name: str, target_class: Union[str, int],
                 output_field: str = "classification_result"):
        """Create a classification filter for the pipeline."""
        async def classification_transform(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            prediction_result = await self.ai_module.predict(model_name, data)

            if prediction_result.get("success", False):
                prediction = prediction_result.get("prediction", [0])[0]

                # Simple classification check
                if isinstance(target_class, int):
                    matches = prediction == target_class
                else:
                    # For string classes, would need class mapping
                    matches = str(prediction) == str(target_class)

                if matches:
                    data[output_field] = prediction_result
                    return data

            return None

        return classification_transform

    def filter_by_prediction(self, model_name: str, condition: Callable[[Dict[str, Any]], bool]):
        """Create a filter based on prediction results."""
        async def prediction_filter(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            prediction_result = await self.ai_module.predict(model_name, data)

            if prediction_result.get("success", False):
                # Combine prediction result with original data for filtering
                combined_data = {**data, "prediction_result": prediction_result}
                return combined_data if condition(combined_data) else None

            return None

        return prediction_filter


class ModelManager:
    """Utility class for managing AI models in StreamFlow."""

    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}

    def register_model(self, name: str, model_type: str, model_path: str,
                      input_features: List[str], output_features: List[str]) -> None:
        """Register a model for easy access."""
        self.models[name] = {
            "type": model_type,
            "path": model_path,
            "input_features": input_features,
            "output_features": output_features
        }

    def create_model_config(self, name: str) -> Optional['ModelConfig']:
        """Create ModelConfig from registered model."""
        if name in self.models:
            model_info = self.models[name]
            return ModelConfig(
                model_type=model_info["type"],
                model_path=model_info["path"],
                input_features=model_info["input_features"],
                output_features=model_info["output_features"]
            )
        return None

    def list_registered_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())

    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered model."""
        return self.models.get(name)
