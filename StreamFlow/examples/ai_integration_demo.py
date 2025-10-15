"""
Example usage of StreamFlow AI integration capabilities.
"""

import asyncio
import time
import random
import numpy as np
from streamflow import StreamFlow


def create_sample_model():
    """Create a simple sample model for demonstration."""
    # Create a simple linear model for demonstration
    # In a real scenario, this would be a trained ML model

    class SimpleModel:
        def __init__(self):
            # Simple coefficients for temperature prediction
            self.coefficients = np.array([0.5, -0.2, 0.1])

        def predict(self, features):
            # Simple linear prediction: temp = 0.5 * humidity - 0.2 * pressure + 0.1
            return np.array([np.sum(features * self.coefficients)])

        def __call__(self, features):
            return self.predict(features)

    return SimpleModel()


async def demo_ai_integration():
    """Demonstrate StreamFlow AI integration features."""

    # Create sample data generator
    class SensorDataSource:
        def __init__(self):
            self._connected = False

        async def connect(self):
            self._connected = True
            print("Connected to sensor data source")

        async def disconnect(self):
            self._connected = False
            print("Disconnected from sensor data source")

        async def read_data(self):
            """Generate sample sensor data."""
            while self._connected:
                data = {
                    "humidity": random.uniform(40, 80),
                    "pressure": random.uniform(990, 1030),
                    "wind_speed": random.uniform(0, 20),
                    "timestamp": time.time(),
                    "_source": "weather_sensor"
                }
                yield data
                await asyncio.sleep(1)  # Generate data every second

    # Create sample model (normally this would be loaded from file)
    sample_model = create_sample_model()

    # In a real scenario, you would save/load the model:
    import pickle
    with open("weather_model.pkl", "wb") as f:
        pickle.dump(sample_model, f)

    # Create StreamFlow instance
    sensor_source = SensorDataSource()
    stream = StreamFlow(sensor_source)

    # Add AI model for temperature prediction
    stream = (stream
              .add_ai_model(
                  name="weather_model",
                  model_type="sklearn",  # Using sklearn as a generic type
                  model_path="weather_model.pkl",  # In real scenario
                  input_features=["humidity", "pressure", "wind_speed"],
                  output_features=["predicted_temperature"]
              ))

    # Add prediction to pipeline
    stream = await stream.predict("weather_model", output_field="predicted_temp")

    # Add anomaly detection based on prediction
    stream = await stream.detect_anomalies("weather_model", threshold=0.3)

    # Add visualization
    stream = (stream
              .plot(chart_type="line", title="Sensor Data", y_field="humidity")
              .plot(chart_type="scatter", title="Temperature Predictions",
                    x_field="predicted_temp", y_field="humidity"))

    # Start dashboard
    await stream.start_dashboard(port=8080)

    # Run the stream
    await stream.run()

    try:
        await asyncio.sleep(30)  # Let it run for 30 seconds
    finally:
        await stream.stop()
        await stream.stop_dashboard()


async def demo_advanced_ai_features():
    """Demonstrate advanced AI features like model management."""

    from streamflow.ai import ModelManager, StreamFlowAI

    # Create AI module
    ai_module = StreamFlowAI()
    model_manager = ModelManager()

    # Register models
    model_manager.register_model(
        name="temperature_model",
        model_type="pytorch",
        model_path="models/temp_predictor.pt",
        input_features=["humidity", "pressure"],
        output_features=["temperature"]
    )

    model_manager.register_model(
        name="anomaly_detector",
        model_type="tensorflow",
        model_path="models/anomaly_model.h5",
        input_features=["temperature", "humidity", "pressure"],
        output_features=["anomaly_score"]
    )

    # Create model configs
    temp_config = model_manager.create_model_config("temperature_model")
    anomaly_config = model_manager.create_model_config("anomaly_detector")

    if temp_config:
        ai_module.add_model("temperature_model", temp_config)

    if anomaly_config:
        ai_module.add_model("anomaly_detector", anomaly_config)

    # Test predictions
    test_data = {
        "humidity": 65.0,
        "pressure": 1013.0,
        "temperature": 22.0
    }

    temp_result = await ai_module.predict("temperature_model", test_data)
    anomaly_result = await ai_module.predict("anomaly_detector", test_data)

    print(f"Temperature prediction: {temp_result}")
    print(f"Anomaly detection: {anomaly_result}")

    # List available models
    models = ai_module.list_models()
    print(f"Available models: {models}")


if __name__ == "__main__":
    print("StreamFlow AI Integration Demo")
    print("This demo shows AI model integration capabilities")
    print("Starting demo in 3 seconds...")
    print("Open http://localhost:8080 in your browser to see the dashboard")

    time.sleep(3)

    try:
        # Run basic demo
        asyncio.run(demo_ai_integration())

        # Run advanced features demo
        print("\n" + "="*50)
        print("Advanced AI Features Demo")
        asyncio.run(demo_advanced_ai_features())

    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"Demo error: {e}")
