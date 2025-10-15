"""
MQTT Data Source implementation for StreamFlow.
"""

import asyncio
import json
import logging
from typing import Any, Dict, AsyncIterator
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MQTTDataSource:
    """MQTT data source implementation."""

    def __init__(self, config):
        self.config = config
        self.client = mqtt.Client()
        self._connected = False

    async def connect(self) -> None:
        """Connect to MQTT broker."""
        try:
            # Parse URL
            url_parts = self.config.url.replace("mqtt://", "").split(":")
            host = url_parts[0]
            port = int(url_parts[1]) if len(url_parts) > 1 else 1883

            # Set up callbacks
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect

            # Connect
            self.client.connect(host, port, 60)
            self.client.loop_start()

            # Wait for connection
            while not self._connected:
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        self.client.loop_stop()
        self.client.disconnect()
        self._connected = False

    async def read_data(self) -> AsyncIterator[Dict[str, Any]]:
        """Read data from MQTT topic."""
        # Subscribe to topic if specified in URL
        if "/topic/" in self.config.url:
            topic = self.config.url.split("/topic/")[-1]
            self.client.subscribe(topic)

        while self._connected:
            try:
                # Get data from buffer
                data = await asyncio.wait_for(self._buffer.get(), timeout=1.0)
                if data:
                    yield data
            except asyncio.TimeoutError:
                continue

    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            self._connected = True
            logger.info("Connected to MQTT broker")
        else:
            logger.error(f"Failed to connect to MQTT broker: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback."""
        self._connected = False
        logger.info("Disconnected from MQTT broker")

    def _on_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            # Parse message payload
            payload = msg.payload.decode('utf-8')

            # Try to parse as JSON, fallback to string
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = {"message": payload, "topic": msg.topic}

            # Add metadata
            data["_timestamp"] = asyncio.get_event_loop().time()
            data["_source"] = "mqtt"
            data["_topic"] = msg.topic

            # Put in buffer (non-blocking)
            try:
                self._buffer.put_nowait(data)
            except asyncio.QueueFull:
                logger.warning("MQTT buffer full, dropping message")

        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
