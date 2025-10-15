"""
WebSocket Data Source implementation for StreamFlow.
"""

import asyncio
import json
import logging
from typing import Any, Dict, AsyncIterator
import aiohttp
from typing import Optional

logger = logging.getLogger(__name__)


class WebSocketDataSource:
    """WebSocket data source implementation."""

    def __init__(self, config):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket: Optional[aiohttp.ClientWebSocketResponse] = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to WebSocket."""
        try:
            self.session = aiohttp.ClientSession()
            self.websocket = await self.session.ws_connect(self.config.url)
            self._connected = True
            logger.info(f"Connected to WebSocket: {self.config.url}")
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        if self.websocket:
            await self.websocket.close()
        if self.session:
            await self.session.close()
        self._connected = False

    async def read_data(self) -> AsyncIterator[Dict[str, Any]]:
        """Read data from WebSocket."""
        while self._connected and self.websocket:
            try:
                msg = await self.websocket.receive()

                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        data["_timestamp"] = asyncio.get_event_loop().time()
                        data["_source"] = "websocket"
                        yield data
                    except json.JSONDecodeError:
                        # Yield raw text data
                        yield {
                            "message": msg.data,
                            "_timestamp": asyncio.get_event_loop().time(),
                            "_source": "websocket"
                        }

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self.websocket.exception()}")
                    break

            except Exception as e:
                logger.error(f"Error reading from WebSocket: {e}")
                break
