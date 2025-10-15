"""
Redis Data Source implementation for StreamFlow.
"""

import asyncio
import logging
from typing import Any, Dict, AsyncIterator
import redis.asyncio as redis
from typing import Optional

logger = logging.getLogger(__name__)


class RedisDataSource:
    """Redis data source implementation."""

    def __init__(self, config):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            # Parse URL
            url_parts = self.config.url.replace("redis://", "").split(":")
            host = url_parts[0]
            port = int(url_parts[1]) if len(url_parts) > 1 else 6379

            self.redis_client = redis.Redis(host=host, port=port, decode_responses=True)
            # Test connection
            await self.redis_client.ping()
            self._connected = True
            logger.info(f"Connected to Redis: {self.config.url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
        self._connected = False

    async def read_data(self) -> AsyncIterator[Dict[str, Any]]:
        """Read data from Redis streams."""
        # For now, implement as a simple key-value reader
        # In a full implementation, would use Redis streams
        while self._connected:
            try:
                # This is a simplified implementation
                # A full implementation would use Redis streams (XREAD, etc.)
                await asyncio.sleep(1.0)

                # Placeholder - would read from Redis stream here
                # For now, just yield a test message
                yield {
                    "message": "Redis data stream",
                    "_timestamp": asyncio.get_event_loop().time(),
                    "_source": "redis"
                }

            except Exception as e:
                logger.error(f"Error reading from Redis: {e}")
                await asyncio.sleep(1.0)
