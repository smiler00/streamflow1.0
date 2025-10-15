"""
API Data Source implementation for StreamFlow.
"""

import asyncio
import logging
from typing import Any, Dict, AsyncIterator
import aiohttp
from typing import Optional

logger = logging.getLogger(__name__)


class APIDataSource:
    """HTTP API data source implementation."""

    def __init__(self, config):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self._connected = False
        self._poll_interval = self.config.options.get('poll_interval', 1.0) if self.config.options else 1.0

    async def connect(self) -> None:
        """Connect to API endpoint."""
        try:
            self.session = aiohttp.ClientSession()
            self._connected = True
            logger.info(f"Connected to API: {self.config.url}")
        except Exception as e:
            logger.error(f"Failed to connect to API: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from API."""
        if self.session:
            await self.session.close()
        self._connected = False

    async def read_data(self) -> AsyncIterator[Dict[str, Any]]:
        """Read data from API endpoint."""
        while self._connected:
            try:
                if self.session:
                    async with self.session.get(self.config.url) as response:
                        if response.status == 200:
                            data = await response.json()

                            # Handle both single objects and arrays
                            if isinstance(data, list):
                                for item in data:
                                    yield item
                            else:
                                yield data
                        else:
                            logger.error(f"API request failed with status {response.status}")

                await asyncio.sleep(self._poll_interval)

            except Exception as e:
                logger.error(f"Error reading from API: {e}")
                await asyncio.sleep(self._poll_interval)

    async def is_connected(self) -> bool:
        """Check if API is accessible."""
        if not self.session or not self._connected:
            return False

        try:
            async with self.session.get(self.config.url) as response:
                return response.status == 200
        except Exception:
            return False
