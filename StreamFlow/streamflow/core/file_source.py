"""
File Data Source implementation for StreamFlow.
"""

import asyncio
import json
import logging
from typing import Any, Dict, AsyncIterator
import aiofiles

logger = logging.getLogger(__name__)


class FileDataSource:
    """File data source implementation."""

    def __init__(self, config):
        self.config = config
        self._connected = False
        self._file_path = self.config.url.replace("file://", "")

    async def connect(self) -> None:
        """Connect to file (check if exists)."""
        try:
            # Check if file exists
            import os
            if not os.path.exists(self._file_path):
                raise FileNotFoundError(f"File not found: {self._file_path}")

            self._connected = True
            logger.info(f"Connected to file: {self._file_path}")
        except Exception as e:
            logger.error(f"Failed to connect to file: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from file."""
        self._connected = False

    async def read_data(self) -> AsyncIterator[Dict[str, Any]]:
        """Read data from file line by line."""
        try:
            async with aiofiles.open(self._file_path, 'r') as file:
                async for line in file:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            # Yield raw line data
                            data = {
                                "message": line,
                                "_source": "file",
                                "_file_path": self._file_path
                            }

                        data["_timestamp"] = asyncio.get_event_loop().time()
                        yield data

        except Exception as e:
            logger.error(f"Error reading from file: {e}")

    async def is_connected(self) -> bool:
        """Check if file is accessible."""
        import os
        return os.path.exists(self._file_path)
