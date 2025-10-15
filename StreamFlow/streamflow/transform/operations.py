"""
Transform module for StreamFlow - provides data transformation operations.
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from collections import deque
import time

logger = logging.getLogger(__name__)


@dataclass
class WindowConfig:
    """Configuration for windowing operations."""
    window_type: str  # "tumbling", "sliding", "session"
    window_size: float  # Window size in seconds
    slide_interval: Optional[float] = None  # For sliding windows


class WindowProcessor:
    """Handles windowing operations for data streams."""

    def __init__(self, config: WindowConfig):
        self.config = config
        self.windows: Dict[str, deque] = {}  # window_id -> deque of data
        self.window_start_times: Dict[str, float] = {}

    def add_data(self, data: Dict[str, Any], timestamp: float) -> List[Dict[str, Any]]:
        """Add data to appropriate windows and return completed windows."""
        completed_windows = []

        if self.config.window_type == "tumbling":
            completed_windows = self._process_tumbling_window(data, timestamp)
        elif self.config.window_type == "sliding":
            completed_windows = self._process_sliding_window(data, timestamp)

        return completed_windows

    def _process_tumbling_window(self, data: Dict[str, Any], timestamp: float) -> List[Dict[str, Any]]:
        """Process tumbling window (non-overlapping fixed-size windows)."""
        completed_windows = []

        # Simple implementation - TODO:in practice would need more sophisticated window management
        window_id = f"window_{int(timestamp // self.config.window_size)}"

        if window_id not in self.windows:
            self.windows[window_id] = deque()
            self.window_start_times[window_id] = timestamp

        self.windows[window_id].append(data)

        # Check if window should be completed
        if timestamp - self.window_start_times[window_id] >= self.config.window_size:
            completed_windows.append({
                "window_data": list(self.windows[window_id]),
                "window_start": self.window_start_times[window_id],
                "window_end": timestamp,
                "window_id": window_id
            })
            del self.windows[window_id]
            del self.window_start_times[window_id]

        return completed_windows

    def _process_sliding_window(self, data: Dict[str, Any], timestamp: float) -> List[Dict[str, Any]]:
        """Process sliding window (overlapping fixed-size windows)."""
        # Simplified implementation
        completed_windows = []

        slide_interval = self.config.slide_interval or self.config.window_size
        window_id = f"window_{int(timestamp // slide_interval)}"

        if window_id not in self.windows:
            self.windows[window_id] = deque()
            self.window_start_times[window_id] = timestamp

        self.windows[window_id].append(data)

        # Check if window should be completed
        if timestamp - self.window_start_times[window_id] >= self.config.window_size:
            completed_windows.append({
                "window_data": list(self.windows[window_id]),
                "window_start": self.window_start_times[window_id],
                "window_end": timestamp,
                "window_id": window_id
            })

        return completed_windows


class AggregationProcessor:
    """Handles aggregation operations on data streams."""

    @staticmethod
    def mean(data: List[Dict[str, Any]], field: str) -> float:
        """Calculate mean of a field across data."""
        values = [item[field] for item in data if field in item and isinstance(item[field], (int, float))]
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def sum(data: List[Dict[str, Any]], field: str) -> float:
        """Calculate sum of a field across data."""
        values = [item[field] for item in data if field in item and isinstance(item[field], (int, float))]
        return sum(values)

    @staticmethod
    def count(data: List[Dict[str, Any]]) -> int:
        """Count items in data."""
        return len(data)

    @staticmethod
    def min(data: List[Dict[str, Any]], field: str) -> float:
        """Find minimum value of a field."""
        values = [item[field] for item in data if field in item and isinstance(item[field], (int, float))]
        return min(values) if values else 0.0

    @staticmethod
    def max(data: List[Dict[str, Any]], field: str) -> float:
        """Find maximum value of a field."""
        values = [item[field] for item in data if field in item and isinstance(item[field], (int, float))]
        return max(values) if values else 0.0


class TransformOperations:
    """Collection of transformation operations for StreamFlow."""

    def __init__(self):
        self.window_processor: Optional[WindowProcessor] = None

    def filter(self, condition: Callable[[Dict[str, Any]], bool]):
        """Create a filter function."""
        def filter_func(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            try:
                return data if condition(data) else None
            except Exception as e:
                logger.error(f"Error in filter: {e}")
                return None
        return filter_func

    def map(self, transform: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Create a map function."""
        def map_func(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            try:
                return transform(data)
            except Exception as e:
                logger.error(f"Error in map: {e}")
                return None
        return map_func

    def window(self, window_type: str = "tumbling", window_size: float = 60.0,
               slide_interval: Optional[float] = None):
        """Create a windowing function."""
        self.window_processor = WindowProcessor(WindowConfig(
            window_type=window_type,
            window_size=window_size,
            slide_interval=slide_interval
        ))

        def window_func(data: Dict[str, Any]) -> List[Dict[str, Any]]:
            if self.window_processor:
                timestamp = data.get("_timestamp", time.time())
                return self.window_processor.add_data(data, timestamp)
            return [data]
        return window_func

    def aggregate(self, operation: str, field: Optional[str] = None):
        """Create an aggregation function."""
        def aggregate_func(window_data: List[Dict[str, Any]]) -> Dict[str, Any]:
            try:
                result = {"operation": operation}

                if operation == "mean" and field:
                    result[field] = AggregationProcessor.mean(window_data, field)
                elif operation == "sum" and field:
                    result[field] = AggregationProcessor.sum(window_data, field)
                elif operation == "count":
                    result["count"] = AggregationProcessor.count(window_data)
                elif operation == "min" and field:
                    result[field] = AggregationProcessor.min(window_data, field)
                elif operation == "max" and field:
                    result[field] = AggregationProcessor.max(window_data, field)
                else:
                    logger.warning(f"Unknown aggregation operation: {operation}")
                    return window_data[0] if window_data else {}

                result["_timestamp"] = time.time()
                return result

            except Exception as e:
                logger.error(f"Error in aggregation: {e}")
                return {}

        return aggregate_func

    def join(self, other_stream_data: List[Dict[str, Any]], key: str,
             join_type: str = "inner"):
        """Create a join function for combining streams."""
        def join_func(data: Dict[str, Any]) -> List[Dict[str, Any]]:
            try:
                if key not in data:
                    return [data]

                results = []
                data_key = data[key]

                for other_data in other_stream_data:
                    if key in other_data and other_data[key] == data_key:
                        if join_type == "inner":
                            # Combine data
                            combined = {**data, **other_data}
                            combined["_join_key"] = key
                            results.append(combined)
                        elif join_type == "left" and data not in results:
                            # Include left data even if no match
                            results.append(data)

                return results if results else [data]

            except Exception as e:
                logger.error(f"Error in join: {e}")
                return [data]

        return join_func

    def group_by(self, key: str):
        """Create a group by function."""
        def group_func(data_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
            try:
                groups = {}
                for data in data_list:
                    if key in data:
                        group_key = data[key]
                        if group_key not in groups:
                            groups[group_key] = []
                        groups[group_key].append(data)
                return groups
            except Exception as e:
                logger.error(f"Error in group_by: {e}")
                return {}

        return group_func

    def distinct(self, field: str):
        """Create a distinct function to remove duplicates based on a field."""
        seen_values = set()

        def distinct_func(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            try:
                if field in data:
                    value = data[field]
                    if value not in seen_values:
                        seen_values.add(value)
                        return data
                return None
            except Exception as e:
                logger.error(f"Error in distinct: {e}")
                return None

        return distinct_func
