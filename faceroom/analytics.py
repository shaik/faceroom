"""Analytics Module for tracking performance metrics.

This module provides functionality to track and retrieve various performance
metrics for the faceroom application, such as frames processed, faces detected,
and error counts.
"""

import logging
import threading
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

# Thread lock for thread-safe operations
_metrics_lock = threading.RLock()

# Global metrics dictionary
METRICS: Dict[str, int] = {
    "frames_processed": 0,
    "faces_detected": 0,
    "detection_errors": 0,
    "recognition_matches": 0,
    "enrollment_count": 0,
    "enrollment_errors": 0,
    "streaming_frames": 0,
    "streaming_errors": 0,
    "streaming_connections": 0
}


class Metrics:
    def __init__(self):
        self._metrics = METRICS

    def __setitem__(self, key: str, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError(f"Expected an integer value for metric '{key}', got {type(value).__name__} instead.")
        with _metrics_lock:
            if key in self._metrics:
                self._metrics[key] = int(value)
                logger.debug(f"Updated metric {key} to {value}")
            else:
                logger.warning(f"Attempted to update unknown metric: {key}")


def increment_metric(name: str, value: int) -> None:
    """Increment the specified metric by the given value."""
    if not name or not isinstance(value, int):
        logger.warning(f"Invalid metric update: {name=}, {value=}")
        return
        
    with _metrics_lock:
        if name in METRICS:
            METRICS[name] += int(value)
            logger.debug(f"Incremented metric {name} by {value}")
        else:
            logger.warning(f"Attempted to increment unknown metric: {name}")


def get_metrics() -> Dict[str, Any]:
    """Get a copy of the current metrics.
    
    Returns:
        Dict[str, Any]: Dictionary containing all current metrics
    """
    with _metrics_lock:
        # Return a copy to prevent external modification
        return METRICS.copy()


def reset_metrics() -> None:
    """Reset all metrics to zero.
    
    This is primarily used for testing or when restarting the application.
    """
    with _metrics_lock:
        for key in METRICS:
            METRICS[key] = 0
        logger.info("All metrics have been reset")


def get_metrics_summary() -> Dict[str, Any]:
    """Get a summary of metrics with additional derived metrics.
    
    Returns:
        Dict[str, Any]: Dictionary containing all current metrics plus derived metrics
    """
    with _metrics_lock:
        metrics = METRICS.copy()
        
        # Add derived metrics
        if metrics["frames_processed"] > 0:
            # Calculate faces per frame and convert to int for type safety
            metrics["faces_per_frame"] = int(round(
                metrics["faces_detected"] / metrics["frames_processed"], 0
            ))
            # Calculate error rate and convert to int for type safety
            metrics["error_rate"] = int(round(
                metrics["detection_errors"] / metrics["frames_processed"], 0
            ))
        else:
            metrics["faces_per_frame"] = 0
            metrics["error_rate"] = 0
            
        return metrics

metrics = Metrics()
