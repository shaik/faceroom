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
    "enrollment_errors": 0
}


def increment_metric(metric_name: str, count: int = 1) -> None:
    """Increment a specific metric counter.
    
    Args:
        metric_name (str): Name of the metric to increment
        count (int, optional): Amount to increment by. Defaults to 1.
    """
    if not metric_name or not isinstance(count, int):
        logger.warning(f"Invalid metric update: {metric_name=}, {count=}")
        return
        
    with _metrics_lock:
        if metric_name in METRICS:
            METRICS[metric_name] += count
            logger.debug(f"Incremented metric {metric_name} by {count}")
        else:
            logger.warning(f"Attempted to increment unknown metric: {metric_name}")


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
            metrics["faces_per_frame"] = round(
                metrics["faces_detected"] / metrics["frames_processed"], 4
            )
            metrics["error_rate"] = round(
                metrics["detection_errors"] / metrics["frames_processed"], 4
            )
        else:
            metrics["faces_per_frame"] = 0
            metrics["error_rate"] = 0
            
        return metrics
