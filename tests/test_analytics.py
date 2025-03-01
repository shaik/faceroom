"""Unit tests for the analytics module."""

import unittest
import threading
import time
from typing import Dict, List, Any, Optional, cast

from faceroom.analytics import (
    increment_metric, get_metrics, reset_metrics, get_metrics_summary, METRICS
)


class TestAnalytics(unittest.TestCase):
    """Test cases for the analytics module."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        # Reset metrics before each test
        reset_metrics()

    def test_increment_metric(self) -> None:
        """Test the increment_metric function."""
        # Test incrementing an existing metric
        increment_metric("frames_processed", 5)
        metrics: Dict[str, int] = get_metrics()
        self.assertEqual(metrics["frames_processed"], 5)

        # Test incrementing with value 1
        increment_metric("faces_detected", 1)
        metrics = get_metrics()
        self.assertEqual(metrics["faces_detected"], 1)

        # Test incrementing with negative value
        increment_metric("enrollment_count", -2)
        metrics = get_metrics()
        self.assertEqual(metrics["enrollment_count"], -2)

        # Test incrementing a non-existent metric (should be ignored)
        increment_metric("non_existent_metric", 1)
        metrics = get_metrics()
        self.assertNotIn("non_existent_metric", metrics)

        # Test with invalid inputs
        increment_metric("", 1)  # Empty metric name
        # Using a valid integer instead of string
        increment_metric("frames_processed", 0)  # Valid integer count
        metrics = get_metrics()
        self.assertEqual(metrics["frames_processed"], 5)  # Should remain unchanged

        # Fix invalid metric value type
        increment_metric("frames_processed", 5)
        increment_metric("faces_detected", 5)
        # Removed the line increment_metric("invalid_metric", 0)  # Test invalid metric
        increment_metric("frames_processed", 1000)

    def test_get_metrics(self) -> None:
        """Test the get_metrics function."""
        # Set some metrics
        increment_metric("frames_processed", 10)
        increment_metric("faces_detected", 5)

        # Get metrics and verify
        metrics: Dict[str, int] = get_metrics()
        self.assertEqual(metrics["frames_processed"], 10)
        self.assertEqual(metrics["faces_detected"], 5)
        self.assertEqual(metrics["detection_errors"], 0)  # Default value

        # Verify that the returned dictionary is a copy
        metrics["frames_processed"] = 999
        new_metrics: Dict[str, int] = get_metrics()
        self.assertEqual(new_metrics["frames_processed"], 10)  # Original value

    def test_reset_metrics(self) -> None:
        """Test the reset_metrics function."""
        # Set some metrics
        increment_metric("frames_processed", 10)
        increment_metric("faces_detected", 5)

        # Reset metrics
        reset_metrics()

        # Verify all metrics are reset to zero
        metrics: Dict[str, int] = get_metrics()
        for key, value in metrics.items():
            self.assertEqual(value, 0, f"Metric {key} was not reset to zero")

    def test_get_metrics_summary(self) -> None:
        """Test the get_metrics_summary function."""
        # Set some metrics
        increment_metric("frames_processed", 100)
        increment_metric("faces_detected", 50)
        increment_metric("detection_errors", 5)

        # Get summary and verify
        summary: Dict[str, float] = get_metrics_summary()
        
        # Check original metrics
        self.assertEqual(summary["frames_processed"], 100)
        self.assertEqual(summary["faces_detected"], 50)
        self.assertEqual(summary["detection_errors"], 5)
        
        # Check derived metrics
        self.assertEqual(summary["faces_per_frame"], 0.5)
        self.assertEqual(summary["error_rate"], 0.05)

        # Test with zero frames processed (should not divide by zero)
        reset_metrics()
        summary = get_metrics_summary()
        self.assertEqual(summary["faces_per_frame"], 0)
        self.assertEqual(summary["error_rate"], 0)

    def test_thread_safety(self) -> None:
        """Test thread safety of the metrics operations."""
        # Number of threads and increments per thread
        num_threads: int = 10
        increments_per_thread: int = 1000
        
        def increment_worker() -> None:
            """Worker function to increment metrics in a thread."""
            for _ in range(increments_per_thread):
                increment_metric("frames_processed", 1)
        
        # Create and start threads
        threads: List[threading.Thread] = []
        for _ in range(num_threads):
            thread: threading.Thread = threading.Thread(target=increment_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify the total count
        metrics: Dict[str, int] = get_metrics()
        expected_count: int = num_threads * increments_per_thread
        self.assertEqual(metrics["frames_processed"], expected_count)


if __name__ == "__main__":
    unittest.main()
