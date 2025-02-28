"""Configuration module for faceroom.

This module provides global configuration settings for the faceroom application,
including runtime-configurable parameters such as the face recognition threshold.
"""

import logging
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

# Global configuration settings
RECOGNITION_THRESHOLD = 0.6  # Default face recognition threshold

def get_recognition_threshold() -> float:
    """Get the current face recognition threshold.
    
    The threshold determines how strictly faces are matched. Lower values are more
    strict (fewer false positives, more false negatives), while higher values are
    more lenient (more false positives, fewer false negatives).
    
    Returns:
        float: The current recognition threshold
    """
    return RECOGNITION_THRESHOLD


def set_recognition_threshold(value: float) -> None:
    """Set the face recognition threshold.
    
    Args:
        value (float): The new threshold value, must be between 0.1 and 1.0
        
    Raises:
        ValueError: If the threshold is outside the valid range
    """
    global RECOGNITION_THRESHOLD
    
    # Validate that the threshold is within a reasonable range
    if 0.1 <= value <= 1.0:
        logger.info(f"Updating recognition threshold from {RECOGNITION_THRESHOLD} to {value}")
        RECOGNITION_THRESHOLD = value
    else:
        logger.error(f"Invalid recognition threshold: {value}. Must be between 0.1 and 1.0")
        raise ValueError("Recognition threshold must be between 0.1 and 1.0")


def get_config_summary() -> Dict[str, Any]:
    """Get a summary of all configuration settings.
    
    Returns:
        Dict[str, Any]: Dictionary containing all configuration values
    """
    return {
        "recognition_threshold": RECOGNITION_THRESHOLD
    }
