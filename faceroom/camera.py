"""Camera module for capturing frames from the system's camera.

This module provides a simple interface to capture frames from the default camera
using OpenCV. It is designed to be stateless and handle errors gracefully.
"""

import logging
import cv2
import numpy as np
from typing import Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

def capture_frame(device_id: int = 0) -> Optional[Tuple[bool, np.ndarray]]:
    """Capture a single frame from the specified camera.

    This function attempts to capture a frame from the camera. It is designed
    to be stateless, opening and closing the camera for each capture to avoid
    resource leaks.

    Args:
        device_id (int): Camera device ID (default: 0 for built-in camera)

    Returns:
        Optional[Tuple[bool, np.ndarray]]: A tuple containing:
            - bool: True if frame was successfully captured
            - np.ndarray: The captured frame as a NumPy array in BGR format
            Returns None if camera could not be accessed

    Raises:
        None: Exceptions are caught and logged
    """
    try:
        # Initialize camera
        cap = cv2.VideoCapture(device_id)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera (device_id: {device_id})")
            return None
        
        # Read a frame
        ret, frame = cap.read()
        
        if not ret:
            logger.error(f"Failed to capture frame from camera (device_id: {device_id})")
            return None
            
        return ret, frame
        
    except Exception as e:
        logger.error(f"Error capturing frame: {str(e)}")
        return None
        
    finally:
        # Always release the camera
        if 'cap' in locals():
            cap.release()
