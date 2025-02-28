"""Camera module for capturing frames from the system's camera.

This module provides a simple interface to capture frames from the default camera
using OpenCV. It maintains persistent camera connections and handles macOS
authorization requirements.
"""

import logging
import os
import cv2
import numpy as np
from typing import Optional, Tuple, Dict
from threading import Lock

# Configure logging
logger = logging.getLogger(__name__)

# Configure OpenCV to skip macOS authorization dialog
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

# Global camera cache and locks
_cameras: Dict[int, cv2.VideoCapture] = {}
_locks: Dict[int, Lock] = {}

def _get_camera(device_id: int) -> Optional[cv2.VideoCapture]:
    """Get or create a camera instance.
    
    Maintains a cache of camera connections to avoid repeatedly
    opening and closing the camera.
    
    Args:
        device_id (int): Camera device ID
    
    Returns:
        Optional[cv2.VideoCapture]: Camera instance or None if failed
    """
    # Create lock for this camera if it doesn't exist
    if device_id not in _locks:
        _locks[device_id] = Lock()
    
    with _locks[device_id]:
        # Return existing camera if it's working
        if device_id in _cameras:
            cap = _cameras[device_id]
            if cap.isOpened():
                return cap
            # Clean up failed camera
            cap.release()
            del _cameras[device_id]
        
        # Create new camera
        try:
            cap = cv2.VideoCapture(device_id)
            if not cap.isOpened():
                logger.error(f"Failed to open camera (device_id: {device_id})")
                return None
            
            # Configure camera for better performance
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
            cap.set(cv2.CAP_PROP_FPS, 30)  # Target 30 FPS
            
            _cameras[device_id] = cap
            return cap
            
        except Exception as e:
            logger.error(f"Error creating camera: {str(e)}")
            return None

def capture_frame(device_id: int = 0) -> Optional[Tuple[bool, np.ndarray]]:
    """Capture a single frame from the specified camera.

    This function maintains a persistent connection to the camera and
    captures frames efficiently.

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
        # Get camera instance
        cap = _get_camera(device_id)
        if cap is None:
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

def cleanup():
    """Release all camera resources.
    
    This should be called when shutting down the application.
    """
    for device_id, lock in _locks.items():
        with lock:
            if device_id in _cameras:
                _cameras[device_id].release()
    
    _cameras.clear()
