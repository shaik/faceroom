"""Live Video Overlay Module for faceroom.

This module integrates the Camera and Face Recognition modules to create
a real-time face detection and overlay system. It processes frames from
the camera and adds visual indicators for detected faces.
"""

import logging
from typing import Optional, Tuple
import cv2
import numpy as np
from faceroom.camera import capture_frame
from faceroom.face_recognition_module import detect_faces

# Configure logging
logger = logging.getLogger(__name__)

# Constants for visualization
BBOX_COLOR = (0, 255, 0)  # Green in BGR
BBOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.6
FONT_COLOR = (255, 255, 255)  # White in BGR
FONT_THICKNESS = 1

def process_frame_and_overlay(
    device_id: int = 0,
    draw_labels: bool = True,
    detection_model: str = "hog",
    scale_factor: float = 0.5
) -> Optional[np.ndarray]:
    """Process a camera frame and overlay face detection results.
    
    This function:
    1. Captures a frame using @func:capture_frame
    2. Optionally downscales the frame for faster face detection
    3. Detects faces using @func:detect_faces
    4. Scales face locations back to original size
    5. Overlays bounding boxes and labels on detected faces
    
    Args:
        device_id (int): Camera device ID (default: 0)
        draw_labels (bool): Whether to draw "Face Detected" labels (default: True)
        detection_model (str): Face detection model to use (default: "hog")
        scale_factor (float): Factor to downscale frame for detection (default: 0.5)
                            Must be between 0 and 1, where 1 means no scaling
    
    Returns:
        Optional[np.ndarray]: The processed frame with overlays, or None if processing fails
    
    Note:
        Face detection is performed on a downscaled version of the frame for better
        performance, while overlays are drawn on the original resolution frame.
    """
    try:
        # Validate scale factor
        if not 0 < scale_factor <= 1:
            logger.error(f"Invalid scale factor: {scale_factor}. Must be between 0 and 1")
            return None
        
        # Capture frame
        capture_result = capture_frame(device_id)
        if not capture_result:
            logger.error("Failed to capture frame")
            return None
            
        ret, frame = capture_result
        if not ret or frame is None:
            logger.error("Invalid frame captured")
            return None
            
        # Get original dimensions
        original_height, original_width = frame.shape[:2]
        
        # Create downscaled frame for detection if needed
        if scale_factor < 1:
            logger.debug(f"Downscaling frame by factor {scale_factor} for detection")
            small_frame = cv2.resize(
                frame,
                (int(original_width * scale_factor), int(original_height * scale_factor))
            )
        else:
            small_frame = frame
        
        # Convert to RGB for face detection
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces on downscaled frame
        face_locations_small, _ = detect_faces(rgb_small_frame, model=detection_model)
        
        # Scale face locations back to original size if needed
        if scale_factor < 1:
            face_locations = [
                (
                    int(top / scale_factor),
                    int(right / scale_factor),
                    int(bottom / scale_factor),
                    int(left / scale_factor)
                )
                for (top, right, bottom, left) in face_locations_small
            ]
        else:
            face_locations = face_locations_small
        
        # Draw bounding boxes and labels on original frame
        annotated_frame = draw_overlays(
            frame,
            face_locations,
            draw_labels
        )
        
        return annotated_frame
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return None

def draw_overlays(
    frame: np.ndarray,
    face_locations: list,
    draw_labels: bool = True
) -> np.ndarray:
    """Draw bounding boxes and labels on detected faces.
    
    Args:
        frame (np.ndarray): The original frame in BGR format
        face_locations (list): List of face locations (top, right, bottom, left)
        draw_labels (bool): Whether to draw labels above boxes
    
    Returns:
        np.ndarray: Frame with overlays added
    """
    # Create a copy to avoid modifying the original frame
    annotated_frame = frame.copy()
    
    for face_location in face_locations:
        top, right, bottom, left = face_location
        
        # Draw bounding box
        cv2.rectangle(
            annotated_frame,
            (left, top),
            (right, bottom),
            BBOX_COLOR,
            BBOX_THICKNESS
        )
        
        if draw_labels:
            # Draw label background
            label = "Face Detected"
            label_size = cv2.getTextSize(
                label,
                FONT,
                FONT_SCALE,
                FONT_THICKNESS
            )[0]
            
            cv2.rectangle(
                annotated_frame,
                (left, top - label_size[1] - 10),
                (left + label_size[0], top),
                BBOX_COLOR,
                cv2.FILLED
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (left, top - 5),
                FONT,
                FONT_SCALE,
                FONT_COLOR,
                FONT_THICKNESS
            )
    
    return annotated_frame
