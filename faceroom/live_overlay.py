"""Live Video Overlay Module for faceroom.

This module integrates the Camera and Face Recognition modules to create
a real-time face detection and overlay system. It processes frames from
the camera and adds visual indicators for detected faces.
"""

import logging
from typing import Optional, Tuple, List, Dict
import cv2
import numpy as np
from faceroom.camera import capture_frame
from faceroom.face_recognition_module import detect_faces, compare_faces, face_distance
from faceroom.enrollment import get_all_enrollments
from faceroom.analytics import increment_metric

# Configure logging
logger = logging.getLogger(__name__)

# Constants for visualization
BBOX_COLOR = (0, 255, 0)  # Green in BGR
BBOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9  # Increased from 0.6
FONT_COLOR = (0, 0, 0)  # Black in BGR for better contrast on green
FONT_THICKNESS = 2  # Increased from 1
UNKNOWN_LABEL = "Unknown"

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
    5. Matches detected faces against enrolled faces
    6. Overlays bounding boxes and labels (with user IDs if matched) on detected faces
    
    Args:
        device_id (int): Camera device ID (default: 0)
        draw_labels (bool): Whether to draw labels (default: True)
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
        face_locations_small, face_encodings = detect_faces(rgb_small_frame, model=detection_model)
        
        # Track faces detected
        if face_locations_small:
            increment_metric("faces_detected", len(face_locations_small))
        
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
        
        # Match detected faces with enrolled faces
        face_labels = []
        if face_encodings and draw_labels:
            face_labels = match_faces_with_enrollments(face_encodings)
        
        # Draw bounding boxes and labels on original frame
        annotated_frame = draw_overlays(
            frame,
            face_locations,
            face_labels if draw_labels else None
        )
        
        increment_metric("frames_processed", 1)
        
        return annotated_frame
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        increment_metric("detection_errors", 1)
        return None

def match_faces_with_enrollments(face_encodings: List[np.ndarray]) -> List[str]:
    """Match detected face encodings with enrolled faces.
    
    Args:
        face_encodings (List[np.ndarray]): List of face encodings to match
        
    Returns:
        List[str]: List of labels (user IDs or "Unknown") for each face
    """
    # Get all enrolled faces
    enrolled_faces = get_all_enrollments()
    
    # If no enrolled faces, return "Unknown" for all
    if not enrolled_faces:
        return [UNKNOWN_LABEL] * len(face_encodings)
    
    # For each detected face, find the best match
    labels = []
    for face_encoding in face_encodings:
        best_match = None
        best_distance = float('inf')
        
        # Check against each enrolled face
        for user_id, enrolled_encoding in enrolled_faces.items():
            try:
                # Skip if enrolled encoding is None or invalid
                if enrolled_encoding is None or not isinstance(enrolled_encoding, (np.ndarray, list)):
                    logger.warning(f"Invalid encoding for user {user_id}, skipping")
                    continue
                    
                # Skip if face encoding is None or invalid
                if face_encoding is None or not isinstance(face_encoding, (np.ndarray, list)):
                    logger.warning(f"Invalid face encoding to match, skipping")
                    continue
                
                # Calculate face distance - ensure proper numpy arrays
                try:
                    # Note: face_distance(known_encoding, face_encodings) but the library expects
                    # parameters in the opposite order: face_distance(face_encodings, face_to_compare)
                    distances = face_distance(enrolled_encoding, [face_encoding])
                    
                    if len(distances) > 0 and distances[0] < best_distance:
                        # Note: compare_faces(known_encoding, face_encodings) but the library expects
                        # parameters in the opposite order: compare_faces(known_face_encodings, face_encoding_to_check)
                        matches = compare_faces(enrolled_encoding, [face_encoding], tolerance=0.6)  # Set a default tolerance
                        
                        if matches and matches[0]:
                            best_distance = distances[0]
                            best_match = user_id
                except Exception as e:
                    logger.error(f"Error calculating distances for user {user_id}: {str(e)}")
                    continue
            except Exception as e:
                logger.error(f"Error matching face with user {user_id}: {str(e)}")
                continue
        
        # Add the best match or "Unknown"
        labels.append(best_match if best_match else UNKNOWN_LABEL)
    
    return labels

def draw_overlays(
    frame: np.ndarray,
    face_locations: list,
    face_labels: Optional[List[str]] = None
) -> np.ndarray:
    """Draw bounding boxes and labels on detected faces.
    
    Args:
        frame (np.ndarray): The original frame in BGR format
        face_locations (list): List of face locations (top, right, bottom, left)
        face_labels (Optional[List[str]]): Optional list of labels for each face
            If provided, must be the same length as face_locations
    
    Returns:
        np.ndarray: Frame with overlays added
    """
    # Create a copy to avoid modifying the original frame
    annotated_frame = frame.copy()
    
    # If no labels provided, use default
    if face_labels is None:
        face_labels = ["Face Detected"] * len(face_locations)
    
    # Ensure labels and locations have the same length
    if len(face_labels) != len(face_locations):
        logger.warning(f"Number of labels ({len(face_labels)}) doesn't match number of faces ({len(face_locations)})")
        face_labels = ["Face Detected"] * len(face_locations)
    
    for i, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        
        # Draw bounding box
        cv2.rectangle(
            annotated_frame,
            (left, top),
            (right, bottom),
            BBOX_COLOR,
            BBOX_THICKNESS
        )
        
        # Get label for this face
        label = face_labels[i]
        
        # Draw label background
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
