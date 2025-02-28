"""Face Enrollment Module for adding new faces to the recognition database.

This module provides functionality to enroll new faces into the recognition system
by processing images, detecting faces, validating quality, and storing the face encodings
for future recognition.
"""

import logging
import json
import os
import base64
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from threading import RLock

from faceroom.face_recognition_module import detect_faces
from faceroom.analytics import increment_metric

# Configure logging
logger = logging.getLogger(__name__)

# In-memory database of enrolled faces
# Maps user_id to face encoding
_enrolled_faces: Dict[str, np.ndarray] = {}
_db_lock = RLock()

# Default database file path
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'enrolled_faces.json')

# Quality check parameters
MIN_FACE_WIDTH_RATIO = 0.1  # Face width should be at least 10% of image width
MAX_FACE_WIDTH_RATIO = 0.8  # Face width should be at most 80% of image width
CENTER_TOLERANCE = 0.3      # Face center should be within 30% of image center


def _check_face_quality(
    image: np.ndarray,
    face_location: Tuple[int, int, int, int]
) -> Tuple[bool, str]:
    """Check if the detected face meets quality criteria.
    
    Args:
        image (np.ndarray): Input image
        face_location (Tuple[int, int, int, int]): Face location as (top, right, bottom, left)
        
    Returns:
        Tuple[bool, str]: (True if quality check passed, reason if failed)
    """
    # Extract image dimensions
    img_height, img_width = image.shape[:2]
    
    # Extract face dimensions
    top, right, bottom, left = face_location
    face_width = right - left
    face_height = bottom - top
    
    # Check face width ratio
    width_ratio = face_width / img_width
    if width_ratio < MIN_FACE_WIDTH_RATIO:
        return False, f"Face too small: {width_ratio:.2f} < {MIN_FACE_WIDTH_RATIO}"
    
    if width_ratio > MAX_FACE_WIDTH_RATIO:
        return False, f"Face too large: {width_ratio:.2f} > {MAX_FACE_WIDTH_RATIO}"
    
    # Check face centering
    face_center_x = (left + right) / 2
    face_center_y = (top + bottom) / 2
    
    img_center_x = img_width / 2
    img_center_y = img_height / 2
    
    # Calculate normalized distance from center (0 to 1)
    x_offset = abs(face_center_x - img_center_x) / img_width
    y_offset = abs(face_center_y - img_center_y) / img_height
    
    if x_offset > CENTER_TOLERANCE or y_offset > CENTER_TOLERANCE:
        return False, f"Face not centered: x_offset={x_offset:.2f}, y_offset={y_offset:.2f}"
    
    return True, ""


def enroll_face(image: np.ndarray, user_id: str) -> bool:
    """Enroll a face in the recognition database.
    
    This function processes an input image to detect a face, validate its quality,
    extract its encoding, and store it in the recognition database.
    
    Args:
        image (np.ndarray): Input image as a NumPy array in RGB format
        user_id (str): Unique identifier for the user
        
    Returns:
        bool: True if enrollment was successful, False otherwise
        
    Raises:
        ValueError: If the input image is invalid or user_id is empty
    """
    # Input validation
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        logger.error("Invalid image provided for enrollment")
        raise ValueError("Invalid image: must be a non-empty numpy array")
    
    if not user_id or not isinstance(user_id, str):
        logger.error("Invalid user_id provided for enrollment")
        raise ValueError("Invalid user_id: must be a non-empty string")
    
    try:
        # Detect faces in the image
        face_locations, face_encodings = detect_faces(image)
        
        # Check if any faces were detected
        if not face_locations:
            logger.warning(f"No faces detected for user_id: {user_id}")
            increment_metric("enrollment_errors")
            return False
        
        # Use the first face for simplicity
        # In a more advanced implementation, we could select the largest or highest quality face
        face_location = face_locations[0]
        
        # Check face quality
        quality_passed, reason = _check_face_quality(image, face_location)
        if not quality_passed:
            logger.warning(f"Face quality check failed for user_id {user_id}: {reason}")
            increment_metric("enrollment_errors")
            return False
        
        # Get the corresponding encoding
        if not face_encodings:
            logger.warning(f"No face encodings generated for user_id: {user_id}")
            increment_metric("enrollment_errors")
            return False
            
        face_encoding = face_encodings[0]
        
        # Store the encoding in the database
        with _db_lock:
            _enrolled_faces[user_id] = face_encoding
            logger.info(f"Successfully enrolled face for user_id: {user_id}")
            increment_metric("enrollment_count")
        
        # Save to persistent storage
        save_enrollment_database()
        
        return True
        
    except Exception as e:
        logger.error(f"Error enrolling face for user_id {user_id}: {str(e)}")
        increment_metric("enrollment_errors")
        return False


def get_face_encoding(user_id: str) -> Optional[np.ndarray]:
    """Retrieve a face encoding by user ID.
    
    Args:
        user_id (str): Unique identifier for the user
        
    Returns:
        Optional[np.ndarray]: Face encoding if found, None otherwise
    """
    with _db_lock:
        return _enrolled_faces.get(user_id)


def list_enrolled_users() -> List[str]:
    """List all enrolled user IDs.
    
    Returns:
        List[str]: List of enrolled user IDs
    """
    with _db_lock:
        return list(_enrolled_faces.keys())


def remove_enrolled_face(user_id: str) -> bool:
    """Remove an enrolled face from the database.
    
    Args:
        user_id (str): Unique identifier for the user
        
    Returns:
        bool: True if the face was removed, False if it wasn't found
    """
    with _db_lock:
        if user_id in _enrolled_faces:
            del _enrolled_faces[user_id]
            logger.info(f"Removed face for user_id: {user_id}")
            increment_metric("enrollment_count", -1)  # Decrement enrollment count
            save_enrollment_database()
            return True
        else:
            logger.warning(f"Attempted to remove non-existent user_id: {user_id}")
            return False


def _encode_array(arr: np.ndarray) -> str:
    """Encode a numpy array to a base64 string for JSON serialization.
    
    Args:
        arr (np.ndarray): Numpy array to encode
        
    Returns:
        str: Base64 encoded string
    """
    return base64.b64encode(arr.tobytes()).decode('ascii')


def _decode_array(encoded: str) -> np.ndarray:
    """Decode a base64 string back to a numpy array.
    
    Args:
        encoded (str): Base64 encoded string
        
    Returns:
        np.ndarray: Decoded numpy array
    """
    decoded = base64.b64decode(encoded)
    # Face encodings from face_recognition are 128-dimensional float64 arrays
    return np.frombuffer(decoded, dtype=np.float64)


def save_enrollment_database(file_path: str = DEFAULT_DB_PATH) -> bool:
    """Save the enrollment database to a JSON file.
    
    Args:
        file_path (str): Path to save the database file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convert numpy arrays to serializable format
        serializable_db = {}
        with _db_lock:
            for user_id, encoding in _enrolled_faces.items():
                serializable_db[user_id] = {
                    'encoding': _encode_array(encoding),
                    'shape': encoding.shape,
                    'dtype': str(encoding.dtype)
                }
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(serializable_db, f, indent=2)
            
        logger.info(f"Saved enrollment database to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving enrollment database: {str(e)}")
        return False


def load_enrollment_database(file_path: str = DEFAULT_DB_PATH) -> bool:
    """Load the enrollment database from a JSON file.
    
    Args:
        file_path (str): Path to the database file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            logger.info(f"No enrollment database found at {file_path}")
            return False
            
        # Read from file
        with open(file_path, 'r') as f:
            serialized_db = json.load(f)
        
        # Convert serialized data back to numpy arrays
        with _db_lock:
            _enrolled_faces.clear()
            for user_id, data in serialized_db.items():
                encoding = _decode_array(data['encoding'])
                _enrolled_faces[user_id] = encoding
                
        logger.info(f"Loaded enrollment database from {file_path} with {len(_enrolled_faces)} entries")
        return True
        
    except Exception as e:
        logger.error(f"Error loading enrollment database: {str(e)}")
        return False


# Try to load the database on module import
try:
    load_enrollment_database()
except Exception as e:
    logger.warning(f"Failed to load enrollment database on startup: {str(e)}")
