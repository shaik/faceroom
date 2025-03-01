"""Face Recognition Module for detecting and encoding faces in images.

This module provides functionality to detect faces in images and generate their
encodings using the face_recognition library. It is designed to be stateless
and handle errors gracefully.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import face_recognition
from faceroom.config import get_recognition_threshold
from faceroom.analytics import increment_metric

# Configure logging
logger = logging.getLogger(__name__)

def detect_faces(
    image: np.ndarray,
    model: str = "hog"
) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
    """Detect faces in an image and generate their encodings.
    
    This function processes an input image to:
    1. Detect face locations (bounding boxes)
    2. Generate face encodings for each detected face
    
    The function is stateless and handles errors gracefully, returning empty
    lists if no faces are detected or if processing fails.
    
    Args:
        image (np.ndarray): Input image as a NumPy array in RGB format
        model (str): Face detection model to use, either 'hog' (CPU) or 
                    'cnn' (GPU). Defaults to 'hog'.
    
    Returns:
        Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]: A tuple containing:
            - List of face locations, each as (top, right, bottom, left)
            - List of corresponding face encodings as numpy arrays
            
    Raises:
        ValueError: If the input image is invalid (None, empty, or wrong type)
    """
    # Input validation
    if image is None:
        raise ValueError("Input image cannot be None")
    
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a numpy array")
        
    if image.size == 0:
        raise ValueError("Input image cannot be empty")
    
    try:
        # Detect face locations
        face_locations = face_recognition.face_locations(
            image,
            model=model
        )
        
        if not face_locations:
            logger.info("No faces detected in the image")
            return [], []
            
        # Generate face encodings
        face_encodings = face_recognition.face_encodings(
            image,
            face_locations
        )
        
        if not face_encodings:
            logger.warning("Failed to generate encodings for detected faces")
            return face_locations, []
            
        return face_locations, face_encodings
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return [], []

def compare_faces(
    known_encoding: np.ndarray,
    face_encodings: List[np.ndarray],
    tolerance: float = 0.6
) -> List[bool]:
    """Compare a known face encoding against a list of face encodings.
    
    Args:
        known_encoding (np.ndarray): The known face encoding to compare against
        face_encodings (List[np.ndarray]): List of face encodings to check
        tolerance (float, optional): How much distance between faces to consider it a match.
                           Lower is more strict. Defaults to 0.6.
    
    Returns:
        List[bool]: List of True/False values indicating which encodings match
    """
    # Ensure the known encoding is a numpy array with proper dtype
    known_encoding = np.asarray(known_encoding, dtype=np.float64)
    
    # Convert each encoding in face_encodings to a numpy array if it isn't already
    face_encodings_array = []
    for fe in face_encodings:
        if fe is None:
            continue
        face_encodings_array.append(np.asarray(fe, dtype=np.float64))
    
    # If no valid encodings, return all False
    if not face_encodings_array:
        return [False] * len(face_encodings)
    
    # Call the face_recognition library function
    valid_matches = face_recognition.compare_faces(face_encodings_array, known_encoding, tolerance=tolerance)
    return valid_matches

def face_distance(
    known_encoding: np.ndarray,
    face_encodings: List[np.ndarray]
) -> np.ndarray:
    """Calculate face distance between encodings.
    
    Args:
        known_encoding (np.ndarray): The known face encoding
        face_encodings (List[np.ndarray]): List of face encodings to check
    
    Returns:
        np.ndarray: Array of face distances for each encoding
    """
    try:
        # Ensure the known encoding is a numpy array with proper dtype
        known_encoding = np.asarray(known_encoding, dtype=np.float64)
        
        # Convert each encoding in face_encodings to a numpy array if it isn't already
        face_encodings_array = []
        for fe in face_encodings:
            # Skip None values
            if fe is None:
                continue
                
            # Convert to numpy array with proper dtype
            try:
                face_encodings_array.append(np.asarray(fe, dtype=np.float64))
            except Exception as e:
                logger.warning(f"Could not convert face encoding to numpy array: {str(e)}")
                continue
        
        # If no valid encodings, return an empty array
        if not face_encodings_array:
            return np.array([float('inf')] * len(face_encodings))
        
        # IMPORTANT: The face_recognition library expects parameters in the opposite order:
        # face_recognition.face_distance(face_encodings, face_to_compare)
        # So we need to swap our parameters compared to our function signature
        return face_recognition.face_distance(
            face_encodings_array,  # List of face encodings to compare
            known_encoding         # A face encoding to compare against
        )
    except Exception as e:
        logger.error(f"Error calculating face distances: {str(e)}")
        return np.array([float('inf')] * len(face_encodings))
