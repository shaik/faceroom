"""Sound Recognition Demo

This script demonstrates the face recognition with sound playback functionality.
It captures video from the camera, recognizes faces, and plays sounds when
recognized faces appear.

Usage:
    python -m examples.sound_recognition_demo

Press 'q' to quit the demo.
"""

import os
import time
import logging
import cv2
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import faceroom modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faceroom.camera import capture_frame
from faceroom.face_recognition_module import detect_faces
from faceroom.enrollment import get_all_enrollments
from faceroom.sound_player import (
    play_sound_for_user, 
    set_cooldown, 
    reset_all_cooldowns,
    cleanup as cleanup_sound
)

# Constants
WINDOW_NAME = "Sound Recognition Demo"
UNKNOWN_LABEL = "Unknown"
BBOX_COLOR = (0, 255, 0)  # Green in BGR
TEXT_COLOR = (0, 0, 0)    # Black in BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
COOLDOWN_SECONDS = 10     # Shorter cooldown for demo purposes


def match_face_with_enrollments(face_encoding, enrolled_faces):
    """Match a face encoding with enrolled faces.
    
    Args:
        face_encoding: Face encoding to match
        enrolled_faces: Dictionary of enrolled faces (user_id -> encoding)
        
    Returns:
        str: User ID of the best match, or "Unknown" if no match found
    """
    from faceroom.face_recognition_module import face_distance, compare_faces
    
    if not enrolled_faces:
        return UNKNOWN_LABEL
        
    best_match = None
    best_distance = float('inf')
    
    for user_id, enrolled_encoding in enrolled_faces.items():
        try:
            # Calculate face distance
            distances = face_distance(enrolled_encoding, [face_encoding])
            
            if len(distances) > 0 and distances[0] < best_distance:
                matches = compare_faces(enrolled_encoding, [face_encoding], tolerance=0.6)
                
                if matches and matches[0]:
                    best_distance = distances[0]
                    best_match = user_id
        except Exception as e:
            logger.error(f"Error matching face with user {user_id}: {str(e)}")
            continue
    
    return best_match if best_match else UNKNOWN_LABEL


def draw_face_box(frame, face_location, label):
    """Draw a bounding box and label for a face.
    
    Args:
        frame: Video frame to draw on
        face_location: Tuple of (top, right, bottom, left)
        label: Label to display
        
    Returns:
        frame: Frame with bounding box and label
    """
    top, right, bottom, left = face_location
    
    # Draw bounding box
    cv2.rectangle(frame, (left, top), (right, bottom), BBOX_COLOR, 2)
    
    # Draw label background
    label_size, _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    label_width, label_height = label_size
    
    cv2.rectangle(
        frame,
        (left, top - label_height - 10),
        (left + label_width + 10, top),
        BBOX_COLOR,
        cv2.FILLED
    )
    
    # Draw label text
    cv2.putText(
        frame,
        label,
        (left + 5, top - 5),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS
    )
    
    return frame


def main():
    """Run the sound recognition demo."""
    logger.info("Starting Sound Recognition Demo")
    
    # Create output window
    cv2.namedWindow(WINDOW_NAME)
    
    # Set cooldown period for demo
    set_cooldown(COOLDOWN_SECONDS)
    logger.info(f"Sound cooldown set to {COOLDOWN_SECONDS} seconds")
    
    # Reset all cooldowns at start
    reset_all_cooldowns()
    
    # Get enrolled faces
    enrolled_faces = get_all_enrollments()
    if enrolled_faces:
        logger.info(f"Loaded {len(enrolled_faces)} enrolled faces")
    else:
        logger.warning("No enrolled faces found. Run the enrollment process first.")
    
    try:
        while True:
            # Capture frame
            capture_result = capture_frame(0)
            if not capture_result:
                logger.error("Failed to capture frame")
                time.sleep(0.1)
                continue
                
            ret, frame = capture_result
            if not ret or frame is None:
                logger.error("Invalid frame captured")
                time.sleep(0.1)
                continue
            
            # Convert to RGB for face detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations, face_encodings = detect_faces(rgb_frame)
            
            # Process each detected face
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                # Match with enrolled faces
                user_id = match_face_with_enrollments(face_encoding, enrolled_faces)
                
                # Draw bounding box and label
                frame = draw_face_box(frame, face_location, user_id)
                
                # Play sound if recognized face
                if user_id != UNKNOWN_LABEL:
                    if play_sound_for_user(user_id):
                        logger.info(f"Playing sound for {user_id}")
            
            # Display frame
            cv2.imshow(WINDOW_NAME, frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")
    finally:
        # Clean up
        cv2.destroyAllWindows()
        cleanup_sound()
        logger.info("Demo ended")


if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    # Run demo
    main()
