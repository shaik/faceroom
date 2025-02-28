"""Unit tests for the live video overlay module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import cv2
from faceroom.live_overlay import process_frame_and_overlay, draw_overlays

# Test data
FRAME_HEIGHT = 480
FRAME_WIDTH = 640
MOCK_FRAME = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
MOCK_FACE_LOCATIONS = [(100, 200, 150, 50)]  # top, right, bottom, left

def test_process_frame_no_faces():
    """Test processing a frame with no faces detected."""
    with patch('faceroom.live_overlay.capture_frame') as mock_capture, \
         patch('faceroom.live_overlay.detect_faces') as mock_detect:
        # Configure mocks
        mock_capture.return_value = (True, MOCK_FRAME)
        mock_detect.return_value = ([], [])
        
        # Process frame
        result = process_frame_and_overlay()
        
        # Verify result
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == MOCK_FRAME.shape

def test_process_frame_with_faces():
    """Test processing a frame with faces detected."""
    with patch('faceroom.live_overlay.capture_frame') as mock_capture, \
         patch('faceroom.live_overlay.detect_faces') as mock_detect:
        # Configure mocks
        mock_capture.return_value = (True, MOCK_FRAME)
        mock_detect.return_value = (MOCK_FACE_LOCATIONS, [])
        
        # Process frame
        result = process_frame_and_overlay()
        
        # Verify result
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == MOCK_FRAME.shape
        
        # Convert to grayscale to check if any drawing occurred
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        assert np.any(gray > 0)  # Should have some non-black pixels from drawing

def test_process_frame_capture_failure():
    """Test behavior when frame capture fails."""
    with patch('faceroom.live_overlay.capture_frame') as mock_capture:
        # Configure mock to simulate capture failure
        mock_capture.return_value = None
        
        # Process frame
        result = process_frame_and_overlay()
        
        # Verify result
        assert result is None

def test_process_frame_detection_error():
    """Test behavior when face detection raises an error."""
    with patch('faceroom.live_overlay.capture_frame') as mock_capture, \
         patch('faceroom.live_overlay.detect_faces') as mock_detect:
        # Configure mocks
        mock_capture.return_value = (True, MOCK_FRAME)
        mock_detect.side_effect = Exception("Test error")
        
        # Process frame
        result = process_frame_and_overlay()
        
        # Verify result
        assert result is None

def test_draw_overlays():
    """Test drawing overlays on a frame."""
    # Create a test frame
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    
    # Draw overlays
    result = draw_overlays(frame, MOCK_FACE_LOCATIONS)
    
    # Verify result
    assert isinstance(result, np.ndarray)
    assert result.shape == frame.shape
    assert not np.array_equal(result, frame)  # Should be modified
    
    # Check if bounding box was drawn
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    assert np.any(gray > 0)  # Should have some non-black pixels

def test_draw_overlays_no_labels():
    """Test drawing overlays without labels."""
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    
    # Draw overlays without labels
    result = draw_overlays(frame, MOCK_FACE_LOCATIONS, draw_labels=False)
    
    # Verify result
    assert isinstance(result, np.ndarray)
    assert result.shape == frame.shape
    assert not np.array_equal(result, frame)  # Should be modified

def test_draw_overlays_empty_locations():
    """Test drawing overlays with no face locations."""
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    
    # Draw overlays with no faces
    result = draw_overlays(frame, [])
    
    # Verify result
    assert isinstance(result, np.ndarray)
    assert result.shape == frame.shape
    assert np.array_equal(result, frame)  # Should be unchanged
