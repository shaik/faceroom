"""Unit tests for the camera module."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import cv2
from faceroom.camera import capture_frame, cleanup

@pytest.fixture(autouse=True)
def cleanup_cameras_after_test():
    """Clean up camera resources after each test."""
    yield
    cleanup()

def test_capture_frame_success():
    """Test successful frame capture."""
    # Create a mock frame
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Mock VideoCapture
    with patch('cv2.VideoCapture') as mock_cap:
        # Configure the mock
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, mock_frame)
        mock_cap.return_value = mock_instance
        
        # First call creates persistent camera
        result = capture_frame()
        assert result is not None
        ret, frame = result
        assert ret is True
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
        
        # Second call uses cached camera
        result2 = capture_frame()
        assert result2 is not None
        ret2, frame2 = result2
        assert ret2 is True
        
        # Verify camera was created only once
        assert mock_cap.call_count == 1
        
        # Cleanup should release camera
        cleanup()
        mock_instance.release.assert_called_once()

def test_capture_frame_camera_not_available():
    """Test behavior when camera is not available."""
    with patch('cv2.VideoCapture') as mock_cap:
        # Configure the mock to simulate unavailable camera
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_cap.return_value = mock_instance
        
        # Call the function
        result = capture_frame()
        
        # Verify the result
        assert result is None
        
        # Verify camera was released
        mock_instance.release.assert_called_once()
        
        # Second call should try to create new camera
        result2 = capture_frame()
        assert result2 is None
        
        # Verify second attempt
        assert mock_cap.call_count == 2

def test_capture_frame_read_failure():
    """Test behavior when frame read fails."""
    with patch('cv2.VideoCapture') as mock_cap:
        # Configure the mock
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (False, None)
        mock_cap.return_value = mock_instance
        
        # Call the function
        result = capture_frame()
        
        # Verify the result
        assert result is None
        
        # Failed read should release camera
        mock_instance.release.assert_called_once()
        
        # Second call should try to create new camera
        result2 = capture_frame()
        assert result2 is None
        
        # Verify second attempt
        assert mock_cap.call_count == 2

def test_capture_frame_exception():
    """Test behavior when an exception occurs."""
    with patch('cv2.VideoCapture') as mock_cap:
        # Configure the mock to raise an exception
        mock_cap.side_effect = Exception("Test error")
        
        # Call the function
        result = capture_frame()
        
        # Verify the result
        assert result is None
        
        # Second call should try again
        result2 = capture_frame()
        assert result2 is None
        
        # Verify both attempts
        assert mock_cap.call_count == 2
