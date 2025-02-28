"""Unit tests for the live video streaming functionality."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from faceroom.streaming import generate_frames
from faceroom.app import app

@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    with app.test_client() as client:
        yield client

def test_live_feed_endpoint_default_camera(client):
    """Test that the /live endpoint returns the correct response type with default camera."""
    response = client.get('/live')
    assert response.status_code == 200
    assert response.content_type == 'multipart/x-mixed-replace; boundary=frame'

def test_live_feed_endpoint_specific_camera(client):
    """Test that the /live endpoint accepts a camera parameter."""
    response = client.get('/live?camera=1')
    assert response.status_code == 200
    assert response.content_type == 'multipart/x-mixed-replace; boundary=frame'

def test_live_feed_endpoint_invalid_camera(client):
    """Test that the /live endpoint handles invalid camera IDs."""
    response = client.get('/live?camera=invalid')
    assert response.status_code == 400
    assert b'Invalid camera ID' in response.data

def test_generate_frames():
    """Test frame generation with mocked dependencies."""
    # Create a mock frame
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    with patch('faceroom.streaming.process_frame_and_overlay') as mock_process:
        # Configure mock to return one frame then raise StopIteration
        mock_process.return_value = mock_frame
        
        # Get the first frame from the generator
        generator = generate_frames()
        frame_data = next(generator)
        
        # Verify frame format
        assert frame_data.startswith(b'--frame\r\n')
        assert b'Content-Type: image/jpeg\r\n\r\n' in frame_data
        assert frame_data.endswith(b'\r\n')
        
        # Verify JPEG data is present
        jpeg_start = frame_data.find(b'\xff\xd8')
        jpeg_end = frame_data.find(b'\xff\xd9')
        assert jpeg_start != -1 and jpeg_end != -1
        assert jpeg_end > jpeg_start

def test_generate_frames_error_handling():
    """Test error handling in frame generation."""
    with patch('faceroom.streaming.process_frame_and_overlay') as mock_process:
        # Configure mock to simulate an error
        mock_process.return_value = None
        
        # Get the first frame from the generator
        generator = generate_frames()
        
        try:
            # Get first frame (should be None due to error)
            frame = next(generator)
            
            # Verify frame is properly formatted despite error
            assert frame.startswith(b'--frame\r\n')
            assert b'Content-Type: image/jpeg\r\n\r\n' in frame
            assert frame.endswith(b'\r\n')
            
        finally:
            # Clean up the stream
            from faceroom.streaming import cleanup
            cleanup()

def test_dashboard_with_video(client):
    """Test that the dashboard page includes the video feed."""
    response = client.get('/dashboard')
    assert response.status_code == 200
    assert b'Live Camera Feed' in response.data
    assert b'video-feed' in response.data
    assert b'url_for(\'live_feed\')' in response.data
