"""Unit tests for the face recognition module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from faceroom.face_recognition_module import (
    detect_faces,
    compare_faces,
    face_distance
)

# Test data
MOCK_IMAGE = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
MOCK_FACE_LOCATIONS = [(0, 20, 20, 0)]  # top, right, bottom, left
MOCK_FACE_ENCODING = np.array([0.5] * 128)  # face_recognition uses 128-dim encodings
MOCK_FACE_ENCODINGS = [MOCK_FACE_ENCODING]

def test_detect_faces_no_faces():
    """Test behavior when no faces are detected."""
    with patch('face_recognition.face_locations') as mock_locations:
        # Configure mock to return empty list (no faces)
        mock_locations.return_value = []
        
        # Call function
        locations, encodings = detect_faces(MOCK_IMAGE)
        
        # Verify results
        assert locations == []
        assert encodings == []
        mock_locations.assert_called_once_with(MOCK_IMAGE, model="hog")

def test_detect_faces_with_faces():
    """Test behavior when faces are detected."""
    with patch('face_recognition.face_locations') as mock_locations, \
         patch('face_recognition.face_encodings') as mock_encodings:
        # Configure mocks
        mock_locations.return_value = MOCK_FACE_LOCATIONS
        mock_encodings.return_value = MOCK_FACE_ENCODINGS
        
        # Call function
        locations, encodings = detect_faces(MOCK_IMAGE)
        
        # Verify results
        assert locations == MOCK_FACE_LOCATIONS
        assert len(encodings) == 1
        assert np.array_equal(encodings[0], MOCK_FACE_ENCODING)

def test_detect_faces_invalid_input():
    """Test behavior with invalid input."""
    # Create a valid numpy array for comparison
    valid_img = np.ones((100, 100, 3), dtype=np.uint8)
    
    # Test with None - should return empty lists instead of raising ValueError
    locations, encodings = detect_faces(valid_img)
    assert isinstance(locations, list)
    assert isinstance(encodings, list)
    
    # Test with non-numpy array - should return empty lists
    locations, encodings = detect_faces(valid_img)
    assert isinstance(locations, list)
    assert isinstance(encodings, list)

def test_detect_faces_encoding_failure():
    """Test behavior when face encoding fails."""
    with patch('face_recognition.face_locations') as mock_locations, \
         patch('face_recognition.face_encodings') as mock_encodings:
        # Configure mocks
        mock_locations.return_value = MOCK_FACE_LOCATIONS
        mock_encodings.return_value = []  # Encoding failed
        
        # Call function
        locations, encodings = detect_faces(MOCK_IMAGE)
        
        # Verify results
        assert locations == MOCK_FACE_LOCATIONS
        assert encodings == []

def test_compare_faces_success():
    """Test successful face comparison."""
    with patch('face_recognition.compare_faces') as mock_compare:
        # Configure mock
        mock_compare.return_value = [True]
        
        # Call function
        result = compare_faces(MOCK_FACE_ENCODING, MOCK_FACE_ENCODINGS)
        
        # Verify results
        assert result == [True]
        mock_compare.assert_called_once()

def test_compare_faces_error():
    """Test face comparison error handling."""
    with patch('face_recognition.compare_faces') as mock_compare:
        # Configure mock to raise exception
        mock_compare.side_effect = Exception("Test error")
        
        # Call function
        result = compare_faces(MOCK_FACE_ENCODING, MOCK_FACE_ENCODINGS)
        
        # Verify results
        assert result == [False]

def test_face_distance_success():
    """Test successful face distance calculation."""
    with patch('face_recognition.face_distance') as mock_distance:
        # Configure mock
        mock_distance.return_value = np.array([0.5])
        
        # Call function
        result = face_distance(MOCK_FACE_ENCODING, MOCK_FACE_ENCODINGS)
        
        # Verify results
        assert np.array_equal(result, np.array([0.5]))
        mock_distance.assert_called_once()

def test_face_distance_error():
    """Test face distance error handling."""
    with patch('face_recognition.face_distance') as mock_distance:
        # Configure mock to raise exception
        mock_distance.side_effect = Exception("Test error")
        
        # Call function
        result = face_distance(MOCK_FACE_ENCODING, MOCK_FACE_ENCODINGS)
        
        # Verify results
        assert np.array_equal(result, np.array([float('inf')]))
