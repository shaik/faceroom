"""Unit tests for face matching functionality in the live overlay module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from faceroom.live_overlay import match_faces_with_enrollments, UNKNOWN_LABEL


@pytest.fixture
def mock_face_encodings():
    """Create mock face encodings for testing."""
    # Create 3 random face encodings (128-dimensional vectors)
    return [
        np.random.rand(128).astype(np.float64)
        for _ in range(3)
    ]


@pytest.fixture
def mock_enrolled_faces():
    """Create mock enrolled faces for testing."""
    return {
        "John Doe": np.random.rand(128).astype(np.float64),
        "Jane Smith": np.random.rand(128).astype(np.float64),
    }


def test_match_faces_with_no_enrollments(mock_face_encodings):
    """Test matching when no faces are enrolled."""
    with patch('faceroom.live_overlay.get_all_enrollments', return_value={}):
        # Test with empty enrollment database
        labels = match_faces_with_enrollments(mock_face_encodings)
        
        # Should return "Unknown" for all faces
        assert len(labels) == len(mock_face_encodings)
        assert all(label == UNKNOWN_LABEL for label in labels)


def test_match_faces_with_enrollments(mock_face_encodings, mock_enrolled_faces):
    """Test matching with enrolled faces."""
    with patch('faceroom.live_overlay.get_all_enrollments', return_value=mock_enrolled_faces):
        # Mock compare_faces to return matches for specific faces
        def mock_compare_faces_side_effect(known_encoding, face_encodings, **kwargs):
            # Get the user_id for this known_encoding
            user_id = None
            for uid, enc in mock_enrolled_faces.items():
                if np.array_equal(known_encoding, enc):
                    user_id = uid
                    break
            
            # First encoding matches John Doe, third encoding matches Jane Smith
            if user_id == "John Doe" and np.array_equal(face_encodings[0], mock_face_encodings[0]):
                return [True]
            elif user_id == "Jane Smith" and np.array_equal(face_encodings[0], mock_face_encodings[2]):
                return [True]
            else:
                return [False]
        
        # Mock face_distance to return appropriate distances
        def mock_face_distance_side_effect(known_encoding, face_encodings):
            # Get the user_id for this known_encoding
            user_id = None
            for uid, enc in mock_enrolled_faces.items():
                if np.array_equal(known_encoding, enc):
                    user_id = uid
                    break
            
            # Return appropriate distances based on the match
            if user_id == "John Doe" and np.array_equal(face_encodings[0], mock_face_encodings[0]):
                return np.array([0.4])  # Good match
            elif user_id == "Jane Smith" and np.array_equal(face_encodings[0], mock_face_encodings[2]):
                return np.array([0.3])  # Better match
            else:
                return np.array([0.8])  # Poor match
        
        with patch('faceroom.live_overlay.compare_faces', side_effect=mock_compare_faces_side_effect):
            with patch('faceroom.live_overlay.face_distance', side_effect=mock_face_distance_side_effect):
                # Call the function
                labels = match_faces_with_enrollments(mock_face_encodings)
                
                # Check results
                assert len(labels) == 3
                assert labels[0] == "John Doe"  # First face should match John Doe
                assert labels[1] == UNKNOWN_LABEL  # Second face should be unknown
                assert labels[2] == "Jane Smith"  # Third face should match Jane Smith


def test_match_faces_with_no_matches(mock_face_encodings, mock_enrolled_faces):
    """Test when no faces match any enrollments."""
    with patch('faceroom.live_overlay.get_all_enrollments', return_value=mock_enrolled_faces):
        # Mock compare_faces to always return False (no matches)
        with patch('faceroom.live_overlay.compare_faces', return_value=[False]):
            with patch('faceroom.live_overlay.face_distance', return_value=np.array([0.9])):
                # Call the function
                labels = match_faces_with_enrollments(mock_face_encodings)
                
                # Check results - all should be unknown
                assert len(labels) == len(mock_face_encodings)
                assert all(label == UNKNOWN_LABEL for label in labels)
