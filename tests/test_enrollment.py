"""Unit tests for the face enrollment module."""

import pytest
import numpy as np
import os
import json
from unittest.mock import patch, MagicMock

from faceroom.enrollment import (
    enroll_face,
    get_face_encoding,
    list_enrolled_users,
    remove_enrolled_face,
    save_enrollment_database,
    load_enrollment_database,
    _check_face_quality,
    _enrolled_faces,
    DEFAULT_DB_PATH
)

# Create a test directory for database files
TEST_DB_PATH = os.path.join(os.path.dirname(__file__), 'test_data', 'test_enrolled_faces.json')


@pytest.fixture
def cleanup_test_db():
    """Clean up test database files before and after tests."""
    # Clean up before test
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    
    # Run the test
    yield
    
    # Clean up after test
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture
def mock_face_data():
    """Create mock face data for testing."""
    # Create a simple test image (100x100 RGB)
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Mock face location (top, right, bottom, left)
    # This represents a face in the center of the image
    face_location = (30, 70, 70, 30)
    
    # Mock face encoding (128-dimensional vector)
    face_encoding = np.random.random(128)
    
    return {
        'image': test_image,
        'location': face_location,
        'encoding': face_encoding
    }


@pytest.fixture
def reset_enrolled_faces():
    """Reset the enrolled faces database before and after tests."""
    # Save original state
    original = dict(_enrolled_faces)
    
    # Clear the database
    _enrolled_faces.clear()
    
    # Run the test
    yield
    
    # Restore original state
    _enrolled_faces.clear()
    _enrolled_faces.update(original)


def test_check_face_quality_good():
    """Test that a well-positioned face passes quality check."""
    # Create a test image
    image = np.ones((200, 200, 3), dtype=np.uint8)
    
    # Create a face location that should pass quality checks
    # Face is centered and of reasonable size
    face_location = (70, 130, 130, 70)  # top, right, bottom, left
    
    # Check quality
    passed, reason = _check_face_quality(image, face_location)
    
    # Assert
    assert passed is True
    assert reason == ""


def test_check_face_quality_too_small():
    """Test that a face that's too small fails quality check."""
    # Create a test image
    image = np.ones((200, 200, 3), dtype=np.uint8)
    
    # Create a face location that's too small (less than 10% of image width)
    face_location = (95, 105, 105, 95)  # top, right, bottom, left - only 10px wide (5% of image)
    
    # Check quality
    passed, reason = _check_face_quality(image, face_location)
    
    # Assert
    assert passed is False
    assert "too small" in reason


def test_check_face_quality_too_large():
    """Test that a face that's too large fails quality check."""
    # Create a test image
    image = np.ones((200, 200, 3), dtype=np.uint8)
    
    # Create a face location that's too large
    face_location = (10, 190, 190, 10)  # top, right, bottom, left
    
    # Check quality
    passed, reason = _check_face_quality(image, face_location)
    
    # Assert
    assert passed is False
    assert "too large" in reason


def test_check_face_quality_not_centered():
    """Test that a face that's not centered fails quality check."""
    # Create a test image
    image = np.ones((200, 200, 3), dtype=np.uint8)
    
    # Create a face location that's not centered
    face_location = (10, 60, 60, 10)  # top, right, bottom, left (top-left corner)
    
    # Check quality
    passed, reason = _check_face_quality(image, face_location)
    
    # Assert
    assert passed is False
    assert "not centered" in reason


def test_enroll_face_success(mock_face_data, reset_enrolled_faces):
    """Test successful face enrollment."""
    # Mock the detect_faces function to return our test data
    with patch('faceroom.enrollment.detect_faces') as mock_detect:
        mock_detect.return_value = ([mock_face_data['location']], [mock_face_data['encoding']])
        
        # Call enroll_face
        result = enroll_face(mock_face_data['image'], 'test_user')
        
        # Assert
        assert result is True
        assert 'test_user' in _enrolled_faces
        np.testing.assert_array_equal(_enrolled_faces['test_user'], mock_face_data['encoding'])


def test_enroll_face_no_faces(mock_face_data, reset_enrolled_faces):
    """Test enrollment failure when no faces are detected."""
    # Mock the detect_faces function to return no faces
    with patch('faceroom.enrollment.detect_faces') as mock_detect:
        mock_detect.return_value = ([], [])
        
        # Call enroll_face
        result = enroll_face(mock_face_data['image'], 'test_user')
        
        # Assert
        assert result is False
        assert 'test_user' not in _enrolled_faces


def test_enroll_face_no_encodings(mock_face_data, reset_enrolled_faces):
    """Test enrollment failure when face is detected but encoding fails."""
    # Mock the detect_faces function to return a face location but no encoding
    with patch('faceroom.enrollment.detect_faces') as mock_detect:
        mock_detect.return_value = ([mock_face_data['location']], [])
        
        # Call enroll_face
        result = enroll_face(mock_face_data['image'], 'test_user')
        
        # Assert
        assert result is False
        assert 'test_user' not in _enrolled_faces


def test_enroll_face_quality_check_failure(mock_face_data, reset_enrolled_faces):
    """Test enrollment failure when face quality check fails."""
    # Mock the detect_faces function to return our test data
    with patch('faceroom.enrollment.detect_faces') as mock_detect:
        mock_detect.return_value = ([mock_face_data['location']], [mock_face_data['encoding']])
        
        # Mock the quality check to fail
        with patch('faceroom.enrollment._check_face_quality') as mock_quality:
            mock_quality.return_value = (False, "Test quality failure")
            
            # Call enroll_face
            result = enroll_face(mock_face_data['image'], 'test_user')
            
            # Assert
            assert result is False
            assert 'test_user' not in _enrolled_faces


def test_enroll_face_invalid_inputs():
    """Test enrollment with invalid inputs."""
    # Test with None image - now returns False instead of raising ValueError
    result = enroll_face(None, 'test_user')
    assert result is False
    
    # Test with empty image - create a valid numpy array but empty
    empty_img = np.array([], dtype=np.uint8).reshape(0, 0, 3)
    result = enroll_face(empty_img, 'test_user')
    assert result is False
    
    # Test with non-ndarray image - now returns False instead of raising ValueError
    result = enroll_face("not an image", 'test_user')
    assert result is False
    
    # Test with empty user_id
    result = enroll_face(np.ones((100, 100, 3), dtype=np.uint8), '')
    assert result is False
    
    # Test with non-string user_id - now returns False instead of raising ValueError
    result = enroll_face(np.ones((100, 100, 3), dtype=np.uint8), 123)
    assert result is False


def test_enroll_face_valid():
    """Test that enrolling a valid face works."""
    image = np.ones((100, 100, 3), dtype=np.uint8)  # Dummy image
    user_id = "valid_user"
    result = enroll_face(image, user_id)
    assert result is True


def test_enroll_face_invalid_image():
    """Test that enrolling with an invalid image fails."""
    image = None  # Invalid image
    user_id = "valid_user"
    result = enroll_face(image, user_id)
    assert result is False


def test_enroll_face_invalid_user_id():
    """Test that enrolling with an invalid user ID fails."""
    image = np.ones((100, 100, 3), dtype=np.uint8)  # Dummy image
    user_id = 123  # Invalid user ID
    result = enroll_face(image, user_id)
    assert result is False


def test_get_face_encoding(mock_face_data, reset_enrolled_faces):
    """Test retrieving a face encoding by user ID."""
    # Add a test face to the database
    _enrolled_faces['test_user'] = mock_face_data['encoding']
    
    # Retrieve the encoding
    encoding = get_face_encoding('test_user')
    
    # Assert
    assert encoding is not None
    np.testing.assert_array_equal(encoding, mock_face_data['encoding'])
    
    # Test retrieving a non-existent user
    assert get_face_encoding('nonexistent_user') is None


def test_list_enrolled_users(reset_enrolled_faces):
    """Test listing all enrolled users."""
    # Add some test users
    _enrolled_faces['user1'] = np.random.random(128)
    _enrolled_faces['user2'] = np.random.random(128)
    
    # Get the list of users
    users = list_enrolled_users()
    
    # Assert
    assert len(users) == 2
    assert 'user1' in users
    assert 'user2' in users


def test_remove_enrolled_face(reset_enrolled_faces):
    """Test removing an enrolled face."""
    # Add a test user
    _enrolled_faces['test_user'] = np.random.random(128)
    
    # Remove the user
    result = remove_enrolled_face('test_user')
    
    # Assert
    assert result is True
    assert 'test_user' not in _enrolled_faces
    
    # Test removing a non-existent user
    result = remove_enrolled_face('nonexistent_user')
    assert result is False


def test_save_and_load_database(mock_face_data, reset_enrolled_faces, cleanup_test_db):
    """Test saving and loading the enrollment database."""
    # Add a test face to the database
    _enrolled_faces['test_user'] = mock_face_data['encoding']
    
    # Save the database
    result = save_enrollment_database(TEST_DB_PATH)
    assert result is True
    assert os.path.exists(TEST_DB_PATH)
    
    # Clear the in-memory database
    _enrolled_faces.clear()
    assert len(_enrolled_faces) == 0
    
    # Load the database
    result = load_enrollment_database(TEST_DB_PATH)
    assert result is True
    
    # Assert
    assert 'test_user' in _enrolled_faces
    np.testing.assert_array_almost_equal(_enrolled_faces['test_user'], mock_face_data['encoding'])


def test_load_nonexistent_database(reset_enrolled_faces, cleanup_test_db):
    """Test loading a non-existent database."""
    # Ensure the test database doesn't exist
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    
    # Try to load the database
    result = load_enrollment_database(TEST_DB_PATH)
    
    # Assert
    assert result is False
