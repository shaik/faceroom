"""Tests for the configuration module.

This module contains unit tests for the configuration module, including
getting and setting the recognition threshold.
"""

import pytest
from faceroom.config import (
    get_recognition_threshold, set_recognition_threshold,
    get_config_summary, RECOGNITION_THRESHOLD
)


def test_get_recognition_threshold():
    """Test that get_recognition_threshold returns the default value."""
    # Get the default threshold
    threshold = get_recognition_threshold()
    
    # Assert it matches the module's default
    assert threshold == RECOGNITION_THRESHOLD
    assert 0.1 <= threshold <= 1.0


def test_set_recognition_threshold_valid():
    """Test that set_recognition_threshold updates the value with valid input."""
    # Save the original threshold
    original_threshold = get_recognition_threshold()
    
    try:
        # Set a new valid threshold
        new_threshold = 0.75
        set_recognition_threshold(new_threshold)
        
        # Assert the threshold was updated
        assert get_recognition_threshold() == new_threshold
        
        # Test another valid value
        new_threshold = 0.1  # Minimum allowed
        set_recognition_threshold(new_threshold)
        assert get_recognition_threshold() == new_threshold
        
        # Test maximum allowed value
        new_threshold = 1.0  # Maximum allowed
        set_recognition_threshold(new_threshold)
        assert get_recognition_threshold() == new_threshold
    finally:
        # Restore the original threshold
        set_recognition_threshold(original_threshold)


def test_set_recognition_threshold_invalid():
    """Test that set_recognition_threshold raises ValueError with invalid input."""
    # Save the original threshold
    original_threshold = get_recognition_threshold()
    
    try:
        # Test values below the minimum
        with pytest.raises(ValueError):
            set_recognition_threshold(0.05)
        
        # Test values above the maximum
        with pytest.raises(ValueError):
            set_recognition_threshold(1.1)
        
        # Test negative values
        with pytest.raises(ValueError):
            set_recognition_threshold(-0.5)
        
        # Assert the threshold was not changed
        assert get_recognition_threshold() == original_threshold
    finally:
        # Restore the original threshold (should be unchanged)
        set_recognition_threshold(original_threshold)


def test_get_config_summary():
    """Test that get_config_summary returns a dictionary with all configuration values."""
    # Get the config summary
    config = get_config_summary()
    
    # Assert it's a dictionary
    assert isinstance(config, dict)
    
    # Assert it contains the recognition threshold
    assert 'recognition_threshold' in config
    assert config['recognition_threshold'] == get_recognition_threshold()
