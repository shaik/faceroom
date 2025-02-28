"""Tests for the configuration endpoints in the web application.

This module contains unit tests for the configuration endpoints in the web application,
including setting the recognition threshold and getting the configuration.
"""

import json
import pytest
from faceroom.app import app
from faceroom.config import get_recognition_threshold, set_recognition_threshold


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def original_threshold():
    """Save and restore the original recognition threshold."""
    original = get_recognition_threshold()
    yield original
    set_recognition_threshold(original)


def test_get_config(client):
    """Test that the /config endpoint returns the current configuration."""
    # Get the config
    response = client.get('/config')
    
    # Assert the response is successful
    assert response.status_code == 200
    
    # Parse the response data
    data = json.loads(response.data)
    
    # Assert the response contains the expected data
    assert data['success'] is True
    assert 'config' in data
    assert 'recognition_threshold' in data['config']
    assert data['config']['recognition_threshold'] == get_recognition_threshold()


def test_set_threshold_valid(client, original_threshold):
    """Test that the /set-threshold endpoint updates the threshold with valid input."""
    # Set a new valid threshold
    new_threshold = 0.75
    response = client.post(
        '/set-threshold',
        data=json.dumps({'threshold': new_threshold}),
        content_type='application/json'
    )
    
    # Assert the response is successful
    assert response.status_code == 200
    
    # Parse the response data
    data = json.loads(response.data)
    
    # Assert the response contains the expected data
    assert data['success'] is True
    assert data['threshold'] == new_threshold
    
    # Assert the threshold was actually updated
    assert get_recognition_threshold() == new_threshold


def test_set_threshold_invalid(client, original_threshold):
    """Test that the /set-threshold endpoint rejects invalid input."""
    # Try to set an invalid threshold
    invalid_threshold = 1.5  # Above maximum
    response = client.post(
        '/set-threshold',
        data=json.dumps({'threshold': invalid_threshold}),
        content_type='application/json'
    )
    
    # Assert the response indicates an error
    assert response.status_code == 400
    
    # Parse the response data
    data = json.loads(response.data)
    
    # Assert the response contains the expected error
    assert data['success'] is False
    assert 'error' in data
    
    # Assert the threshold was not updated
    assert get_recognition_threshold() == original_threshold


def test_set_threshold_missing_data(client, original_threshold):
    """Test that the /set-threshold endpoint rejects requests with missing data."""
    # Send a request with missing threshold
    response = client.post(
        '/set-threshold',
        data=json.dumps({}),
        content_type='application/json'
    )
    
    # Assert the response indicates an error
    assert response.status_code == 400
    
    # Parse the response data
    data = json.loads(response.data)
    
    # Assert the response contains the expected error
    assert data['success'] is False
    assert 'error' in data
    assert 'No threshold provided' in data['error']
    
    # Assert the threshold was not updated
    assert get_recognition_threshold() == original_threshold
