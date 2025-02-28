"""Unit tests for the web application module."""

import pytest
from faceroom.app import app

@pytest.fixture
def client():
    """Create a test client for the Flask application.
    
    Returns:
        FlaskClient: A test client for making requests to the application.
    """
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_dashboard_route(client):
    """Test the dashboard route returns expected content and status code."""
    # Make a GET request to /dashboard
    response = client.get('/dashboard')
    
    # Check status code
    assert response.status_code == 200
    
    # Check content
    content = response.data.decode()
    assert 'Faceroom Dashboard' in content
    assert 'Welcome to Faceroom' in content
    assert 'Status: Under Construction' in content
    
    # Check expected feature list
    assert 'Live camera feed' in content
    assert 'Face recognition configuration' in content
    assert 'User enrollment' in content
    assert 'System statistics' in content

def test_index_route(client):
    """Test the index route redirects to dashboard."""
    # Make a GET request to /
    response = client.get('/')
    
    # Check status code
    assert response.status_code == 200
    
    # Check redirect meta tag
    content = response.data.decode()
    assert 'meta http-equiv="refresh"' in content
    assert 'url=/dashboard' in content

def test_dashboard_error_handling(client, monkeypatch):
    """Test error handling in dashboard route."""
    # Mock render_template_string to raise an exception
    def mock_render(*args, **kwargs):
        raise Exception("Test error")
    
    monkeypatch.setattr(
        'faceroom.app.render_template_string',
        mock_render
    )
    
    # Make a GET request to /dashboard
    response = client.get('/dashboard')
    
    # Check status code and error message
    assert response.status_code == 500
    assert b"Internal Server Error" in response.data
