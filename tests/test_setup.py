"""Basic test to verify pytest setup."""

def test_environment_setup():
    """Verify that the test environment is properly configured."""
    assert True, "Basic test passed"

def test_import_dependencies():
    """Verify that core dependencies can be imported."""
    try:
        import cv2
        import face_recognition
        import flask
        import pygame
        assert True, "All dependencies imported successfully"
    except ImportError as e:
        assert False, f"Failed to import dependencies: {str(e)}"
