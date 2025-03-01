"""Web application module for faceroom.

This module provides a Flask-based web interface for the faceroom application,
including a dashboard and live video streaming capabilities.
"""

# Patch multiprocessing resource tracker to avoid semaphore warnings on macOS
import multiprocessing.resource_tracker as resource_tracker

# Patch register function
_orig_register = resource_tracker.register

def _patched_register(*args):
    if len(args) > 1 and args[1] == "semaphore":
        return  # Skip tracking semaphores
    return _orig_register(*args)

resource_tracker.register = _patched_register

# Patch unregister function
_orig_unregister = resource_tracker.unregister

def _patched_unregister(*args):
    if len(args) > 1 and args[1] == "semaphore":
        return  # Skip unregistering semaphores
    return _orig_unregister(*args)

resource_tracker.unregister = _patched_unregister

# Patch getfd function to avoid KeyError in main function
_orig_getfd = resource_tracker._resource_tracker.getfd

def _patched_getfd(*args):
    if len(args) > 1 and args[1] == "semaphore":
        return -1  # Return invalid fd for semaphores
    return _orig_getfd(*args)

import logging
import atexit
import json
import cv2
import numpy as np
from typing import Tuple, Any, Union, Dict, List
from flask import Flask, Response, render_template_string, request, jsonify
from faceroom.streaming import generate_frames, cleanup as cleanup_streaming
from faceroom.enrollment import (
    enroll_face, get_face_encoding, list_enrolled_users,
    remove_enrolled_face, save_enrollment_database, load_enrollment_database
)
from faceroom.config import get_recognition_threshold, set_recognition_threshold, get_config_summary
from faceroom.analytics import get_metrics, get_metrics_summary

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Basic HTML template for the dashboard
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Faceroom Dashboard</title>
    <style>
        html, body {
            margin: 0;
            padding: 0;
            height: 100%;
            overflow: hidden;
            font-family: Arial, sans-serif;
            background-color: #f2f2f2;
            color: #333;
            line-height: 1.6;
        }

        .grid-container {
            display: grid;
            grid-template-columns: 1fr 2fr 1fr; /* Left and right columns: 1 fraction each, center: 2 fractions */
            gap: 10px;
            height: 100vh; /* Fill the full viewport height */
            padding: 10px;
            box-sizing: border-box;
        }

        /* Columns styling */
        .left-column, .center-column, .right-column {
            background-color: #fff;
            padding: 10px;
            overflow: auto;  /* Allow scrolling inside each column if needed */
            border: 1px solid #ddd;
            border-radius: 5px;
        }

        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
            margin-top: 0;
            font-size: 1.5em;
        }

        h2 {
            color: #2c3e50;
            font-size: 1.2em;
            margin-top: 0;
        }

        h3 {
            font-size: 1em;
            margin-top: 10px;
        }

        .video-container {
            text-align: center;
        }

        .video-feed {
            width: 100%; /* Make sure the video scales to the container's width */
            border: 2px solid #ddd;
            border-radius: 5px;
        }

        .enrollment-container, .config-container, .analytics-container {
            margin-bottom: 10px;
            padding: 10px;
            border: 1px solid #eee;
            border-radius: 5px;
            background-color: #f9f9f9;
            font-size: 0.9em;
        }

        .button {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 8px 16px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 14px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
        }

        .button-danger {
            background-color: #f44336;
        }

        .enrolled-users {
            margin-top: 15px;
        }

        .user-item {
            padding: 8px;
            margin: 4px 0;
            background-color: #e9e9e9;
            border-radius: 3px;
            display: flex;
            justify-content: space-between;
        }

        .slider-container {
            margin: 15px 0;
        }

        .slider {
            width: 100%;
        }

        .slider-value {
            font-weight: bold;
            margin-left: 10px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 15px;
        }

        .metric-card {
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            text-align: center;
        }

        .metric-value {
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
            margin: 8px 0;
        }

        .metric-label {
            font-size: 12px;
            color: #7f8c8d;
        }

        .refresh-button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            margin-left: 8px;
        }

        .last-updated {
            font-size: 11px;
            color: #7f8c8d;
            margin-top: 5px;
            text-align: right;
        }
    </style>
    <script>
        // Function to enroll a face using the current camera frame
        function enrollFace() {
            const userId = document.getElementById('user-id').value;
            if (!userId) {
                alert('Please enter a user ID');
                return;
            }
            
            fetch('/enroll', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: userId,
                    capture_from_camera: true
                }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Face enrolled successfully!');
                    loadEnrolledUsers();
                    loadAnalytics(); // Refresh analytics after enrollment
                } else {
                    alert('Failed to enroll face: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while enrolling face');
            });
        }
        
        // Function to load enrolled users
        function loadEnrolledUsers() {
            fetch('/enrolled-users')
            .then(response => response.json())
            .then(data => {
                const usersList = document.getElementById('enrolled-users-list');
                usersList.innerHTML = '';
                
                if (data.users.length === 0) {
                    usersList.innerHTML = '<p>No users enrolled yet.</p>';
                    return;
                }
                
                data.users.forEach(userId => {
                    const userItem = document.createElement('div');
                    userItem.className = 'user-item';
                    
                    const userIdSpan = document.createElement('span');
                    userIdSpan.textContent = userId;
                    
                    const removeButton = document.createElement('button');
                    removeButton.className = 'button button-danger';
                    removeButton.textContent = 'Remove';
                    removeButton.onclick = function() {
                        removeUser(userId);
                    };
                    
                    userItem.appendChild(userIdSpan);
                    userItem.appendChild(removeButton);
                    usersList.appendChild(userItem);
                });
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('enrolled-users-list').innerHTML = 
                    '<p>Failed to load enrolled users.</p>';
            });
        }
        
        // Function to remove an enrolled user
        function removeUser(userId) {
            if (!confirm(`Are you sure you want to remove ${userId}?`)) {
                return;
            }
            
            fetch('/remove-user', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: userId
                }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert(`User ${userId} removed successfully!`);
                    loadEnrolledUsers();
                    loadAnalytics(); // Refresh analytics after removal
                } else {
                    alert('Failed to remove user: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while removing user');
            });
        }
        
        // Function to update recognition threshold
        function updateThreshold() {
            const threshold = parseFloat(document.getElementById('threshold-slider').value);
            
            fetch('/set-threshold', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    threshold: threshold
                }),
            })
            .then(response => response.json())
            .then(data => {
                if (!data.success) {
                    alert('Failed to update threshold: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while updating threshold');
            });
        }
        
        // Function to load analytics data
        function loadAnalytics() {
            fetch('/analytics')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Update metrics display
                    updateMetricDisplay('frames-processed', data.metrics.frames_processed);
                    updateMetricDisplay('faces-detected', data.metrics.faces_detected);
                    updateMetricDisplay('recognition-matches', data.metrics.recognition_matches);
                    updateMetricDisplay('detection-errors', data.metrics.detection_errors);
                    updateMetricDisplay('enrollment-count', data.metrics.enrollment_count);
                    updateMetricDisplay('faces-per-frame', data.metrics.avg_faces_per_frame);
                    
                    // Update last updated timestamp
                    const now = new Date();
                    document.getElementById('last-updated').textContent = 
                        now.toLocaleTimeString();
                } else {
                    console.error('Failed to load analytics:', data.error);
                }
            })
            .catch(error => {
                console.error('Error loading analytics:', error);
            });
        }
        
        // Helper function to update a metric display
        function updateMetricDisplay(id, value) {
            const element = document.getElementById(id);
            if (element) {
                if (typeof value === 'number' && !Number.isInteger(value)) {
                    // Format floating point numbers to 4 decimal places
                    element.textContent = value.toFixed(4);
                } else {
                    element.textContent = value;
                }
            }
        }
        
        // Set up periodic analytics refresh
        function setupAnalyticsRefresh() {
            // Initial load
            loadAnalytics();
            
            // Refresh every 5 seconds
            setInterval(loadAnalytics, 5000);
        }
        
        // Load enrolled users and configuration when the page loads
        window.onload = function() {
            loadEnrolledUsers();
            setupAnalyticsRefresh();
            
            // Load current threshold value
            fetch('/config')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const threshold = data.config.recognition_threshold;
                    document.getElementById('threshold-slider').value = threshold;
                    document.getElementById('threshold-value').textContent = threshold.toFixed(2);
                }
            })
            .catch(error => {
                console.error('Error loading configuration:', error);
            });
        };
    </script>
</head>
<body>
    <div class="grid-container">
        <!-- Left column: Face Management -->
        <div class="left-column">
            <h1>Faceroom</h1>
            <div class="enrollment-container">
                <h2>Face Enrollment</h2>
                <p>Enroll a new face for recognition by entering a user ID and clicking the button below.</p>
                <div>
                    <label for="user-id">User ID:</label>
                    <input type="text" id="user-id" name="user-id" placeholder="Enter a unique user ID">
                    <button class="button" onclick="enrollFace()">Enroll Current Face</button>
                </div>
                <div class="enrolled-users">
                    <h3>Enrolled Users</h3>
                    <div id="enrolled-users-list">
                        <p>Loading users...</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Center column: Video Feed -->
        <div class="center-column">
            <div class="video-container">
                <h2>Live Camera Feed</h2>
                <img src="/live" alt="Live Camera Feed" class="video-feed">
            </div>
        </div>
        
        <!-- Right column: Analytics -->
        <div class="right-column">
            <div class="analytics-container">
                <h2>System Analytics <button class="refresh-button" onclick="loadAnalytics()">Refresh</button></h2>
                <p>Real-time metrics about system performance and usage.</p>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Frames Processed</div>
                        <div id="frames-processed" class="metric-value">0</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Faces Detected</div>
                        <div id="faces-detected" class="metric-value">0</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Recognition Matches</div>
                        <div id="recognition-matches" class="metric-value">0</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Detection Errors</div>
                        <div id="detection-errors" class="metric-value">0</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Enrolled Users</div>
                        <div id="enrollment-count" class="metric-value">0</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Faces Per Frame</div>
                        <div id="faces-per-frame" class="metric-value">0</div>
                    </div>
                </div>
                <div class="last-updated">
                    Last updated: <span id="last-updated">-</span>
                </div>
            </div>
            <div class="config-container">
                <h2>Recognition Settings</h2>
                <p>Adjust the recognition threshold to control how strictly faces are matched.</p>
                <div class="slider-container">
                    <label for="threshold-slider">Recognition Threshold:</label>
                    <input type="range" id="threshold-slider" class="slider" 
                           min="0.1" max="1.0" step="0.05" value="{{ threshold }}"
                           oninput="document.getElementById('threshold-value').textContent = parseFloat(this.value).toFixed(2);"
                           onchange="updateThreshold()">
                    <span id="threshold-value" class="slider-value">{{ threshold }}</span>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/dashboard')
def dashboard() -> Tuple[str, int]:
    """Render the dashboard page with live video feed.
    
    Returns:
        Tuple[str, int]: A tuple containing:
            - The rendered HTML content
            - HTTP status code (200 for success)
    """
    try:
        # Get current threshold for the template
        threshold = get_recognition_threshold()
        return render_template_string(DASHBOARD_TEMPLATE, threshold=threshold), 200
    except Exception as e:
        logger.error(f"Error rendering dashboard: {str(e)}")
        return "Internal Server Error", 500

@app.route('/live')
def live_feed() -> Union[Response, Tuple[str, int]]:
    """Stream live video with face detection overlays.
    
    This route provides an MJPEG stream of the camera feed with real-time
    face detection overlays using @func:generate_frames.
    
    Query Parameters:
        camera (int): Optional camera device ID (default: 0)
    
    Returns:
        Union[Response, Tuple[str, int]]: Either:
            - A streaming response with MJPEG content
            - An error tuple with message and status code
    """
    try:
        # Get camera ID from query parameters
        camera_param = request.args.get('camera', default='0')
        
        # Validate camera ID is a valid integer
        try:
            camera_id = int(camera_param)
        except ValueError:
            logger.error(f"Invalid camera ID provided: {camera_param}")
            return "Invalid camera ID", 400
        
        # Create a streaming response
        return Response(
            generate_frames(device_id=camera_id),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        logger.error(f"Error in live feed: {str(e)}")
        return "Error: Could not stream video", 500

@app.route('/')
def index() -> Tuple[str, int]:
    """Redirect root URL to dashboard.
    
    Returns:
        Tuple[str, int]: A tuple containing:
            - A meta refresh redirect to /dashboard
            - HTTP status code (200 for success)
    """
    return '<meta http-equiv="refresh" content="0; url=/dashboard">', 200

# Enrollment route for adding new faces
@app.route('/enroll', methods=['POST'])
def enroll() -> Tuple[Response, int]:
    """Enroll a new face in the recognition database.
    
    This endpoint accepts a POST request with JSON data containing:
    - user_id: A unique identifier for the user
    - capture_from_camera: Boolean indicating whether to capture from camera
    
    Returns:
        Tuple[Response, int]: A tuple containing:
            - JSON response with success/error information
            - HTTP status code
    """
    try:
        # Parse request data
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'No user_id provided'}), 400
        
        # Check if capturing from camera
        if data.get('capture_from_camera', False):
            try:
                # Capture a frame from the camera
                from faceroom.camera import capture_frame
                result = capture_frame()
                
                if not result:
                    return jsonify({'success': False, 'error': 'Failed to capture frame from camera'}), 500
                
                ret, frame = result
                
                if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                    return jsonify({'success': False, 'error': 'Invalid frame captured from camera'}), 500
                
                # Convert BGR to RGB for face_recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Enroll the face with additional error handling
                try:
                    enrollment_result = enroll_face(rgb_frame, user_id)
                    if enrollment_result:
                        return jsonify({'success': True}), 200
                    else:
                        return jsonify({'success': False, 'error': 'Failed to enroll face. No face detected or quality check failed.'}), 400
                except Exception as e:
                    logger.error(f"Error in face enrollment process: {str(e)}")
                    return jsonify({'success': False, 'error': f'Face enrollment error: {str(e)}'}), 500
            except Exception as e:
                logger.error(f"Error capturing frame: {str(e)}")
                return jsonify({'success': False, 'error': f'Camera error: {str(e)}'}), 500
        else:
            # TODO: Handle uploaded images in a future enhancement
            return jsonify({'success': False, 'error': 'Image upload not yet supported'}), 501
            
    except Exception as e:
        logger.error(f"Error in enrollment: {str(e)}")
        return jsonify({'success': False, 'error': f'Enrollment error: {str(e)}'}), 500


@app.route('/enrolled-users', methods=['GET'])
def get_enrolled_users() -> Tuple[Response, int]:
    """Get a list of all enrolled users.
    
    Returns:
        Tuple[Response, int]: A tuple containing:
            - JSON response with list of user IDs
            - HTTP status code
    """
    try:
        users = list_enrolled_users()
        return jsonify({'success': True, 'users': users}), 200
    except Exception as e:
        logger.error(f"Error getting enrolled users: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/remove-user', methods=['POST'])
def remove_user() -> Tuple[Response, int]:
    """Remove an enrolled user from the database.
    
    This endpoint accepts a POST request with JSON data containing:
    - user_id: The identifier of the user to remove
    
    Returns:
        Tuple[Response, int]: A tuple containing:
            - JSON response with success/error information
            - HTTP status code
    """
    try:
        # Parse request data
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'No user_id provided'}), 400
        
        # Remove the user
        if remove_enrolled_face(user_id):
            return jsonify({'success': True}), 200
        else:
            return jsonify({'success': False, 'error': f'User {user_id} not found'}), 404
            
    except Exception as e:
        logger.error(f"Error removing user: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/set-threshold', methods=['POST'])
def set_threshold() -> Tuple[Response, int]:
    """Set the face recognition threshold.
    
    This endpoint accepts a POST request with JSON data containing:
    - threshold: The new threshold value (between 0.1 and 1.0)
    
    Returns:
        Tuple[Response, int]: A tuple containing:
            - JSON response with success/error information
            - HTTP status code
    """
    try:
        # Parse request data
        data = request.json
        if not data or 'threshold' not in data:
            return jsonify({'success': False, 'error': 'No threshold provided'}), 400
        
        # Parse and validate the threshold
        try:
            new_threshold = float(data['threshold'])
            set_recognition_threshold(new_threshold)
            return jsonify({'success': True, 'threshold': get_recognition_threshold()}), 200
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
            
    except Exception as e:
        logger.error(f"Error setting threshold: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/config', methods=['GET'])
def get_config() -> Tuple[Response, int]:
    """Get the current configuration settings.
    
    Returns:
        Tuple[Response, int]: A tuple containing:
            - JSON response with configuration information
            - HTTP status code
    """
    try:
        config = get_config_summary()
        return jsonify({'success': True, 'config': config}), 200
    except Exception as e:
        logger.error(f"Error getting configuration: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/analytics', methods=['GET'])
def analytics() -> Tuple[Response, int]:
    """Get real-time analytics metrics.
    
    Returns:
        Tuple[Response, int]: A tuple containing:
            - JSON response with analytics metrics
            - HTTP status code
    """
    try:
        # Get detailed metrics with derived values
        metrics = get_metrics_summary()
        return jsonify({'success': True, 'metrics': metrics}), 200
    except Exception as e:
        logger.error(f"Error retrieving analytics: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Register cleanup handlers
atexit.register(cleanup_streaming)
atexit.register(save_enrollment_database)

# If this module is run directly, start the server
if __name__ == "__main__":
    app.run(debug=True, threaded=True)
