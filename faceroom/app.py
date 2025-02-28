"""Web application module for faceroom.

This module provides a Flask-based web interface for the faceroom application,
including a dashboard and live video streaming capabilities.
"""

import logging
import atexit
from typing import Tuple, Any, Union
from flask import Flask, Response, render_template_string, request
from faceroom.streaming import generate_frames, cleanup as cleanup_streaming

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
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
            color: #333;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        .video-container {
            margin: 20px 0;
            text-align: center;
        }
        .video-feed {
            max-width: 100%;
            border: 2px solid #ddd;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Faceroom Dashboard</h1>
        <p>Welcome to Faceroom - Real-time Face Recognition System</p>
        
        <div class="video-container">
            <h2>Live Camera Feed</h2>
            <img src="{{ url_for('live_feed') }}" alt="Live Camera Feed" class="video-feed">
        </div>
        
        <h2>Features</h2>
        <ul>
            <li>Real-time face detection</li>
            <li>Face recognition configuration</li>
            <li>User enrollment</li>
            <li>System statistics</li>
        </ul>
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
        return render_template_string(DASHBOARD_TEMPLATE), 200
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
        Union[Response, Tuple[str, int]]: Flask response object containing the MJPEG stream,
            or an error tuple (message, status_code)
    """
    try:
        # Get camera ID from query parameter, default to 0
        try:
            camera_id = int(request.args.get('camera', 0))
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid camera ID: {str(e)}")
            return "Invalid camera ID", 400
            
        return Response(
            generate_frames(device_id=camera_id),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        logger.error(f"Error setting up video stream: {str(e)}")
        return "Video stream unavailable", 500

@app.route('/')
def index() -> Tuple[str, int]:
    """Redirect root URL to dashboard.
    
    Returns:
        Tuple[str, int]: A tuple containing:
            - A meta refresh redirect to /dashboard
            - HTTP status code (200 for success)
    """
    return '<meta http-equiv="refresh" content="0; url=/dashboard">', 200

# Register cleanup handler
atexit.register(cleanup_streaming)

if __name__ == '__main__':
    # Configure logging when running directly
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the Flask application in debug mode
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)
