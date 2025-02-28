"""Web application module for faceroom.

This module provides a Flask-based web interface for the faceroom application.
It currently implements a basic dashboard that will be extended with configuration
and monitoring capabilities in future iterations.
"""

import logging
from typing import Tuple
from flask import Flask, render_template_string

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
    </style>
</head>
<body>
    <div class="container">
        <h1>Faceroom Dashboard</h1>
        <p>Welcome to Faceroom - Real-time Face Recognition System</p>
        <p>Status: Under Construction</p>
        <p>Features coming soon:</p>
        <ul>
            <li>Live camera feed</li>
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
    """Render the dashboard page.
    
    This route provides a simple placeholder dashboard that will be enhanced
    with additional functionality in future iterations.
    
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

@app.route('/')
def index() -> Tuple[str, int]:
    """Redirect root URL to dashboard.
    
    Returns:
        Tuple[str, int]: A tuple containing:
            - A meta refresh redirect to /dashboard
            - HTTP status code (200 for success)
    """
    return '<meta http-equiv="refresh" content="0; url=/dashboard">', 200

if __name__ == '__main__':
    # Configure logging when running directly
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the Flask application in debug mode
    app.run(debug=True, host='127.0.0.1', port=5000)
