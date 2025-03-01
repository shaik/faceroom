# Faceroom

A sophisticated real-time face recognition system that detects people and plays personalized sounds when they enter a room.

## Overview

Faceroom is a powerful Python-based application that leverages computer vision and machine learning to detect and recognize faces in real-time using your device's camera. When a known face is detected, it plays a customizable sound associated with that person. The system includes a web-based dashboard for easy management, monitoring, and configuration.

## Features

### Core Functionality
- **Real-time Face Detection**: Continuously processes video frames to identify human faces
- **Face Recognition**: Matches detected faces against a database of enrolled users
- **Sound Playback**: Plays personalized sounds when recognized faces appear
- **Cooldown System**: Prevents repeated sound playback with configurable intervals
- **Visual Overlays**: Displays bounding boxes and labels around detected faces

### User Interface
- **Web Dashboard**: Intuitive browser-based control panel
- **Live Video Feed**: Real-time display of camera input with face recognition overlays
- **User Management**: Enroll, list, and remove users through the interface
- **System Log Console**: Visual feedback of system events and operations
- **Configuration Controls**: Adjust recognition threshold and sound cooldown settings

### Technical Features
- **Asynchronous Processing**: Non-blocking sound playback and video processing
- **Metrics Tracking**: Comprehensive analytics on system performance
- **Configurable Parameters**: Adjustable recognition sensitivity and sound cooldown periods
- **Error Handling**: Robust error management and reporting
- **Resource Management**: Proper cleanup of system resources

## System Architecture

Faceroom is built with a modular architecture that separates concerns and promotes maintainability:

### Core Modules
- **Camera Module**: Handles video capture and frame processing
- **Face Recognition Module**: Detects and identifies faces in video frames
- **Enrollment Module**: Manages the database of enrolled users and their face encodings
- **Sound Player Module**: Controls sound file selection and playback with cooldown logic
- **Live Overlay Module**: Generates visual indicators for the video feed
- **Analytics Module**: Tracks and reports system metrics

### Web Application
- **Flask Backend**: Serves the web interface and API endpoints
- **Streaming Module**: Handles real-time video streaming to the browser
- **RESTful API**: Provides endpoints for all system operations
- **Frontend Interface**: Responsive dashboard for system control

## Requirements

### Software Dependencies
- Python 3.8+
- OpenCV (computer vision library)
- face_recognition (face detection and recognition)
- Flask (web framework)
- Pygame (audio playback)
- NumPy (numerical processing)

### Hardware Requirements
- Webcam or camera device
- Audio output capability
- Recommended: CPU with decent processing power for real-time operations

### Operating System Compatibility
- Primary: macOS (fully tested)
- Secondary: Linux (should work with minimal adjustments)
- Windows: May require additional configuration

## Installation

### Standard Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/faceroom.git
cd faceroom
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python -m faceroom
```

5. Access the dashboard at http://localhost:5000

### Development Setup

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Set up pre-commit hooks (optional):
```bash
pre-commit install
```

## Usage Guide

### Getting Started

1. **Launch the Application**: Run `python -m faceroom` to start the server
2. **Access the Dashboard**: Open http://localhost:5000 in your browser
3. **Enroll Users**: Enter a user ID and click "Enroll Current Face" when a face is visible
4. **Add Sound Files**: Place MP3 files in the `data/sounds` directory, named to match user IDs (e.g., `john.mp3`)
5. **Configure Settings**: Adjust recognition threshold and sound cooldown as needed

### Face Enrollment

To add a new user to the system:
1. Position the person's face in the camera view
2. Enter a unique user ID in the enrollment field
3. Click "Enroll Current Face"
4. Verify the user appears in the enrolled users list

### Sound Management

The system plays sounds when recognized faces appear:
- Sound files should be placed in `data/sounds/`
- Files should be named to match user IDs (e.g., `john.mp3` for user "john")
- Case-insensitive matching is supported
- If no matching sound file exists, `default.mp3` will be played
- Sounds will only play once per appearance, with a configurable cooldown period

### Configuration Options

#### Recognition Threshold
Controls the strictness of face matching:
- Lower values (0.4-0.5): More strict matching, fewer false positives
- Higher values (0.6-0.7): More lenient matching, may have more false positives
- Default: 0.6

#### Sound Cooldown
Controls how frequently sounds can play for the same person:
- Range: 1-300 seconds
- Default: 30 seconds
- Prevents sound spam when a person remains in view

## API Reference

Faceroom provides a RESTful API for programmatic control:

### Enrollment Endpoints
- `POST /enroll`: Enroll a new face
- `GET /users`: List all enrolled users
- `POST /remove-user`: Remove an enrolled user

### Configuration Endpoints
- `GET /config`: Get current configuration
- `POST /set-threshold`: Set recognition threshold
- `GET /get-sound-cooldown`: Get current sound cooldown period
- `POST /set-sound-cooldown`: Set sound cooldown period

### Analytics Endpoints
- `GET /analytics`: Get system metrics and statistics

## Development

### Project Structure
```
faceroom/
├── data/
│   ├── enrolled_faces.json  # Database of enrolled users
│   └── sounds/              # Sound files for playback
├── faceroom/                # Main application code
│   ├── __init__.py
│   ├── __main__.py          # Entry point
│   ├── analytics.py         # Metrics tracking
│   ├── app.py               # Web application
│   ├── camera.py            # Camera handling
│   ├── config.py            # Configuration management
│   ├── enrollment.py        # User enrollment
│   ├── face_recognition_module.py  # Face detection/recognition
│   ├── live_overlay.py      # Visual indicators
│   ├── sound_player.py      # Sound playback
│   └── streaming.py         # Video streaming
├── tests/                   # Unit tests
├── requirements.txt         # Dependencies
└── README.md               # This file
```

### Running Tests
```bash
pytest                  # Run all tests
pytest -xvs             # Verbose mode
pytest --cov=faceroom   # With coverage report
```

### Code Quality Tools
```bash
mypy .                  # Type checking
flake8 .                # Linting
black .                 # Code formatting
```

## Troubleshooting

### Common Issues

#### Camera Access Problems
- Ensure your camera is connected and working
- Check that your browser has permission to access the camera
- Try a different camera device ID if you have multiple cameras

#### Sound Playback Issues
- Verify sound files exist in the correct directory
- Check that file names match user IDs (case-insensitive)
- Ensure your system's audio output is working
- Check that pygame is properly installed

#### Recognition Problems
- Adjust the recognition threshold for better accuracy
- Ensure good lighting conditions for optimal face detection
- Re-enroll users if recognition is consistently failing

### Logs and Debugging

Check the system log console in the web interface for real-time feedback on operations and errors.

## Contributing

Contributions to Faceroom are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

### Coding Standards
- Follow PEP 8 style guidelines
- Include type hints for all functions
- Write unit tests for new features
- Document all public functions and classes

## License

MIT License - See LICENSE file for details

## Acknowledgements

- [OpenCV](https://opencv.org/) - Computer vision library
- [face_recognition](https://github.com/ageitgey/face_recognition) - Face recognition library
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Pygame](https://www.pygame.org/) - Audio playback library
