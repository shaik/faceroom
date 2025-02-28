# Faceroom

A real-time face recognition system that greets people with funny sounds when they enter a room.

## Overview

Faceroom is a Python-based application that uses your Mac's camera to detect and recognize faces in real-time. When a known face is detected, it plays a customizable greeting sound. Perfect for creating an interactive and fun environment in your workspace!

## Features

- Real-time face detection and recognition using OpenCV and face_recognition
- Dynamic face enrollment through web interface
- Configurable recognition sensitivity
- Customizable greeting sounds
- Web-based dashboard for monitoring and configuration
- Comprehensive logging and audit system

## Requirements

- Python 3.8+
- macOS (tested on latest versions)
- Webcam access
- System audio output

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/faceroom.git
cd faceroom
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Development

### Running Tests
```bash
pytest
```

### Code Quality
```bash
mypy .
flake8 .
black .
```

### Project Structure
- `faceroom/` - Main application code
- `tests/` - Unit tests
- `static/` - Web assets and resources

## License

MIT License - See LICENSE file for details
