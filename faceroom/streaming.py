"""Video streaming module for faceroom.

This module provides functionality to stream live video with face detection
overlays using MJPEG format. It integrates with the live overlay module
to provide real-time face detection visualization.
"""

import logging
import time
import threading
from typing import Generator, Optional
import cv2
import numpy as np
from faceroom.live_overlay import process_frame_and_overlay
from faceroom.camera import cleanup as cleanup_cameras

# Configure logging
logger = logging.getLogger(__name__)

# Track active streams and their locks
_active_streams = set()
_stream_lock = threading.Lock()

def create_error_frame(
    width: int = 640,
    height: int = 480,
    message: str = "Frame Error"
) -> np.ndarray:
    """Create an error frame with a message.
    
    Args:
        width (int): Frame width in pixels (default: 640)
        height (int): Frame height in pixels (default: 480)
        message (str): Error message to display (default: "Frame Error")
    
    Returns:
        np.ndarray: Error frame with message
    """
    # Create black frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add error message
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (0, 0, 255)  # Red in BGR
    
    # Get text size to center it
    text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    cv2.putText(
        frame,
        message,
        (text_x, text_y),
        font,
        font_scale,
        color,
        thickness
    )
    
    return frame

def generate_frames(
    device_id: int = 0,
    jpeg_quality: int = 90,
    frame_interval: float = 0.033  # ~30 FPS
) -> Generator[bytes, None, None]:
    """Generate a sequence of JPEG frames for MJPEG streaming.
    
    This generator continuously captures frames using @func:process_frame_and_overlay,
    converts them to JPEG format, and yields them in the MJPEG format.
    
    Args:
        device_id (int): Camera device ID (default: 0)
        jpeg_quality (int): JPEG compression quality, 0-100 (default: 90)
        frame_interval (float): Minimum time between frames in seconds (default: 0.033)
    
    Yields:
        bytes: JPEG frame data in MJPEG format
    """
    stream_id = id(time.time())
    with _stream_lock:
        _active_streams.add(stream_id)
    
    try:
        while stream_id in _active_streams:
            try:
                start_time = time.time()
                
                # Capture and process frame
                frame = process_frame_and_overlay(device_id)
                if frame is None:
                    logger.warning("Failed to capture frame, yielding error frame...")
                    frame = create_error_frame()
                
                # Encode frame as JPEG
                success, jpeg_data = cv2.imencode(
                    '.jpg',
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                )
                
                if not success:
                    logger.error("Failed to encode frame as JPEG, yielding error frame...")
                    frame = create_error_frame(message="JPEG Encoding Error")
                    success, jpeg_data = cv2.imencode(
                        '.jpg',
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                    )
                    
                    if not success:
                        logger.error("Failed to encode error frame, skipping...")
                        continue
                
                # Format as MJPEG frame
                yield b'--frame\r\n' \
                      b'Content-Type: image/jpeg\r\n\r\n' + \
                      jpeg_data.tobytes() + \
                      b'\r\n'
                
                # Maintain frame rate
                elapsed = time.time() - start_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                    
            except Exception as e:
                logger.error(f"Error in frame generation: {str(e)}")
                try:
                    # Try to yield an error frame
                    frame = create_error_frame(message="Internal Error")
                    success, jpeg_data = cv2.imencode(
                        '.jpg',
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                    )
                    
                    if success:
                        yield b'--frame\r\n' \
                              b'Content-Type: image/jpeg\r\n\r\n' + \
                              jpeg_data.tobytes() + \
                              b'\r\n'
                except:
                    logger.error("Failed to create error frame")
                    
                time.sleep(frame_interval)  # Avoid rapid error loops
    finally:
        with _stream_lock:
            try:
                _active_streams.remove(stream_id)
            except KeyError:
                # Stream might have been removed by cleanup()
                pass
            
            if not _active_streams:
                cleanup_cameras()

def cleanup():
    """Stop all active streams and cleanup resources.
    
    This should be called when shutting down the application.
    Thread-safe and idempotent.
    """
    with _stream_lock:
        _active_streams.clear()
        cleanup_cameras()
