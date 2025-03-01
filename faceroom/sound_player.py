"""Sound Player Module for Faceroom.

This module provides functionality to play sounds when faces are recognized.
It handles mapping face IDs to sound files, enforcing cooldown periods between
sound playbacks, and asynchronous sound playback.
"""

import os
import time
import logging
import threading
from typing import Dict, Optional
from pathlib import Path
import pygame.mixer

# Configure logging
logger = logging.getLogger(__name__)

# Initialize pygame mixer
pygame.mixer.init()

# Constants
DEFAULT_SOUND_FILE = "default.mp3"
DEFAULT_COOLDOWN_SECONDS = 30
SOUNDS_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sounds")

# Ensure sounds directory exists
os.makedirs(SOUNDS_DIRECTORY, exist_ok=True)

# Global state
_last_seen_times: Dict[str, float] = {}  # Maps user_id to last seen timestamp
_sound_cache: Dict[str, pygame.mixer.Sound] = {}  # Cache loaded sounds
_playback_lock = threading.RLock()  # Lock for thread-safe operations
_cooldown_seconds = DEFAULT_COOLDOWN_SECONDS


def set_cooldown(seconds: int) -> None:
    """Set the cooldown period between sound playbacks for the same face.
    
    Args:
        seconds (int): Cooldown period in seconds
    """
    global _cooldown_seconds
    if seconds < 0:
        logger.warning(f"Invalid cooldown value: {seconds}. Using default.")
        seconds = DEFAULT_COOLDOWN_SECONDS
    
    with _playback_lock:
        _cooldown_seconds = seconds
        logger.info(f"Sound playback cooldown set to {seconds} seconds")


def get_cooldown() -> int:
    """Get the current cooldown period.
    
    Returns:
        int: Cooldown period in seconds
    """
    return _cooldown_seconds


def _get_sound_path(user_id: str) -> str:
    """Get the path to the sound file for a user.
    
    Args:
        user_id (str): User ID to find sound for
    
    Returns:
        str: Path to the sound file
    """
    # Try exact match first
    sound_file = f"{user_id}.mp3"
    sound_path = os.path.join(SOUNDS_DIRECTORY, sound_file)
    
    if os.path.exists(sound_path):
        return sound_path
    
    # Try case-insensitive match
    for filename in os.listdir(SOUNDS_DIRECTORY):
        if filename.lower() == sound_file.lower():
            return os.path.join(SOUNDS_DIRECTORY, filename)
    
    # Fall back to default sound
    default_path = os.path.join(SOUNDS_DIRECTORY, DEFAULT_SOUND_FILE)
    if not os.path.exists(default_path):
        logger.warning(f"Default sound file not found at {default_path}")
        # Create an empty file to prevent repeated warnings
        Path(default_path).touch()
    
    return default_path


def _load_sound(sound_path: str) -> Optional[pygame.mixer.Sound]:
    """Load a sound file into memory.
    
    Args:
        sound_path (str): Path to the sound file
    
    Returns:
        Optional[pygame.mixer.Sound]: Loaded sound or None if loading failed
    """
    if sound_path in _sound_cache:
        return _sound_cache[sound_path]
    
    try:
        sound = pygame.mixer.Sound(sound_path)
        _sound_cache[sound_path] = sound
        return sound
    except Exception as e:
        logger.error(f"Failed to load sound file {sound_path}: {e}")
        return None


def _play_sound_thread(sound_path: str) -> None:
    """Thread function to play a sound asynchronously.
    
    Args:
        sound_path (str): Path to the sound file
    """
    sound = _load_sound(sound_path)
    if sound:
        try:
            sound.play()
        except Exception as e:
            logger.error(f"Failed to play sound {sound_path}: {e}")


def play_sound_for_user(user_id: str) -> bool:
    """Play the sound associated with a user if cooldown period has passed.
    
    Args:
        user_id (str): User ID to play sound for
    
    Returns:
        bool: True if sound was played, False otherwise
    """
    current_time = time.time()
    
    with _playback_lock:
        # Check if we're in cooldown period
        last_seen = _last_seen_times.get(user_id, 0)
        if current_time - last_seen < _cooldown_seconds:
            # Update last seen time but don't play sound
            _last_seen_times[user_id] = current_time
            return False
        
        # Update last seen time
        _last_seen_times[user_id] = current_time
    
    # Get sound path
    sound_path = _get_sound_path(user_id)
    if not os.path.exists(sound_path):
        logger.warning(f"Sound file not found for user {user_id}: {sound_path}")
        return False
    
    # Play sound in a separate thread
    threading.Thread(
        target=_play_sound_thread,
        args=(sound_path,),
        daemon=True
    ).start()
    
    logger.info(f"Playing sound for user {user_id}")
    return True


def mark_user_seen(user_id: str) -> None:
    """Mark a user as seen without playing a sound.
    
    This is useful when tracking faces that are continuously visible
    but we don't want to play sounds for them.
    
    Args:
        user_id (str): User ID to mark as seen
    """
    with _playback_lock:
        _last_seen_times[user_id] = time.time()


def reset_user_cooldown(user_id: str) -> None:
    """Reset the cooldown for a specific user.
    
    This allows a sound to be played immediately the next time
    the user is detected, regardless of when they were last seen.
    
    Args:
        user_id (str): User ID to reset cooldown for
    """
    with _playback_lock:
        if user_id in _last_seen_times:
            del _last_seen_times[user_id]
            logger.debug(f"Reset cooldown for user {user_id}")


def reset_all_cooldowns() -> None:
    """Reset cooldowns for all users."""
    with _playback_lock:
        _last_seen_times.clear()
        logger.debug("Reset all user cooldowns")


def cleanup() -> None:
    """Clean up resources used by the sound player."""
    global _sound_cache
    
    with _playback_lock:
        _sound_cache.clear()
    
    try:
        pygame.mixer.quit()
        logger.debug("Sound player resources cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up sound player: {e}")
