"""Tests for the sound player module.

This module tests the sound player functionality, including:
- Sound file mapping
- Cooldown functionality
- Sound playback
"""

import os
import time
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from faceroom.sound_player import (
    play_sound_for_user,
    mark_user_seen,
    reset_user_cooldown,
    reset_all_cooldowns,
    set_cooldown,
    get_cooldown,
    _get_sound_path,
    _load_sound,
    SOUNDS_DIRECTORY,
    DEFAULT_SOUND_FILE
)


class TestSoundPlayer(unittest.TestCase):
    """Test cases for the sound player module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test sound files
        self.temp_dir = tempfile.mkdtemp()
        
        # Save original directory and patch it for testing
        self.original_sounds_dir = SOUNDS_DIRECTORY
        
        # Reset all cooldowns before each test
        reset_all_cooldowns()
        
        # Reset cooldown to default
        set_cooldown(30)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        
        # Reset all cooldowns
        reset_all_cooldowns()
    
    @patch('faceroom.sound_player.SOUNDS_DIRECTORY')
    @patch('faceroom.sound_player._load_sound')
    @patch('faceroom.sound_player.pygame.mixer.Sound.play')
    def test_play_sound_for_user(self, mock_play, mock_load_sound, mock_sounds_dir):
        """Test playing a sound for a user."""
        # Setup mocks
        mock_sounds_dir.return_value = self.temp_dir
        mock_sound = MagicMock()
        mock_load_sound.return_value = mock_sound
        
        # Test first play
        result = play_sound_for_user("test_user")
        self.assertTrue(result)
        mock_load_sound.assert_called_once()
        mock_play.assert_called_once()
        
        # Reset mocks
        mock_load_sound.reset_mock()
        mock_play.reset_mock()
        
        # Test cooldown - should not play again
        result = play_sound_for_user("test_user")
        self.assertFalse(result)
        mock_load_sound.assert_not_called()
        mock_play.assert_not_called()
    
    def test_cooldown_functionality(self):
        """Test the cooldown functionality."""
        # Set a short cooldown for testing
        set_cooldown(2)
        self.assertEqual(get_cooldown(), 2)
        
        # Mark user as seen
        mark_user_seen("test_user")
        
        # Should not play sound immediately
        with patch('faceroom.sound_player._play_sound_thread') as mock_play:
            result = play_sound_for_user("test_user")
            self.assertFalse(result)
            mock_play.assert_not_called()
        
        # Wait for cooldown to expire
        time.sleep(3)
        
        # Should play sound after cooldown
        with patch('faceroom.sound_player._play_sound_thread') as mock_play:
            with patch('faceroom.sound_player._get_sound_path') as mock_path:
                mock_path.return_value = os.path.join(self.temp_dir, "test.mp3")
                with patch('os.path.exists') as mock_exists:
                    mock_exists.return_value = True
                    result = play_sound_for_user("test_user")
                    self.assertTrue(result)
                    mock_play.assert_called_once()
    
    def test_reset_user_cooldown(self):
        """Test resetting the cooldown for a specific user."""
        # Mark user as seen
        mark_user_seen("test_user")
        
        # Reset cooldown for user
        reset_user_cooldown("test_user")
        
        # Should play sound immediately
        with patch('faceroom.sound_player._play_sound_thread') as mock_play:
            with patch('faceroom.sound_player._get_sound_path') as mock_path:
                mock_path.return_value = os.path.join(self.temp_dir, "test.mp3")
                with patch('os.path.exists') as mock_exists:
                    mock_exists.return_value = True
                    result = play_sound_for_user("test_user")
                    self.assertTrue(result)
                    mock_play.assert_called_once()
    
    @patch('faceroom.sound_player.SOUNDS_DIRECTORY', new_callable=lambda: tempfile.mkdtemp())
    def test_get_sound_path(self, mock_dir):
        """Test getting the sound path for a user."""
        sounds_dir = mock_dir
        
        # Create test sound files
        test_file = os.path.join(sounds_dir, "test_user.mp3")
        default_file = os.path.join(sounds_dir, DEFAULT_SOUND_FILE)
        
        with open(test_file, 'w') as f:
            f.write("test")
        with open(default_file, 'w') as f:
            f.write("default")
        
        # Test exact match
        path = _get_sound_path("test_user")
        self.assertEqual(path, test_file)
        
        # Test case-insensitive match
        path = _get_sound_path("TEST_USER")
        self.assertEqual(path, test_file)
        
        # Test fallback to default
        path = _get_sound_path("nonexistent_user")
        self.assertEqual(path, default_file)
        
        # Clean up
        shutil.rmtree(sounds_dir)
    
    def test_invalid_cooldown(self):
        """Test setting an invalid cooldown value."""
        # Try to set a negative cooldown
        set_cooldown(-10)
        
        # Should use default instead
        self.assertEqual(get_cooldown(), 30)


if __name__ == '__main__':
    unittest.main()
