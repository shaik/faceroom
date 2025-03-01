"""Main entry point for the faceroom application.

This module serves as the entry point when running the application as a module
(e.g., python -m faceroom).
"""

# Patch multiprocessing resource tracker to avoid semaphore warnings on macOS
import multiprocessing.resource_tracker as resource_tracker

# Patch register function
_orig_register = resource_tracker.register

def _patched_register(*args, **kwargs):
    if len(args) > 1 and args[1] == "semaphore":
        return  # Skip tracking semaphores
    return _orig_register(*args, **kwargs)

resource_tracker.register = _patched_register

# Patch unregister function
_orig_unregister = resource_tracker.unregister

def _patched_unregister(*args, **kwargs):
    if len(args) > 1 and args[1] == "semaphore":
        return  # Skip unregistering semaphores
    return _orig_unregister(*args, **kwargs)

resource_tracker.unregister = _patched_unregister

# Patch getfd function to avoid KeyError in main function
_orig_getfd = resource_tracker._resource_tracker.getfd

def _patched_getfd(*args, **kwargs):
    if len(args) > 1 and args[1] == "semaphore":
        return -1  # Return invalid fd for semaphores
    return _orig_getfd(*args, **kwargs)

resource_tracker._resource_tracker.getfd = _patched_getfd

# Import and run the Flask application
from faceroom.app import app

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
