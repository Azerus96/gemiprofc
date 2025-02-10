import pickle
import os
import logging

logger = logging.getLogger(__name__)

def save_ai_progress(data, filename):
    """Saves AI progress data to a file using pickle."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"AI Progress data saved to {filename}")
        return True
    except (pickle.PickleError, OSError, Exception) as e:
        logger.exception(f"Error saving AI progress data to {filename}: {e}")
        return False

def load_ai_progress(filename):
    """Loads AI progress data from a file using pickle."""
    if not os.path.exists(filename):
        logger.info(f"AI Progress file {filename} not found.")
        return None

    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            logger.info(f"AI Progress data loaded from {filename}")
            return data
    except (pickle.PickleError, OSError, EOFError, Exception) as e:
        logger.exception(f"Error loading AI progress data from {filename}: {e}")
        return None
