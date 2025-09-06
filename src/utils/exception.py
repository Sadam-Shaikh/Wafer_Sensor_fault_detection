import sys

# Try to import from src or directly
try:
    from src.utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger(__name__)

class CustomException(Exception):
    def __init__(self, error_message, error_detail=None):
        super().__init__(error_message)
        self.error_message = error_message
        self.error_detail = error_detail or sys.exc_info()
        logger.error(f"Exception occurred: {error_message}")
        
    def __str__(self):
        return self.error_message