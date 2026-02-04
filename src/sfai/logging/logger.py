import logging
import re
from tqdm.auto import tqdm

class TqdmHandler(logging.Handler):
    """Custom handler to write logs while a tqdm bar is running.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)
        
class UltralyticsInferenceFilter(logging.Filter):
    """Custom log filter to filter Ultralytics inference logs
    """
    SPEED_PATTERN = re.compile(r"Speed:.*inference", re.IGNORECASE)
    BATCH_PATTERN = re.compile(r"^\d+:\s+\d+x\d+", re.IGNORECASE)

    def filter(self, record):
        msg = record.getMessage()

        if not msg or not msg.strip():
            return False

        if self.SPEED_PATTERN.search(msg) or self.BATCH_PATTERN.search(msg):
            return False
        
        return True
    
LOGGER = logging.getLogger("sfai")
"""Logger. Should be used everywhere to display log messages.
"""
