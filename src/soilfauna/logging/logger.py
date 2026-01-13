import logging
import sys
from typing import Dict
import re
from tqdm.auto import tqdm

VERBOSE_LEVEL: Dict[int, int] = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG
}

class TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)
        
class UltralyticsInferenceFilter(logging.Filter):
    SPEED_PATTERN = re.compile(r"Speed:.*inference", re.IGNORECASE)
    BATCH_PATTERN = re.compile(r"^\d+:\s+\d+x\d+", re.IGNORECASE)

    def filter(self, record):
        msg = record.getMessage()

        if not msg or not msg.strip():
            return False

        if self.SPEED_PATTERN.search(msg) or self.BATCH_PATTERN.search(msg):
            return False
        
        return True
    
LOGGER = logging.getLogger("soilfauna")
"""Logger. Should be used everywhere to display log messages.
"""
