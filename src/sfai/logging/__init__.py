import logging
from .logger import LOGGER, TqdmHandler, UltralyticsInferenceFilter
from .progess import PipelineProgess

__all__ = [
    "LOGGER",
    "PipelineProgress",
    "set_log_level",
]

_INITIALIZED = False

def _base_init():
    global _INITIALIZED
    if _INITIALIZED:
        return
    
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False

    if LOGGER.handlers:
        return 
    
    handler = TqdmHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
        )
    )

    LOGGER.addHandler(handler)

    ul_logger = logging.getLogger("ultralytics")
    ul_logger.setLevel(logging.DEBUG)
    ul_logger.addFilter(UltralyticsInferenceFilter())
    ul_logger.handlers = LOGGER.handlers
    ul_logger.propagate = False

def set_log_level(*, level: int = 1):
    """Globally sets the log level

    Args:
        level (int, optional): New log level. Defaults to 1.
    """
    _base_init()

    if level is not None:
        LOGGER.setLevel(level)

_base_init()
