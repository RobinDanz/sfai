import logging
import sys
from typing import Dict, Optional
import re
from tqdm.auto import tqdm

VERBOSE_LEVEL: Dict[int, int] = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG
}

class BaseLogger:
    _instances: Dict[str, "BaseLogger"] = {}
    
    def __new__(cls, name: str = "soil-fauna-ai", level: int = 1):
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]
    
    def __init__(self, name: str = "soil-fauna-ai", level: int = 1):
        if hasattr(self, "_initialized"):
            return
        
        self._initialized = True
        
        self.name = name
        self.level = level
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(VERBOSE_LEVEL.get(self.level, logging.INFO))
        self.logger.propagate = False
        
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(self._default_formatter())
            self.logger.addHandler(handler)
            
    def set_level(self, level: int):
        self.level = level
        self.logger.setLevel(VERBOSE_LEVEL.get(self.level, logging.INFO))
        
    def set_formatter(self, fmt: str):
        formatter = logging.Formatter(fmt)
        for handler in self.logger.handlers:
            handler.setFormatter(formatter)
            
    def debug(self, msg: str):
        self.logger.debug(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)
            
    def _default_formatter(self) -> logging.Formatter:
        # logging.Formatter("[%(levelname)s] %(message)s")
        return logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
        )
        
    def catch_external_logs(
        self,
        logger_name: str,
        prefix: str = "",
        formatter: Optional[logging.Formatter] = None
    ):
    
        ext_logger = logging.getLogger(logger_name)
        ext_logger.setLevel(logging.DEBUG)


class PipelineLogger(BaseLogger):
    SAM_PATTERN = re.compile(r"Speed:.*inference", re.IGNORECASE)
    
    def __new__(cls, name: str = "pipeline", level: int = 1):
        return super().__new__(cls, name, level)
    
    def __init__(self, name: str = "pipeline", level: int = 1):
        super().__init__(name=name, level=level)
        
        self._progess_bar: Optional[tqdm] = None
        self._current_image_id: Optional[str] = None
        
        self._filter_ultralytics_logs()
        
    def start_image(self, image_id: str, nb_tiles: int):
        self.end_image()
        
        self._current_image_id = image_id
        self._clear_console()
        
        self.logger.info(f"Processing image: {image_id}")
        
        self._progess_bar = tqdm(
            total=nb_tiles,
            desc=f"Tiles ({image_id})",
            unit="tile",
            leave=False,
        )
        
    def update(self, step: int = 1):
        if self._progess_bar:
            self._progess_bar.update(step)
    
    def end_image(self):
        if self._progess_bar:
            self._progess_bar.close()
            self._progess_bar = None
        
        self._current_image_id = None
        
    def _filter_ultralytics_logs(self):
        ul_logger = logging.getLogger("ultralytics")
        ul_logger.setLevel(logging.INFO)
        ul_logger.propagate = True
        ul_logger.handlers.clear()
        
        ul_logger.addHandler(UltralyticsInferenceHandler())
        ul_logger.propagate = True
    
    def _clear_console(self):
        if sys.stdout.isatty():
            print("\033[2J\033[H", end="")


class UltralyticsInferenceHandler(logging.Handler):
    SPEED_PATTERN = re.compile(r"Speed:.*inference", re.IGNORECASE)
    BATCH_PATTERN = re.compile(r"^\d+:\s+\d+x\d+", re.IGNORECASE)
    
    def emit(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        
        if self.SPEED_PATTERN.search(msg):
            return
        
        if self.BATCH_PATTERN.search(msg):
            return