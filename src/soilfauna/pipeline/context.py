from dataclasses import dataclass
import numpy as np
from typing import Any, Dict
from soilfauna.export import OutputHandler
from soilfauna.data import ImageInfo

@dataclass
class PipelineContext:
    index: int
    image: np.ndarray
    image_info: ImageInfo
    clean_image: np.ndarray | None = None
    binary_mask: np.ndarray | None = None
    sam_mask: np.ndarray | None = None
    contours: Any | None = None
    points: np.ndarray | None = None
    output_handler: OutputHandler | None = None
    metadata: Dict[str, Any] = None