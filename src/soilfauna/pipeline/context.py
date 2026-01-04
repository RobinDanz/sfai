from dataclasses import dataclass
import numpy as np
from typing import Any, Dict

@dataclass
class PipelineContext:
    image: np.ndarray
    clean_image: np.ndarray | None = None
    binary_mask: np.ndarray | None = None
    sam_mask: np.ndarray | None = None
    contours: Any | None = None
    points: np.ndarray | None = None
    metadata: Dict[str, Any] = None