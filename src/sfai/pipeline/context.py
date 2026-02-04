from dataclasses import dataclass
import numpy as np
from typing import Any, Dict
from sfai.export import OutputHandler
from sfai.data import ImageInfo

@dataclass
class PipelineContext:
    """Pipeline context used to pass data through all the stages of the pipeline.

    Attributes:
        index (int): Index of the image.
        image (np.ndarray): Numpy array representing the image.
        image_info (ImageInfo): Object holding informations about the image.
        clean_image (np.ndarray): Image without background.
        binary_mask (np.ndarray): Image after being binarized.
        sam_mask (np.ndarray): Result of the SAM segmentation.
        contours (Any | None): Result of the contour detection
        points (np.ndarray): Center points of objects. Used as an input for SAM segmentation.
        output_handler (OutputHandler | None):
        metadata (Dict[str, Any]): Additional data
    """
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