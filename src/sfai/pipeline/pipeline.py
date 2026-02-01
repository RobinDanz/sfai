from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from sfai.operators import Operator    
    from sfai.export import OutputHandler
    from sfai.data import ImageInfo

from sfai.pipeline import PipelineContext

class Pipeline:
    """Pipeline class used to execute the operators on an image.

    Args:
        operators (list[Operator]): List of operator to be executed on the image

    Attributes:
        operators (list[Operator]): List of operator to be executed on the image
    """
    def __init__(self, operators: list[Operator]):
        self.operators = operators
        
    def run(self, image: np.ndarray, image_info: ImageInfo, index: int, output_handler: OutputHandler) -> PipelineContext:
        """

        Args:
            image (np.ndarray): _description_
            image_info (ImageInfo): _description_
            index (int): _description_
            output_handler (OutputHandler): _description_

        Returns:
            PipelineContext: _description_
        """
        ctx = PipelineContext(
            index=index,
            image=image,
            image_info=image_info, 
            output_handler=output_handler, 
            metadata={}
        )
        
        for op in self.operators:
            ctx = op(ctx)
        return ctx