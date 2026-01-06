from soilfauna.operators import Operator
from soilfauna.pipeline import PipelineContext
from soilfauna.export import OutputHandler
from soilfauna.data import ImageInfo
import numpy as np

class Pipeline:
    def __init__(self, operators: list[Operator]):
        self.operators = operators
        
    def run(self, image: np.ndarray, image_info: ImageInfo, index: int, output_handler: OutputHandler) -> PipelineContext:
        
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