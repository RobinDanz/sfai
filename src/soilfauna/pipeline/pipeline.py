from soilfauna.operators import Operator
from soilfauna.pipeline import PipelineContext
import numpy as np

class Pipeline:
    def __init__(self, operators: list[Operator]):
        self.operators = operators
        
    def run(self, image: np.ndarray) -> PipelineContext:
        ctx = PipelineContext(image=image, metadata={})
        for op in self.operators:
            ctx = op(ctx)
        return ctx