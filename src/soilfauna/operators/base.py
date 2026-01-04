from abc import ABC, abstractmethod
from soilfauna.pipeline import PipelineContext

class Operator(ABC):
    """
    Base abstract class for operators
    """
    @abstractmethod
    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        pass
    
def save_artifacts(*artifacts):
    """"""
    

    
