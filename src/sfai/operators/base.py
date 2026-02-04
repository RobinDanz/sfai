from abc import ABC, abstractmethod
from functools import wraps
from sfai.pipeline import PipelineContext
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
from pathlib import Path

class Operator(ABC):
    """
    Base abstract class for Operators.

    The operations on image should be done in the `__call__` method.
    """
    @abstractmethod
    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        """Applies the operator. Can read and write to the PipelineContext

        Args:
            ctx (PipelineContext): 

        Returns:
            PipelineContext: 
        """
        pass
    
    @abstractmethod
    def result_image(self, ctx) -> Tuple[np.ndarray, Path, Dict[str, Any]]:
        """Returns the result image after the operator is applied. Debug purpose.

        Args:
            ctx PipelineContext:

        Returns:
            T (Tuple[np.ndarray, Path, Dict[str, Any]]): Tuple that contains image, output path and optionnal pyplot `imsave` parameters
        """
        pass
    
def save_artifacts(method):
    """
    Decorator to save artifact image
    """
    @wraps(method)
    def wrapper(self: Operator, ctx: PipelineContext, *args, **kwargs):
        result = method(self, ctx)
        
        if self.save:
            img, save_path, *rest = self.result_image(ctx)
            
            kwargs = rest[0] if rest else {}
            
            plt.imsave(save_path, img, dpi=200, **kwargs)
        
        return result
    return wrapper
    

    
