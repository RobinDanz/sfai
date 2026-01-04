from soilfauna.operators import Operator
from soilfauna.pipeline import PipelineContext
import cv2
import numpy as np
import matplotlib.pyplot as plt

class BinaryTransform(Operator):
    def __init__(self):
        pass
    
    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        mask = (ctx.clean_image == [255, 255, 255]).all(axis=-1)
        binary = np.zeros_like(ctx.clean_image)
        binary[mask] = [255, 255, 255]
        
        binary = cv2.bitwise_not(binary)
            
        gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        binary = binary.astype(np.uint8)
        binary = cv2.morphologyEx(
            binary, 
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        )
        
        ctx.binary_mask = binary
        
        return ctx
        

        