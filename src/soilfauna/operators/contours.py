from soilfauna.operators import Operator
from soilfauna.pipeline import PipelineContext
import numpy as np
import cv2

class ContourDetection(Operator):
    """
    Contour detection based on labelled image
    """
    def __init__(self):
        pass
    
    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        watershed_labels = ctx.metadata.get('labels', [])
        unique_labels = np.unique(watershed_labels)
        unique_labels = unique_labels[unique_labels != 0]
        contours = []
        
        for label in unique_labels:
            mask = (watershed_labels == label).astype(np.uint8) * 255
            conts, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            contours.append(conts)
            
        ctx.contours = contours
        
        return ctx