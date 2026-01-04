from soilfauna.operators import Operator
from soilfauna.pipeline import PipelineContext
import cv2

class CentersDetection(Operator):
    """
    Object center detection based on contours
    """
    def __init__(self):
        pass
    
    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        centers = []
        
        for contour in ctx.contours:
            for i in contour:
                M = cv2.moments(i)
                if M["m00"] > 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    centers.append([cx, cy])
                
        ctx.points = centers
        
        return ctx
        