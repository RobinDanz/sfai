from soilfauna.operators import Operator
from soilfauna.pipeline import PipelineContext
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import scipy.ndimage as ndi

class WatershedSegmentation(Operator):
    """
    Watershed segmentation operator
    """
    def __init__(self):
        pass
    
    def __call__(self, ctx: PipelineContext):
        distance = cv2.distanceTransform(ctx.binary_mask, cv2.DIST_L2, 5)
        distance_smooth = cv2.GaussianBlur(distance, (0,0), sigmaX=2)
        mask = np.zeros_like(ctx.binary_mask, dtype=bool)
        
        local_max = peak_local_max(
            distance_smooth,
            min_distance=10,
            threshold_rel=0.05,
            labels=ctx.binary_mask
        )
        
        mask[tuple(local_max.T)] = True
        
        markers, _ = ndi.label(mask)

        labels = watershed(-distance, markers, mask=ctx.binary_mask)
        
        ctx.metadata['labels'] = labels

        return ctx
        