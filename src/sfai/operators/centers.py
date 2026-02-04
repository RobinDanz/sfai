from sfai.operators import Operator, save_artifacts
from sfai.pipeline import PipelineContext
from scipy import ndimage
import numpy as np
import cv2

class CentersDetection(Operator):
    """Detects objects center in labelled mask

    Args:
        save (bool, optional): Save artifact or not. Defaults to False.
    """
    save_folder = 'centers'
    
    def __init__(self, save: bool = False):
        self.save = save
    
    @save_artifacts
    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        labels = ctx.metadata['labels']

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]

        centers = ndimage.center_of_mass(
            np.ones_like(labels),
            labels,
            unique_labels
        )

        centers_xy = [(int(c[1]), int(c[0])) for c in centers]

        ctx.points = centers_xy

        return ctx
    
    def result_image(self, ctx: PipelineContext):
        crop_subfolder = ctx.output_handler.generate_crop_subfodler(
            ctx.image_info.name,
            self.save_folder
        )
        
        save_path = crop_subfolder / f'{ctx.index}.jpg'
        
        img = ctx.clean_image.copy()
        
        for center in ctx.points:
            cv2.circle(img, center, 1, (0, 0, 255), 4)
        
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), save_path
        