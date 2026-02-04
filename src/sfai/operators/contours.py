from sfai.operators import Operator, save_artifacts
from sfai.pipeline import PipelineContext
import numpy as np
import cv2

class ContourDetection(Operator):
    """
    NOT USED ANYMORE
    
    Contour detection operator.

    Args:
        save (bool, optional): Save artifact or not. Defaults to False.
    """
    save_folder = 'contours'
    
    def __init__(self, save: bool = False):
        self.save = save
    
    @save_artifacts
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
    
    def result_image(self, ctx: PipelineContext):
        crop_subfolder = ctx.output_handler.generate_crop_subfodler(
            ctx.image_info.name,
            self.save_folder
        )
        
        save_path = crop_subfolder / f'{ctx.index}.jpg'
        
        img = ctx.clean_image.copy()
        
        for conts in ctx.contours:
            cv2.drawContours(img, conts, -1, (0, 0, 255), 3)
        
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), save_path