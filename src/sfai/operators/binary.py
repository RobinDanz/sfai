from sfai.operators import Operator, save_artifacts
from sfai.pipeline import PipelineContext
import cv2
import numpy as np

class BinaryTransform(Operator):
    
    save_folder = 'binary'
    
    def __init__(self, save: bool = False):
        self.save = save
    
    @save_artifacts
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
    
    def result_image(self, ctx: PipelineContext):
        crop_subfolder = ctx.output_handler.generate_crop_subfodler(
            ctx.image_info.name,
            self.save_folder
        )
        
        save_path = crop_subfolder / f'{ctx.index}.jpg'
        
        plt_kwargs = {
            'cmap': 'gray'
        }
        
        return ctx.binary_mask, save_path, plt_kwargs
        

        