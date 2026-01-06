from soilfauna.operators import Operator, save_artifacts
from soilfauna.pipeline import PipelineContext
import cv2
import numpy as np
from typing import Tuple
from pathlib import Path

class HSVBackgroundRemoval(Operator):
    """
    Removes background based on HSV colors
    """
    
    save_folder = 'no_background'
    
    def __init__(self, lower_bound=[90,  40,  40], upper_bound=[145, 255, 255], save: bool = False):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.save = save
    
    @save_artifacts
    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        hsv = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array([90,  40,  40])
        upper_blue = np.array([145, 255, 255])
        
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        kernel = np.ones((3,3), np.uint8)
        mask_clean = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=1)
        
        mask_clean = cv2.GaussianBlur(mask_clean, (5,5), 0)
        
        cleaned = ctx.image.copy()
        cleaned[mask_clean > 0] = [255, 255, 255]
        
        ctx.clean_image = cleaned
        
        return ctx
    
    def result_image(self, ctx: PipelineContext) -> Tuple[np.ndarray, Path]:
        crop_subfolder = ctx.output_handler.generate_crop_subfodler(
            ctx.image_info.name,
            self.save_folder
        )
        
        save_path = crop_subfolder / f'{ctx.index}.jpg'
        
        return cv2.cvtColor(ctx.clean_image, cv2.COLOR_BGR2RGB), save_path
        