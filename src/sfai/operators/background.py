from sfai.operators import Operator, save_artifacts
from sfai.pipeline import PipelineContext
import cv2
import numpy as np
from typing import Tuple
from pathlib import Path

class HSVBackgroundRemoval(Operator):
    """
    Removes background based on HSV colors.

    Args:
        lower_bound (Tuple[int, int, int], optional): HSV lower bound. Defaults to [90,  40,  40].
        upper_bound (Tuple[int, int, int], optional): HSV upper bound. Defaults to [145, 255, 255].
        save (bool, optional): If set to true, a result image is saved on disk. Defaults to False.

    Attributes:
        lower_bound (Tuple[int, int, int]): HSV lower bound.
        upper_bound (Tuple[int, int, int]): HSV upper bound.
        save (bool): If result image should be saved on disk or not.
    """
    
    save_folder = 'no_background'
    
    def __init__(self, lower_bound: Tuple[int, int, int] = [90,  40,  40], upper_bound: Tuple[int, int, int] = [145, 255, 255], save: bool = False):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.save = save
    
    @save_artifacts
    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        hsv = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array(self.lower_bound)
        upper_blue = np.array(self.upper_bound)
        
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
        