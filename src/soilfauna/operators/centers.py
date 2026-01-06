from soilfauna.operators import Operator, save_artifacts
from soilfauna.pipeline import PipelineContext
import cv2

class CentersDetection(Operator):
    """
    Object center detection based on contours
    """
    
    save_folder = 'centers'
    
    def __init__(self, save: bool = False):
        self.save = save
    
    @save_artifacts
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
        