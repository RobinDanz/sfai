from pathlib import Path
from ultralytics.models import SAM
import torch
import numpy as np

class SAMSegmenter:
    def __init__(self, model: Path):
        self.model = SAM(model.absolute())
        self.device = self.get_device()
        
    def get_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def model_info(self):
        return self.model.info()
    
    def predict(self, image, points=None):
        """
        Run SAM prediction on image.
        
        Returns masks sorted by size
        """
        results = self.model.predict(image, points=points)
        masks = [mask.cpu().numpy().astype(np.uint8) for result in results for mask in result.masks.data]
    
        return masks
    
    def merge_masks(self, masks, IoU_thresh=0.5, inclusion_thresh=0.9):
        """
        Merges masks together
        """
        size_sorted_masks = sorted(masks, key=lambda m: m.sum(), reverse=True)
        
        merged_masks = []
        
        for mask in size_sorted_masks:
            merged = False
            mask_area = mask.sum()
            
            for i, existing in enumerate(merged_masks):
                intersection = np.logical_and(existing, mask).sum()
                union = np.logical_or(existing, mask).sum()
                IoU = intersection / union
                inclusion = intersection / mask_area if mask_area > 0 else 0
                
                if IoU > IoU_thresh or inclusion > inclusion_thresh:
                    merged_masks[i] = np.logical_or(existing, mask).astype(np.uint8)
                    merged = True
                    break
            
            if not merged:
                merged_masks.append(mask)
                
        return merged_masks
        
        