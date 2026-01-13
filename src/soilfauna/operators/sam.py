from soilfauna.operators import Operator, save_artifacts
from soilfauna.pipeline import PipelineContext
from pathlib import Path
import numpy as np

from typing import TYPE_CHECKING

def load_sam():
    try:
        from ultralytics import SAM
        
        return SAM
    except ImportError as e:
        raise RuntimeError(
            'Ultralytics is not installed.'
            'Install it by running pip install ".[sam]"'
        ) from e

def load_torch():
    try:
        import torch
        
        return torch
    except ImportError as e:
        raise RuntimeError(
            'PyTorch is not installed.'
            'Install it manually'
        )
        
if TYPE_CHECKING:
    import torch
    from ultralytics import SAM

torch = load_torch()        
SAM = load_sam()


class SAMSegmentation(Operator):
    """
    Objects segmentation using SAM
    """
    save_folder = 'sam'
    
    def __init__(self, model: Path | str, save: bool = False):
        self.model = SAM(model.absolute())
        self.device = self._get_device()
        self.save = save
    
    @save_artifacts
    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        tile_mask = np.zeros(ctx.image.shape[:2], dtype=np.uint16)
        points = self.merge_centers(ctx.points)
        label_count = 0
        
        if len(points) > 0:
            results = self.model.predict(ctx.image, points=points)
            object_masks = [mask.cpu().numpy().astype(np.uint8) for result in results for mask in result.masks.data]
            merged_object_masks, label_count = self.merge_masks(object_masks)
            
            for i, mask in enumerate(merged_object_masks):
                labeled_mask = mask.astype(np.uint16) * (i+1)
                tile_mask = np.maximum(tile_mask, labeled_mask)

        ctx.sam_mask = tile_mask
        ctx.metadata['label_count'] = label_count
        return ctx
    
    def _get_device(self):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def model_info(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.model.info()
    
    def merge_centers(self, centers, dist_thresh=20):
        centers = np.array(centers)
        used = np.zeros(len(centers), dtype=bool)
        merged = []

        for i in range(len(centers)):
            if used[i]:
                continue

            close = np.linalg.norm(centers - centers[i], axis=1) < dist_thresh
            used[close] = True

            merged_center = centers[close].mean(axis=0).astype(int)
            merged.append(merged_center.tolist())

        return merged
    
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
        
        return merged_masks, len(merged_masks)
    
    def result_image(self, ctx: PipelineContext):
        crop_subfolder = ctx.output_handler.generate_crop_subfodler(
            ctx.image_info.name,
            self.save_folder
        )
        
        save_path = crop_subfolder / f'{ctx.index}.jpg'
        
        plt_kwargs = {
            'cmap': 'nipy_spectral'
        }
        
        return ctx.sam_mask, save_path, plt_kwargs