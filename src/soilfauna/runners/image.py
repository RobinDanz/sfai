import numpy as np
from typing import List
from dataclasses import dataclass
import matplotlib.pyplot as plt
import cv2

from soilfauna.pipeline import Pipeline, PipelineContext
from soilfauna.data import ImageTiler, Tile
from soilfauna.stitch import MaskStitcher
from soilfauna.mask import MaskProcessor
from soilfauna.config import SegmentationConfig
from soilfauna.data import ImageInfo
from soilfauna.export.data import CocoAnnotation
from soilfauna.export import OutputHandler

from soilfauna.logging import PIPELINE_LOGGER


@dataclass
class TileResult:
    tile: Tile
    ctx: PipelineContext

class ImagePipelineRunner:
    """
    Pipeline runner for a single image.
    
    Handles image operations.
    """
    def __init__(self, tiler: ImageTiler, pipeline: Pipeline, config: SegmentationConfig):
        self.config = config
        self.tiler = tiler
        self.pipeline = pipeline
        
    def run(self, image_info: ImageInfo, image: np.ndarray, output_handler: OutputHandler) -> List[CocoAnnotation]:
        annotations: List[CocoAnnotation] = []
        
        stitcher = MaskStitcher()
        mask_processor = MaskProcessor()
        
        tile_results = TilePipelinRunner(
            self.tiler,
            self.pipeline
        ).run(image_info, image, output_handler)
        
        label_image = stitcher.stitch(results=tile_results, image_shape=image.shape[:2])
        
        for annotation in mask_processor.build_annotations(label_image, image_id=image_info.id, category_id=1):
            annotations.append(annotation)
            
        if self.config.save_final_images:
            path = output_handler.image_dir / image_info.name
            
            plt.imsave(f'{path}_labels.png', label_image, cmap='nipy_spectral', format='png', dpi=400)
            
            img = image.copy()
            
            for ann in annotations:
                seg = ann.segmentation
                
                polygons = [
                    np.array(shape, dtype=np.int32).reshape(-1, 1, 2)
                    for shape in seg
                ]
                
                cv2.polylines(img, polygons, isClosed=True, color=(0, 0, 255), thickness=3)
            
            plt.imsave(f'{path}_contours.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB), format='png', dpi=400)
                
            
        return annotations
        
class TilePipelinRunner:
    """
    Pipeline runner for tiles.
    
    Splits an image into tiles and runs operations on each tile.
    """
    def __init__(self, tiler: ImageTiler, pipeline: Pipeline):
        self.tiler = tiler
        self.pipeline = pipeline
        
    def run(self, image_info: ImageInfo, image: np.ndarray, output_handler: OutputHandler) -> List[TileResult]:
        tiles = self.tiler.split(image)
        results = []
        
        PIPELINE_LOGGER.start_image(image_info.file_name, nb_tiles=len(tiles))
        
        for index, tile in enumerate(tiles):
            ctx = self.pipeline.run(tile.image, image_info, index, output_handler)

            results.append(TileResult(
                tile=tile,
                ctx=ctx
            ))
            
            PIPELINE_LOGGER.update()
        
        PIPELINE_LOGGER.end_image()
            
        return results