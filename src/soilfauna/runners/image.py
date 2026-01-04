import numpy as np
from typing import List
from dataclasses import dataclass

from soilfauna.pipeline import Pipeline, PipelineContext
from soilfauna.data import ImageTiler, Tile
from soilfauna.stitch import MaskStitcher
from soilfauna.mask import MaskProcessor
from soilfauna.config import SegmentationConfig
from soilfauna.data import ImageInfo
from soilfauna.export.data import CocoAnnotation


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
        
    def run(self, image_info: ImageInfo, image: np.ndarray) -> List[CocoAnnotation]:
        annotations: List[CocoAnnotation] = []
        
        stitcher = MaskStitcher()
        mask_processor = MaskProcessor()
        
        tile_results = TilePipelinRunner(
            self.tiler,
            self.pipeline
        ).run(image)
        
        label_image = stitcher.stitch(results=tile_results, image_shape=image.shape[:2])
        
        for annotation in mask_processor.build_annotations(label_image, image_id=image_info.id, category_id=1):
            annotations.append(annotation)
            
        return annotations
        
        
class TilePipelinRunner:
    """
    Pipeline runner for tiles.
    
    Splits an image into tiles and handles operations on each tile.
    """
    def __init__(self, tiler: ImageTiler, pipeline: Pipeline):
        self.tiler = tiler
        self.pipeline = pipeline
        
    def run(self, image: np.ndarray) -> List[TileResult]:
        tiles = self.tiler.split(image)
        results = []
        
        for tile in tiles:
            ctx = self.pipeline.run(tile.image)

            results.append(TileResult(
                tile=tile,
                ctx=ctx
            ))
            
        return results