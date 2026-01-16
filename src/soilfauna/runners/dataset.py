from __future__ import annotations
from typing import TYPE_CHECKING, List

from soilfauna.data import Dataset
from soilfauna.export import JsonlBufferedWriter, CocoWriter
from soilfauna.export.data import CocoImage, DEFAULT_CATEGORY
from soilfauna.logging import LOGGER
from soilfauna.runners.image import ImagePipelineRunner   

if TYPE_CHECKING:
    from soilfauna.export import OutputHandler
    from soilfauna.config import SegmentationConfig
    from soilfauna.operators import Operator 
    from soilfauna.export.data import CocoCategory

class DatasetRunner:
    def __init__(self, dataset: Dataset, operators: List[Operator], output_handler: OutputHandler, config: SegmentationConfig):
        self.operators = operators
        self.config = config
        self.dataset = dataset
        self.output_handler = output_handler

        self.image_runner = ImagePipelineRunner(
            operators=operators,
            config=self.config
        )

    def run(self):
        categories: List[CocoCategory] = [DEFAULT_CATEGORY]
        
        annotation_out = self.output_handler.annotation_dir / 'result.json'
        
        coco_writer = CocoWriter(
            self.output_handler.images_jsonl_path,
            self.output_handler.annotations_jsonl_path,
            categories,
            annotation_out
        )
        
        images_writer = JsonlBufferedWriter(self.output_handler.images_jsonl_path)
        annotations_writer = JsonlBufferedWriter(self.output_handler.annotations_jsonl_path)
        
        for i, (image_info, image) in enumerate(self.dataset, 1):
            LOGGER.info(f"Image: {i}/{self.dataset.length}")
            coco_img = CocoImage(
                id=image_info.id,
                width=image_info.width,
                height=image_info.height,
                file_name=image_info.file_name
            )
            
            images_writer.write(coco_img)
            
            annotations = self.image_runner.run(image_info, image, self.output_handler)
            annotations_writer.write_list(annotations)
            
        images_writer.close()
        annotations_writer.close()
        
        coco_writer.write()