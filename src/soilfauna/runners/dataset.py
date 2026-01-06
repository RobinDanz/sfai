from __future__ import annotations
from typing import TYPE_CHECKING, List

from soilfauna.data import Dataset
from soilfauna.export import JsonlBufferedWriter, CocoWriter, OutputHandler
from soilfauna.export.data import CocoImage, CocoCategory, DEFAULT_CATEGORY
from soilfauna.logging import GLOBAL_LOGGER

if TYPE_CHECKING:
    from soilfauna.runners import ImagePipelineRunner    

class DatasetRunner:
    def __init__(self, dataset: Dataset, image_runner: ImagePipelineRunner, output_handler: OutputHandler):
        self.dataset = dataset
        self.image_runner = image_runner
        self.output_handler = output_handler

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
            GLOBAL_LOGGER.info(f"Image: {i}/{self.dataset.length}")
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