from __future__ import annotations
from typing import TYPE_CHECKING, List

from soilfauna.data import Dataset
from soilfauna.export import JsonlWriter, CocoWriter, OutputHandler
from soilfauna.export.data import CocoImage, CocoAnnotation, CocoCategory, DEFAULT_CATEGORY

if TYPE_CHECKING:
    from soilfauna.runners import ImagePipelineRunner    

class DatasetRunner:
    def __init__(self, dataset: Dataset, image_runner: ImagePipelineRunner, output: OutputHandler):
        self.dataset = dataset
        self.image_runner = image_runner
        self.output = output

    def run(self):
        images: List[CocoImage] = []
        annotations: List[CocoAnnotation] = []
        categories: List[CocoCategory] = [DEFAULT_CATEGORY]
        
        annotation_out = self.output.annotation_dir / 'result.json'
        
        coco_writer = CocoWriter(
            self.output.images_jsonl_path,
            self.output.annotations_jsonl_path,
            categories,
            annotation_out
        )
        
        images_writer = JsonlWriter(self.output.images_jsonl_path)
        annotations_writer = JsonlWriter(self.output.annotations_jsonl_path)
        
        for image_info, image in self.dataset:
            annotations += self.image_runner.run(image_info, image)
            images.append(
                CocoImage(
                    id=image_info.id,
                    width=image_info.width,
                    height=image_info.height,
                    file_name=image_info.file_name
                )
            )
            
        images_writer.write_list(images)
        annotations_writer.write_list(annotations)
            
        annotations_writer.close()
        images_writer.close()
        
        coco_writer.write()