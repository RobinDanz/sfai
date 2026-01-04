import json
from pathlib import Path
from typing import List, Dict, Any
from soilfauna.export.data import CocoAnnotation, CocoCategory, CocoImage, CocoData

class JsonlWriter:
    """
    Writer for jsonl files
    """
    def __init__(self, path: str):
        self.file = open(path, 'a')
        
    def write(self, object: CocoData):
        """
        Writes a line into the jsonl file
        """
        self.file.write(json.dumps(object.to_dict()))
        self.file.write('\n')
        
    def write_list(self, objects: List[CocoData]):
        for obj in objects:
            self.write(obj)
            
    def close(self):
        """
        Closes the file.
        """
        self.file.close()
        
class CocoWriter:
    def __init__(
        self,
        images_jsonl: Path,
        annotations_jsonl: Path,
        categories: List[CocoCategory],
        output_path: Path
    ):
        self.images_jsonl = images_jsonl
        self.annotations_jsonl = annotations_jsonl
        self.categories = categories
        self.output_path = output_path
        
    def _read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        data = []
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def _load_images(self) -> List[CocoImage]:
        raw_images = self._read_jsonl(self.images_jsonl)
        
        return [CocoImage.from_dict(data) for data in raw_images]
        
    def _load_annotations(self) -> List[CocoAnnotation]:
        raw_annotations = self._read_jsonl(self.annotations_jsonl)
        
        return [CocoAnnotation.from_dict(data) for data in raw_annotations]
    
    def build_coco(self) -> Dict[str, Any]:
        images = self._load_images()
        annotations = self._load_annotations()
        
        coco = {
            'images': [img.to_dict() for img in images],
            'annotations': [ann.to_dict() for ann in annotations],
            'categories': [cat.to_dict() for cat in self.categories]
        }
        
        return coco
        
    def write(self) -> None:
        coco = self.build_coco()
        
        with self.output_path.open('x', encoding='utf-8') as f:
            json.dump(coco, f, indent=2)