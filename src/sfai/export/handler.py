import os
from pathlib import Path
from sfai.config import default


class OutputHandler:
    def __init__(self, base_dir: Path = default.DEFAULT_OUTPUT_DIR, subname: str | None = None):
        self.base_dir = base_dir
        
        if subname:
            self.base_dir = self.base_dir / subname
        
    def generate_output_folders(self):
        self.annotation_dir.mkdir(parents=True, exist_ok=True)
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_crop_subfodler(self, image_name: str, subfolder: str):
        crop_subfolder = self.crop_dir / image_name / subfolder
        crop_subfolder.mkdir(parents=True, exist_ok=True)
        
        return crop_subfolder
    
    @property
    def annotation_dir(self) -> Path:
        return Path(os.path.join(self.base_dir, 'annotations'))
    
    @property
    def crop_dir(self) -> Path:
        return Path(os.path.join(self.base_dir, 'crops'))
    
    @property
    def image_dir(self) -> Path:
        return Path(os.path.join(self.base_dir, 'images'))
    
    @property
    def images_jsonl_path(self) -> Path:
        return Path(os.path.join(self.annotation_dir, 'images.jsonl'))
    
    @property
    def annotations_jsonl_path(self) -> Path:
        return Path(os.path.join(self.annotation_dir, 'annotations.jsonl'))
    
    @property
    def categories_jsonl_path(self) -> Path:
        return Path(os.path.join(self.annotation_dir, 'categories.jsonl'))
        