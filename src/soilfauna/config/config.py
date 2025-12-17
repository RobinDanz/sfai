from dataclasses import dataclass, field
from pathlib import Path
import os
import yaml

class DefaultConfig:
    """
    Config class holding default configuration values
    """
    ROOT_DIR = Path(__file__).parent.parent.parent.parent.as_posix()
    
    DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, 'models')
    DEFAULT_MODEL = os.path.join(DEFAULT_MODEL_PATH, 'sam2_b.pt')
    
    DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, 'results')
    
    DEFAULT_RUN_DIR = os.path.join(DEFAULT_OUTPUT_DIR, 'runs')
    DEFAULT_RUN_NAME = 'default'
    
    DEFAULT_OUTPUT_ANNOTATION_DIR_NAME = 'annotations'
    DEFAULT_OUTPUT_CROPS_DIR_NAME = 'crops'
    DEFAULT_OUTPUT_IMAGES_DIR_NAME = 'images'
    
@dataclass
class RunConfig:
    """
    Dataclass holding a run config
    """
    id: int = 1
    name: str = 'default'
    run_dir: str = field(init=False)
    annotations_dir: str = field(init=False)
    crops_dir: str = field(init=False)
    images_dir: str = field(init=False)
    
    def __post_init__(self):
        self.run_dir = os.path.join(DefaultConfig.DEFAULT_RUN_DIR, self.name, f'{self.id}')
        self.annotations_dir = os.path.join(self.run_dir, DefaultConfig.DEFAULT_OUTPUT_ANNOTATION_DIR_NAME)
        self.crops_dir = os.path.join(self.run_dir, DefaultConfig.DEFAULT_OUTPUT_CROPS_DIR_NAME)
        self.images_dir = os.path.join(self.run_dir, DefaultConfig.DEFAULT_OUTPUT_IMAGES_DIR_NAME)
        
class Config:
    def __init__():
        pass
            
    def load_from_file(self, path: str) -> dict:
        with open(path) as stream:
            try:
                print(yaml.safe_load(stream))
            except yaml.YAMLError as e:
                print(e)
    
    
    
    
    
    