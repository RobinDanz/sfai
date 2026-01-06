from abc import ABC, abstractmethod
from pathlib import Path
import cv2
from dataclasses import dataclass
from typing import List
import os

from soilfauna.config import UserConfig

@dataclass
class ImageInfo:
    id: int
    name: str
    file_name: str
    path: Path
    width: int
    height: int

class Dataset(ABC):
    """
    Base abstract class for datasets
    """
    @abstractmethod
    def __iter__(self):
        """
        Yields (image_path, image_array)
        """
        pass
    
    @property
    @abstractmethod
    def length(self):
        """
        Returns file count
        """
        pass
    
class ImageFolderDataset(Dataset):
    def __init__(self, root: Path, extensions=['.jpg', '.jpeg', '.png']):
        self.root = root
        self.paths = []

        for p in root.iterdir():
            if p.suffix.lower() in extensions:
                self.paths.append(p)
        
    def __iter__(self):
        for id, path in enumerate(self.paths, 1):
            img = cv2.imread(str(path))
            
            info = ImageInfo(
                id=id,
                name=path.stem,
                file_name=path.name,
                path=path,
                height=img.shape[0],
                width=img.shape[1]
            )
            
            yield info, img
    
    @property
    def length(self):
        return len(self.paths)
            
class SingleImageDataset(Dataset):
    def __init__(self, path: Path):
        self.root = path.parent
        self.paths = []
        
        if path.is_file() and path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            self.paths.append(path)
            
    def __iter__(self):
        for id, path in enumerate(self.paths, 1):
            img = cv2.imread(str(path))
            
            info = ImageInfo(
                id=id,
                name=path.stem,
                file_name=path.name,
                path=path,
                height=img.shape[0],
                width=img.shape[1]
            )
            
            yield info, img
    
    @property
    def length(self):
        return len(self.paths)
            
def generate_datasets(datasets: List[Path]) -> List[Dataset]:
    out = []
    
    for path in datasets:
        if path.is_file():
            out.append(
                SingleImageDataset(path)
            )
        else:
            out.append(
                ImageFolderDataset(path)
            )
            
    return out
            
            
                