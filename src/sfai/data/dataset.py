from abc import ABC, abstractmethod
from pathlib import Path
import cv2
from dataclasses import dataclass
from typing import List

@dataclass
class ImageInfo:
    """
    Dataclass holding image informations
    """
    id: int
    name: str
    file_name: str
    path: Path
    width: int
    height: int

class Dataset(ABC):
    """
    Base abstract class for datasets.
    """
    @abstractmethod
    def __iter__(self):
        pass
    
    @property
    @abstractmethod
    def length(self):
        pass
    
class ImageFolderDataset(Dataset):
    """
    Dataset implementation for an image folder

    Args:
        root (Path): Path to the folder
        extensions (List[str], optional): Specific extensions to look for. Default is ['.jpg', '.jpeg', '.png']
    
    Attributes:
        root (Path): Base directory
        extensions (List[str], optional): Specific extensions to look for. Default is ['.jpg', '.jpeg', '.png']
    """
    def __init__(self, root: Path, extensions=['.jpg', '.jpeg', '.png']):
        self.root = root
        self.paths = []

        for p in root.iterdir():
            if p.suffix.lower() in extensions:
                self.paths.append(p)
        
    def __iter__(self):
        """
        Iterates over the dataset

        Yields:
            (Tuple[ImageInfo, np.ndarray]): _description_
        """
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
    """
    Dataset implementation for a single image.

    Args:
        path (Path): Path to the image
    
    Attributes:
        root (Path): Root directory
        paths (List[Path]): List of images (contains a single image by definition)
    """
    def __init__(self, path: Path):
        self.root = path.parent
        self.paths = []
        
        if path.is_file() and path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            self.paths.append(path)
            
    def __iter__(self):
        """
        Iterates over the dataset

        Yields:
            (Tuple[ImageInfo, np.ndarray]): _description_
        """
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
        """
        int: Number of files in the dataset
        """
        return len(self.paths)
            
def generate_datasets(datasets: List[Path]) -> List[Dataset]:
    """
    Utility method to generate datasets from a list of path
    
    Args:
        datasets (List[Path]): List of path to images or folders
    
    Returns:
        (List[Dataset]): List of Dataset objects.
    """
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
            
            
                