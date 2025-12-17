from pathlib import Path
import cv2
from typing import List
        
class Dataset:
    """
    Dataset loader
    """
    def __init__(self, data_path, preload=True, file_pattern='*.jpg', check_subdir=True,):
        self.data_path = data_path

        self.data: List[ImageData] = []

        if preload:
            self.find(file_pattern=file_pattern, check_subdir=check_subdir)

    def find(self, file_pattern='*.jpg', check_subdir=False):
        data_path = Path(self.data_path)
        
        if data_path.is_file():
            self.append(
                ImageData(image_path=data_path)
            )
            return
        
        search_pattern = file_pattern
        
        if check_subdir:
            search_pattern = f'**/{file_pattern}'
        
        files = list(data_path.glob(search_pattern))
        for image in files:
            self.append(
                ImageData(image_path=image)
            )
            
    def __getitem__(self, index):
        return self.data[index]
    
    def __setitem__(self, index, value):
        self.data[index] = value

    def __delitem__(self, index):
        del self.data[index]

    def __iter__(self):
        return iter(self.data)
    
    def append(self, value):
        self.data.append(value)

class ImageData:
    """
    Object holding image
    """
    def __init__(self, image_path: Path, crops_padding=10):
        self.image_path = image_path
        self.padding = crops_padding

        self.image = None
        self.loaded = False

    def load(self):
        if not self.loaded:
            self.image = self.read_image(self.image_path)
            self.full_height, self.full_width = self.image.shape[:2]

            self.loaded = True

        return self.image

    def read_image(self, image_path):
        return cv2.imread(image_path)
    
    def get_crops(self, rows=5, cols=5):
        if not self.loaded and not self.metadata:
            return []
        
        crops = []

        height, width, _ = self.image.shape

        crop_y = height//rows
        crop_x = width//cols

        crops = []

        for y in range(0, height, crop_y):
            for x in range(0, width, crop_x):
                x1 = max(x-self.padding, 0)
                y1 = max(y-self.padding, 0)
                x2 = min(x + crop_x + self.padding, width)
                y2 = min(y + crop_y + self.padding, height)
                crops.append(
                    (
                        self.image[y1:y2, x1:x2],
                        ((x2//2), (y2//2)),
                        (x1, y1, x2, y2),
                        self.image[y1:y2, x1:x2]
                    )
                )
        return crops
        
    def __str__(self):
        return f'Image: {str(self.image_path)}'