import numpy as np
from dataclasses import dataclass

@dataclass
class Tile:
    image: np.ndarray
    center: tuple[int, int]
    coords: tuple[int, int, int, int]
    width: int
    height: int

class ImageTiler:
    def __init__(self, rows: int = 5, cols: int = 5, overlap: int = 10):
        self.rows = rows
        self.cols = cols
        self.overlap = overlap
        
    def split(self, image: np.ndarray) -> list[Tile]:
        h, w = image.shape[:2]
        
        tiles = []

        tile_y = h//self.rows
        tile_x = w//self.cols

        for y in range(0, h, tile_y):
            for x in range(0, w, tile_x):
                x1 = max(x-self.overlap, 0)
                y1 = max(y-self.overlap, 0)
                x2 = min(x + tile_x + self.overlap, w)
                y2 = min(y + tile_y + self.overlap, h)
                tile = image[y1:y2, x1:x2]
                center = ((x2//2), (y2//2))
                coords = (x1, y1, x2, y2)
                tiles.append(
                    Tile(
                        image=tile,
                        center=center,
                        coords=coords,
                        width=tile.shape[1],
                        height=tile.shape[0]
                    )
                )
                
        return tiles