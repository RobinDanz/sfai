import numpy as np
from dataclasses import dataclass

@dataclass
class Tile:
    """Dataclass representing a Tile.

    Attributes:
        image (np.ndarray): Numpy array representing the tile
        center (Tuple[int, int]): Center point of the tile in the source image
        coords (Tuple[int, int, int, int]): Coordinates of the tile in the source image
        width (int): Width of the tile
        height (int): Height of the tile
    """
    image: np.ndarray
    center: tuple[int, int]
    coords: tuple[int, int, int, int]
    width: int
    height: int

class ImageTiler:
    """Utility class to split an image into multiple tiles.

    Attributes:
        rows (int, optional): Number of rows. Defaults to 5.
        cols (int, optional): Number of columns. Defaults to 5.
        overlap (int, optional): Overlap between tiles in pixels. Defaults to 10.
    """
    def __init__(self, rows: int = 5, cols: int = 5, overlap: int = 10):
        self.rows = rows
        self.cols = cols
        self.overlap = overlap
        
    def split(self, image: np.ndarray) -> list[Tile]:
        """Returns a list of tiles generated from the input image.

        Args:
            image (np.ndarray):

        Returns:
            list[Tile]:
        """
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