import cv2
import numpy as np

from typing import Iterator, Tuple
from shapely import MultiPolygon, Polygon, union_all
from shapely.geometry.polygon import orient
from sfai.export.data import CocoAnnotation


class MaskProcessor:
    """_summary_
    """
    def __init__(
        self,        
        min_area: int = 500,
    ):
        self.min_area = min_area
        
    def extract_masks(self, label_image: np.ndarray):
        """_summary_

        Args:
            label_image (np.ndarray): _description_

        Yields:
            _type_: _description_
        """
        object_ids = np.unique(label_image)
        object_ids = object_ids[object_ids != 0]
        
        for id in object_ids:
            mask = (label_image == id).astype(np.uint8) * 255
            
            yield mask
            
    def close_polygon(self, coords: np.ndarray) -> np.ndarray:
        """Check if a list of coordinates are closed (first == last).

        Args:
            coords (np.ndarray): _description_

        Returns:
            np.ndarray: closed list of coordinates
        """
        if not np.array_equal(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])
        
        return coords
    
    def validate_polygon(self, polygon: Polygon) -> Polygon | None:
        """Validates a Polygon using Shapely `.is_valid` method.

        Args:
            polygon (Polygon): _description_

        Returns:
            Polygon | None: _description_
        """
        if polygon.is_valid:
            return polygon
        
        repaired = polygon.buffer(0)
        
        if repaired.is_valid and not repaired.is_empty:
            return repaired
        
        return None
    
    def clean_mask(self, mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Cleans a maks by applying an OPENNING and a CLOSING right after.

        Args:
            mask (np.ndarray): _description_
            kernel_size (int, optional): Size of kernel used. Defaults to 3.

        Returns:
            np.ndarray: Cleaned mask
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return mask
    
    def smooth_contour(self, contour):
        epsilon = 0.001 * cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, epsilon, True)
            
    def mask_to_polygons(self, mask: np.ndarray, clean: bool = True):
        """Converts a mask to a polygon

        Args:
            mask (np.ndarray): _description_
            clean (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if clean:
            mask = self.clean_mask(mask, 5)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        polygons: list[Polygon] = []
        
        for i, cnt in enumerate(contours):

            if hierarchy[0][i][3] != -1:
                continue

            if cv2.contourArea(cnt) < self.min_area:
                continue

            cnt = self.smooth_contour(cnt)

            if cnt.shape[0] < 3:
                continue
            
            exterior = cnt.reshape(-1, 2)
            
            holes = []

            for j, h in enumerate(contours):
                if hierarchy[0][j][3] == i:
                    if cv2.contourArea(h) < self.min_area:
                        continue
                    h = self.smooth_contour(h)
                    if h.shape[0] >= 3:
                        holes.append(h.reshape(-1, 2))

            polygon = Polygon(exterior, holes)
                
            polygon = self.validate_polygon(polygon)
            if polygon is None:
                continue
            
            if polygon.area >= self.min_area:
                polygon = orient(polygon, sign=1.0)
                polygons.append(polygon)
                
        if not polygons:
            return []
        
        merged_polygons = union_all(polygons)
        
        if isinstance(merged_polygons, Polygon):
            merged_polygons = [merged_polygons]
        elif isinstance(merged_polygons, MultiPolygon):
            merged_polygons = list(merged_polygons.geoms)
            
        coco_polygons = []
        
        for poly in merged_polygons:
            coords = np.array(poly.exterior.coords)
            coords = self.close_polygon(coords)
            coco_polygons.append(coords.flatten().tolist())
            
        return coco_polygons
    
    @staticmethod
    def compute_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Calculates the bouding-box of a mask

        Args:
            mask (np.ndarray): Input mask

        Returns:
            Tuple[int, int, int, int]: The bouding-box: x, y, length, height
        """
        ys, xs = np.where(mask > 0)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        return [
            int(x_min),
            int(y_min),
            int(x_max - x_min),
            int(y_max - y_min),
        ]
        
    @staticmethod
    def compute_area(mask: np.ndarray) -> int:
        """Calculates the area in pixel of a mask

        Args:
            mask (np.ndarray): Input mask

        Returns:
            int: Number of nonzero pixel in the input mask
        """
        return int(np.count_nonzero(mask))
    
    def build_annotations(self, label_image: np.ndarray, image_id: int, category_id: int) -> Iterator[CocoAnnotation]:
        """_summary_

        Args:
            label_image (np.ndarray): _description_
            image_id (int): _description_
            category_id (int): _description_

        Yields:
            Iterator[CocoAnnotation]: _description_
        """
        for i, mask in enumerate(self.extract_masks(label_image), 1):
            polygons = self.mask_to_polygons(mask, clean=False)
            
            if not polygons:
                continue
            
            yield CocoAnnotation(
                id=i,
                image_id=image_id,
                category_id=category_id,
                segmentation=polygons,
                area=self.compute_area(mask),
                bbox=self.compute_bbox(mask)
            )