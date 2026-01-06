import cv2
import numpy as np

from typing import Iterator, Dict, Any
from shapely import MultiPolygon, Polygon
from shapely.ops import unary_union
from soilfauna.export.data import CocoAnnotation


class MaskProcessor:
    def __init__(
        self, 
        simplify_tolerance: float = 1.0,
        min_area: int = 20,
    ):
        self.simplification_tolerance = simplify_tolerance
        self.min_area = min_area
        
    def extract_masks(self, label_image: np.ndarray):
        object_ids = np.unique(label_image)
        object_ids = object_ids[object_ids != 0]
        
        for id in object_ids:
            mask = (label_image == id).astype(np.uint8) * 255
            
            yield mask
            
    def close_polygon(self, coords: np.ndarray):
        if not np.array_equal(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])
        
        return coords
    
    def validate_polygon(self, polygon: Polygon) -> Polygon | None:
        if polygon.is_valid:
            return polygon
        
        repaired = polygon.buffer(0)
        
        if repaired.is_valid and not repaired.is_empty:
            return repaired
        
        return None
            
    def mask_to_polygons(self, mask: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        polygons: list[Polygon] = []
        
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                continue
            
            if cnt.shape[0] < 3:
                continue
            
            coords = cnt.reshape(-1, 2)
            
            polygon = Polygon(coords)
            
            polygon = self.validate_polygon(polygon)
            if polygon is None:
                continue
            
            polygon = polygon.simplify(
                self.simplification_tolerance,
                preserve_topology=True
            )
            
            if polygon.area >= self.min_area:
                polygons.append(polygon)
                
        if not polygons:
            return []
                
        merged_polygons = unary_union(polygons)
        
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
    def compute_bbox(mask: np.ndarray):
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
    def compute_area(mask: np.ndarray):
        return int(np.count_nonzero(mask))
    
    def build_annotations(self, label_image: np.ndarray, image_id: int, category_id: int) -> Iterator[CocoAnnotation]:
        for i, mask in enumerate(self.extract_masks(label_image), 1):
            polygons = self.mask_to_polygons(mask)
            
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