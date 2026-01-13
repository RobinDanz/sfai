from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from collections import defaultdict

if TYPE_CHECKING:
    from soilfauna.runners import TileResult
    
class DSU:
    """_summary_
    """
    def __init__(self, n):
        """_summary_

        Args:
            n (_type_): _description_
        """
        self.parent = list(range(n + 1))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        pa, pb = self.find(a), self.find(b)
        if pa != pb:
            self.parent[pb] = pa
    

class MaskStitcher:
    """_summary_
    """
    def __init__(self):
        """_summary_
        """
        self.border_equiv = defaultdict(set)
    
    def stitch(self, tiles: list[TileResult], image_shape):
        """_summary_

        Args:
            tiles (list[TileResult]): _description_
            image_shape (_type_): _description_

        Returns:
            _type_: _description_
        """
        H, W = image_shape[:2]
        final_mask = np.zeros((H, W), dtype=np.uint16)
        
        total_label_count = 0
        
        for t in tiles:
            tile = t.tile
            tile_mask = t.ctx.sam_mask
            x1, y1, x2, y2 = tile.coords
            h,w = tile_mask.shape
            
            tile_mask[tile_mask > 0] += total_label_count
            
            if x1 > 0:
                old = final_mask[y1:y2, x1]
                new = tile_mask[:, 0]
                for a, b in zip(old, new):
                    if a > 0 and b > 0:
                        self.border_equiv[a].add(b)
                        self.border_equiv[b].add(a)
                    
            if x2 < W:
                old = final_mask[y1:y2, x2 - 1]
                new = tile_mask[:, w - 1]
                for a, b in zip(old, new):
                    if a > 0 and b > 0:
                        self.border_equiv[a].add(b)
                        self.border_equiv[b].add(a)
                        
            if y1 > 0:
                old = final_mask[y1, x1:x2]
                new = tile_mask[0, :]
                for a, b in zip(old, new):
                    if a > 0 and b > 0:
                        self.border_equiv[a].add(b)
                        self.border_equiv[b].add(a)
            
            if y2 < H:
                old = final_mask[y2 - 1, x1:x2]
                new = tile_mask[h - 1, :]
                for a, b in zip(old, new):
                    if a > 0 and b > 0:
                        self.border_equiv[a].add(b)
                        self.border_equiv[b].add(a)

            final_mask[y1:y2, x1:x2] = np.maximum(
                final_mask[y1:y2, x1:x2],
                tile_mask
            )
            
            total_label_count += t.ctx.metadata.get('label_count', 0)
        
        dsu = DSU(total_label_count)
    
        for a, neighbors in self.border_equiv.items():
            for b in neighbors:
                dsu.union(a, b)

        final_image = np.zeros_like(final_mask)

        unique = np.unique(final_mask)
        unique = unique[unique > 0]

        for label in unique:
            root = dsu.find(label)
            final_image[final_mask == label] = root
        
        return final_image