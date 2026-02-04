from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from sfai.runners import TileResult
    
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
        
        offsets = []
        total_label_count = 0

        for t in tiles:
            offsets.append(total_label_count)
            total_label_count += t.ctx.metadata.get('label_count', 0)
        
        dsu = DSU(total_label_count)

        for t, offset in zip(tiles, offsets):
            tile = t.tile
            tile_mask = t.ctx.sam_mask
            x1, y1, x2, y2 = tile.coords
            h,w = tile_mask.shape
            
            tile_mask = np.where(tile_mask > 0, tile_mask + offset, 0)
            
            if x1 > 0:
                old = final_mask[y1:y2, x1]
                new = tile_mask[:, 0]
                for a, b in zip(old, new):
                    if a > 0 and b > 0:
                        dsu.union(a, b)
                    
            if x2 < W:
                old = final_mask[y1:y2, x2 - 1]
                new = tile_mask[:, w - 1]
                for a, b in zip(old, new):
                    if a > 0 and b > 0:
                        dsu.union(a, b)
                        
            if y1 > 0:
                old = final_mask[y1, x1:x2]
                new = tile_mask[0, :]
                for a, b in zip(old, new):
                    if a > 0 and b > 0:
                        dsu.union(a, b)
            
            if y2 < H:
                old = final_mask[y2 - 1, x1:x2]
                new = tile_mask[h - 1, :]
                for a, b in zip(old, new):
                    if a > 0 and b > 0:
                        dsu.union(a, b)

            final_mask[y1:y2, x1:x2] = np.maximum(
                final_mask[y1:y2, x1:x2],
                tile_mask
            )

            plt.imsave(f'split.png', final_mask, cmap='nipy_spectral', format='png', dpi=400)

        unique = np.unique(final_mask)
        unique = unique[unique > 0]

        label_map = np.arange(total_label_count + 1, dtype=np.uint16)

        for label in unique:
            label_map[label] = dsu.find(label)

        final_image = label_map[final_mask]
        
        return final_image