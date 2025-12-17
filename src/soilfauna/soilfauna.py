from soilfauna.cli import add_segment_parser

from soilfauna.config import DefaultConfig, RunConfig
from soilfauna.preprocess.operators import BackgroundRemoveHSVOperator, BinaryConvertOperator
from soilfauna.dataset import Dataset
import matplotlib.pyplot as plt
import cv2
import numpy as np
from ultralytics import SAM
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import scipy.ndimage as ndi
from collections import defaultdict

from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser(
    prog='soilfauna',
    description='Set of tools to handle image files.'
)

parser.add_argument(
    "-V", "--verbose",
    help="Verbose mode..",
    required=False
)

subparsers = parser.add_subparsers(
    title="subcommands",
    dest="command",
    required=True
)

add_segment_parser(subparsers)


ROOT_DIR = Path(__file__).parent.parent.parent.as_posix()

class DSU:
    def __init__(self, n):
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
            
def merge_centers(centers, dist_thresh=20):
    centers = np.array(centers)
    used = np.zeros(len(centers), dtype=bool)
    merged = []

    for i in range(len(centers)):
        if used[i]: 
            continue

        close = np.linalg.norm(centers - centers[i], axis=1) < dist_thresh
        used[close] = True

        merged_center = centers[close].mean(axis=0).astype(int)
        merged.append(merged_center.tolist())

    return merged

def generate_run_config(run_name=DefaultConfig.DEFAULT_RUN_NAME) -> RunConfig:
    """
    Generate a run configuration
    
    :param run_name: Description
    :return: Description
    :rtype: RunConfig
    """
    output_folder = os.path.join(DefaultConfig.DEFAULT_RUN_DIR, run_name)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    run_id = len(next(os.walk(output_folder))[1])
        
    run_config = RunConfig(id=run_id, name=run_name)
   
    os.makedirs(run_config.run_dir)
    os.makedirs(run_config.annotations_dir)
    os.makedirs(run_config.crops_dir)
    os.makedirs(run_config.images_dir)
    
    return run_config

def main():
    args = parser.parse_args()
    args.func(args)
    
def test():
    if not os.path.isdir(DefaultConfig.DEFAULT_MODEL_PATH):
        os.makedirs(DefaultConfig.DEFAULT_MODEL_PATH)
    
    if not os.path.isdir(DefaultConfig.DEFAULT_OUTPUT_DIR):
        os.makedirs(DefaultConfig.DEFAULT_OUTPUT_DIR)
    
    if not os.path.isdir(DefaultConfig.DEFAULT_RUN_DIR):
        os.makedirs(DefaultConfig.DEFAULT_RUN_DIR)
    
    run_config = generate_run_config()
    
    dataset = Dataset(data_path='/Users/robin/Pictures/soil-fauna-ai/250925/R04_B_margo_r3c5.jpg', check_subdir=True)
    sam = SAM(DefaultConfig.DEFAULT_MODEL)
    
    
    for data in dataset:
        data.load()
        result_label = 1
        H, W = data.full_height, data.full_width
        image_masks = np.zeros((H, W), dtype=np.uint16)
        border_equiv = defaultdict(set)
        for crop, bbox_centers, coords, raw in data.get_crops():
            ################################
            #       PREPROCESS
            ################################
            x1, y1, x2, y2 = coords
            no_bg_operator = BackgroundRemoveHSVOperator()
            no_bg = no_bg_operator.apply(crop)
            binary_operator = BinaryConvertOperator()
            mask = binary_operator.apply(no_bg)
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            binary = binary.astype(np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
            
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            dist_smooth = cv2.GaussianBlur(dist, (0,0), sigmaX=2)
            mask = np.zeros_like(binary, dtype=bool)
            
            local_max = peak_local_max(
                dist_smooth,
                min_distance=10,
                threshold_rel=0.05,
                labels=binary
            )
            
            mask[tuple(local_max.T)] = True
            
            markers, _ = ndi.label(mask)

            labels = watershed(-dist, markers, mask=binary)
            
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != 0]
            
            centers = []
            
            for lab in unique_labels:
                mask_label = (labels == lab).astype(np.uint8) * 255
                contours, hierarchy = cv2.findContours(
                    mask_label,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                for cnt in contours:
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = int(M["m10"]/M["m00"])
                        cy = int(M["m01"]/M["m00"])
                        centers.append([cx, cy])
                        
            ################################
            #           SAM 
            ################################           
            crop_mask_labeled = np.zeros((y2 - y1, x2 - x1), dtype=np.uint16)
            
            if len(centers) > 0:
                centers = merge_centers(centers=centers)
                results = sam.predict(crop, points=centers)
                sam_masks = [mask.cpu().numpy().astype(np.uint8) for result in results for mask in result.masks.data]
                sam_masks_sorted = sorted(sam_masks, key=lambda m: m.sum(), reverse=True)
                merged_masks = []
                
                for m in sam_masks_sorted:
                    merged = False
                    m_area = m.sum()
                    for i, existing in enumerate(merged_masks):
                        intersection = np.logical_and(existing, m).sum()
                        union = np.logical_or(existing, m).sum()
                        IoU = intersection / union
                        inclusion = intersection / m_area if m_area > 0 else 0
                        if IoU > 0.5 or inclusion > 0.9:
                            merged_masks[i] = np.logical_or(existing, m).astype(np.uint8)
                            merged = True
                            break
                    if not merged:
                        merged_masks.append(m)
                
                for region_mask in merged_masks:
                    labeled_mask = region_mask.astype(np.uint16) * result_label
                    crop_mask_labeled = np.maximum(crop_mask_labeled, labeled_mask)

                    result_label += 1
                        
            # fig, axs = plt.subplots(2, 2)
            # axs[0][0].imshow(crop)
            # axs[0][1].imshow(dist_smooth)
            # axs[1][0].imshow(markers, cmap='gray')
            
            # axs[1][1].imshow(labels, cmap="nipy_spectral")
            
            # plt.show()
            
            ################################
            #       POSTPROCESS
            ################################
            
            h,w = crop_mask_labeled.shape
            
            if x1 > 0:
                old = image_masks[y1:y2, x1]
                new = crop_mask_labeled[:, 0]
                for a, b in zip(old, new):
                    if a > 0 and b > 0:
                        border_equiv[a].add(b)
                        border_equiv[b].add(a)
                        
            if x2 < W:
                old = image_masks[y1:y2, x2 - 1]
                new = crop_mask_labeled[:, w - 1]
                for a, b in zip(old, new):
                    if a > 0 and b > 0:
                        border_equiv[a].add(b)
                        border_equiv[b].add(a)
                        
            if y1 > 0:
                old = image_masks[y1, x1:x2]
                new = crop_mask_labeled[0, :]
                for a, b in zip(old, new):
                    if a > 0 and b > 0:
                        border_equiv[a].add(b)
                        border_equiv[b].add(a)
            
            if y2 < H:
                old = image_masks[y2 - 1, x1:x2]
                new = crop_mask_labeled[h - 1, :]
                for a, b in zip(old, new):
                    if a > 0 and b > 0:
                        border_equiv[a].add(b)
                        border_equiv[b].add(a)  
            
            image_masks[y1:y2, x1:x2] = np.maximum(
                image_masks[y1:y2, x1:x2],
                crop_mask_labeled
            )
            
        #####################################
        #       FINAL IMAGE BUILD
        #####################################
        
        dsu = DSU(result_label)
        
        for a, neighbors in border_equiv.items():
            for b in neighbors:
                dsu.union(a, b)
                
        final_image = np.zeros_like(image_masks)

        unique = np.unique(image_masks)
        unique = unique[unique > 0]

        for label in unique:
            root = dsu.find(label)
            final_image[image_masks == label] = root
        
        ###########################################
        #               DEBUG DISPLAY
        ###########################################
        debug_display = final_image.copy()

        nonzero = debug_display > 0
        debug_display[nonzero] = (debug_display[nonzero] % 254) + 1

        # plt.figure(figsize=(10, 10))
        # plt.imshow(debug_display, cmap="nipy_spectral")
        # plt.axis("off")
        # plt.show()
        
        
        unique_labels_final = np.unique(final_image)
        unique_labels_final = unique_labels_final[unique_labels_final != 0]
        
        for lab in unique_labels_final:
            mask_label = (final_image == lab).astype(np.uint8) * 255
            contours, hierarchy = cv2.findContours(
                mask_label,
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_TC89_L1
            )
            
            cv2.drawContours(data.image, contours, -1, (0,255,0), 3)
            
        path = os.path.join(run_config.images_dir, data.image_path.name)
        plt.imsave(path, cv2.cvtColor(data.image, cv2.COLOR_BGR2RGB))
                   
                   
    # stitch.test2()
    # run_config = generate_run_config()
    # print(run_config)
    # gen_mosaic('/Users/robin/Pictures/soil-fauna-ai/full_images/A02-E/')
    # gen_mosaic('/Users/robin/Pictures/soil-fauna-ai/full_images/F02-C/')
    # segment(DefaultConfig.DEFAULT_MODEL, dataset_path, None, run_config.crops_dir, run_config.annotations_dir, run_config.images_dir)
    
if __name__ == '__main__':
    main()
    
    
        
    
    



