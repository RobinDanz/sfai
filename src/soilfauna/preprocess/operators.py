from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import cv2


class BaseOperator:
    def __init__(self):
        pass
    
    def apply(self, image):
        return image
        
class BackgroundRemoveKMeansOperator(BaseOperator):
    def __init__(self, n_clusters=3):
        super().__init__()
        
        self.n_cluster = n_clusters
        
    def get_dominant_colors(self, image):
        img = Image.fromarray(image)
        resized = img.resize((300, 300))
        
        img_np = np.array(resized)
        pixels = img_np.reshape((-1, 3))

        kmeans = KMeans(n_clusters=self.n_cluster, random_state=42)
        kmeans.fit(pixels)

        colors = kmeans.cluster_centers_.astype(int)
        
        return colors
    
    def sort_blue(self, colors):
        colors = colors.astype(np.uint8)
        colors_hsv = cv2.cvtColor(colors.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        
        H = colors_hsv[:, 0].astype(float)
        S = colors_hsv[:, 1].astype(float)
        V = colors_hsv[:, 2].astype(float)

        wH = 1.0     # Hue weight
        wS = 0.3     # Staturation weight
        wV = 0.1     # Value weight

        score = wH * H + wS * S + wV * V

        sorted_indices = np.argsort(-score)
        sorted_colors = colors[sorted_indices]
        
        # plt.figure(figsize=(10, 2))
        # for i, color in enumerate(sorted_colors):
        #     plt.subplot(1, len(sorted_colors), i + 1)
        #     plt.imshow([[color / 255]])
        #     plt.axis("off")
        # plt.show()
        
        return colors[sorted_indices]
    
    def apply_kmeans(self, image, init_centers):
        shape = image.shape
        rgb_image = image.reshape(-1, 3)
        kmeans_init_centers = init_centers

        kmeans = KMeans(n_clusters=self.n_cluster, init=kmeans_init_centers, random_state=42)
        cluster_labels = kmeans.fit_predict(rgb_image)
        
        init_centers = init_centers.astype(np.uint8)

        # Conversion en HSV
        hsv = cv2.cvtColor(init_centers.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)

        BLUE_H_MIN = 100
        BLUE_H_MAX = 130

        blue_indices = [
            i for i, (h, s, v) in enumerate(hsv)
            if BLUE_H_MIN <= h <= BLUE_H_MAX and s > 60 and v > 60
        ]
        
        print(blue_indices)
        
        mask = ~np.isin(cluster_labels, blue_indices)

        img_array = np.array(rgb_image)
        new_img = np.full_like(img_array, 255)
        new_img[mask] = img_array[mask]

        new_img = new_img.reshape(shape)

        return new_img
    
    def apply(self, image):
        colors = self.get_dominant_colors(image)
        centers = self.sort_blue(colors)
        no_background = self.apply_kmeans(image, centers)
        
        # plt.figure(figsize=(20, 10))

        # plt.subplot(1, 2, 1)
        # plt.imshow(image)

        # plt.subplot(1, 2, 2)
        # plt.imshow(no_background)
        
        return no_background
    
class BackgroundRemoveHSVOperator(BaseOperator):
    def apply(self, image):
        # rgb = image.copy()
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array([90,  40,  40])
        upper_blue = np.array([145, 255, 255])
        
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        kernel = np.ones((3,3), np.uint8)
        mask_clean = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=1)
        
        mask_clean = cv2.GaussianBlur(mask_clean, (5,5), 0)
        
        result = image.copy()
        result[mask_clean > 0] = [255, 255, 255]
        
        return result
        
class BinaryConvertOperator(BaseOperator):
    def __init__(self, invert=True):
        super().__init__()
        self.invert = invert
        
    def apply(self, image):
        mask = (image == [255, 255, 255]).all(axis=-1)
        binary = np.zeros_like(image)
        binary[mask] = [255, 255, 255]
        
        if self.invert:
            return cv2.bitwise_not(binary)

        return binary
        