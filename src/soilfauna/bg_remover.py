from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2

def get_dominant_colors(image, filename, n_colors=3, show_palette=True):
    img = Image.open(image)
    resized = img.resize((300, 300))
    
    # Conversion en tableau numpy
    img_np = np.array(resized)
    pixels = img_np.reshape((-1, 3))

    # K-means
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)
    
    # if show_palette:
    #     plt.figure(figsize=(10, 2))
    #     for i, color in enumerate(colors):
    #         plt.subplot(1, n_colors, i + 1)
    #         plt.imshow([[color / 255]])
    #         plt.axis("off")
    #     plt.show()
    
        
    sorted_c = sort_blue(colors, n_colors, show_palette=show_palette)
    
    no_bg = apply_kmeans(img, sorted_c, n_colors)
    
    plt.imsave(f'./output/{filename}', no_bg)
    return colors

def sort_blue(colors, n_colors, show_palette):
    print(colors)
    colors_bgr = colors[:, ::-1].astype(np.uint8)
    print(colors_bgr)
    colors_hsv = cv2.cvtColor(colors_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    
    H = colors_hsv[:, 0].astype(float)
    S = colors_hsv[:, 1].astype(float)
    V = colors_hsv[:, 2].astype(float)

    # Poids ajustables selon ton besoin
    wH = 1.0     # poids pour la teinte
    wS = 0.3     # poids pour la saturation
    wV = 0.1     # poids pour la luminosité

    score = wH * H + wS * S + wV * V

    # Tri décroissant (plus bleu → score plus élevé)
    sorted_indices = np.argsort(-score)
    sorted_colors = colors[sorted_indices]
    
    if show_palette:
        plt.figure(figsize=(10, 2))
        for i, color in enumerate(sorted_colors):
            plt.subplot(1, n_colors, i + 1)
            plt.imshow([[color / 255]])
            plt.axis("off")
        plt.show()
    
    return sorted_colors
    
def apply_kmeans(image, init_centers, n_clusters):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    shape = image.shape
    rgb_image = image.reshape(-1, 3)
    kmeans_init_centers = init_centers

    kmeans = KMeans(n_clusters=n_clusters, init=kmeans_init_centers, random_state=42)
    cluster_labels = kmeans.fit_predict(rgb_image)
    
    # RGB -> BGR pour OpenCV
    bgr = init_centers[:, ::-1].astype(np.uint8)

    # Conversion en HSV
    hsv = cv2.cvtColor(bgr.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)

    BLUE_H_MIN = 100
    BLUE_H_MAX = 120

    blue_indices = [
        i for i, (h, s, v) in enumerate(hsv)
        if BLUE_H_MIN <= h <= BLUE_H_MAX and s > 50 and v > 50
    ]
    
    print(blue_indices)
    
    mask = ~np.isin(cluster_labels, blue_indices)

    img_array = np.array(rgb_image)
    new_img = np.full_like(img_array, 255)
    new_img[mask] = img_array[mask]

    new_img = new_img.reshape(shape)

    return new_img
    


