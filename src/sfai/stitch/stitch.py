# import cv2
# import numpy as np
# from stitch2d import StructuredMosaic
# import glob
# import re
# import os
# import matplotlib.pyplot as plt

# pattern = re.compile(r"r(\d+)c(\d+)")

# def sort_key(filename):
#     r, c = pattern.search(filename).groups()
#     return int(r), int(c)

# def chunk_and_pad(lst, size=8, fill=None):
#     chunks = [lst[i:i+size] for i in range(0, len(lst), size)]

#     if len(chunks[-1]) < size:
#         chunks[-1] += [fill] * (size - len(chunks[-1]))

#     return chunks

# def prepare_file_list(root_dir):
#     images = glob.glob('*.jpg', root_dir=root_dir)
#     images_sorted = sorted(images, key=sort_key)
    
#     padded = chunk_and_pad(images_sorted)
    
#     for i in range(len(padded)):
#         if not i % 2:
#             padded[i] = padded[i][::-1]
    
#     return padded
    
# def overlap_phasecorr(imgA, imgB, band_size=200):
#     A = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
#     B = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

#     bandA = A[:, -band_size:]
#     bandB = B[:, :band_size]

#     bandA = cv2.normalize(bandA, None, 0, 255, cv2.NORM_MINMAX)
#     bandB = cv2.normalize(bandB, None, 0, 255, cv2.NORM_MINMAX)

#     shift, response = cv2.phaseCorrelate(
#         np.float32(bandA),
#         np.float32(bandB)
#     )
#     return shift, response

# def gen_mosaic(root_dir):
#     image_names = prepare_file_list(root_dir=root_dir)
#     band_size = 406
    
#     best_score = 0
#     best_shift = (0,0)
    
#     for row in image_names:
#         for a, b in zip(row, row[1:]):
#             if a and b:
#                 im_left = cv2.imread(os.path.join(root_dir, a))
#                 im_right = cv2.imread(os.path.join(root_dir, b))
                
#                 shift, response = overlap_phasecorr(im_left, im_right, band_size)
#                 if response > best_score:
#                     best_score = response
#                     best_shift = shift
#                     print(best_score)
#                     print(best_shift)
                    
#     for row in image_names:
#         row = stitch_row(row, root_dir, best_shift[0], best_shift[1], band_size)
    
# def stitch_row(row, root_dir, dx, dy, band_size=200):
#     images = [cv2.imread(os.path.join(root_dir, im)) for im in row]
    
#     dx = int(round(dx))

#     left = images[0]
#     right = images[1]

#     H = left.shape[0]
#     W = left.shape[1] + right.shape[1]

#     # IMPORTANT : correction du signe !
#     x_B = left.shape[1] - band_size - dx

#     canvas = np.zeros((H, W, 3), dtype=np.uint8)
#     canvas[:, :left.shape[1]] = left
#     canvas[:, x_B:x_B+right.shape[1]] = right

#     plt.figure(figsize=(20, 10)) 
#     plt.imsave('stich.jpg', canvas) 
#     plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)) 
#     plt.axis("off")
#     plt.show()
    
# def draw_guides(row, root_dir, dx, band_size=200):
#     images = [cv2.imread(os.path.join(root_dir, im)) for im in row]
    
#     dx = int(round(dx))
    
#     nb_images = len(images)
#     im_width = images[0].shape[1]
    
#     x_B = band_size + dx

#     left = images[0]
#     right = images[1]
    
#     cv2.line(right, (band_size, 0), (band_size, right.shape[0] ), (0, 255, 0), 3)
#     cv2.line(right, (x_B, 0), (x_B, right.shape[0] ), (0, 0, 255), 3)
#     plt.figure(figsize=(20, 10))
#     plt.imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
#     plt.axis("off")
#     plt.show()
    

            
    
# def stitch_horizontal(imgA, imgB, band_size=200):
#     # Récupérer shift horizontal via phase correlation
#     (dx, dy), response = overlap_phasecorr(imgA, imgB, band_size)
#     dx = int(round(dx))   # arrondi du shift (flottant → entier)

#     # Calcul de la position horizontale où placer imgB
#     x_B = imgA.shape[1] - band_size + dx

#     # Création du canevas final
#     H = max(imgA.shape[0], imgB.shape[0])
#     W = x_B + imgB.shape[1]
#     canvas = np.zeros((H, W, 3), dtype=np.uint8)

#     # Placement de A et B non mélangés
#     canvas[:, :imgA.shape[1]] = imgA
#     canvas[:, x_B:x_B+imgB.shape[1]] = imgB

#     # ----------------------------------------------------------
#     # Blending dans la zone d'overlap (optionnel mais conseillé)
#     # ----------------------------------------------------------
#     overlap_start = imgA.shape[1] - band_size + dx
#     overlap_width = band_size - dx

#     if overlap_width > 0:
#         bandA_full = imgA[:, -overlap_width:]
#         bandB_full = imgB[:, :overlap_width]

#         alpha = np.linspace(1, 0, overlap_width).reshape(1, -1, 1)

#         blended = (bandA_full * alpha + bandB_full * (1 - alpha)).astype(np.uint8)

#         canvas[:, overlap_start:overlap_start+overlap_width] = blended

#     return canvas

            
# def test():
#     root_dir = '/Users/robin/Pictures/soil-fauna-ai/full_images/A02-E/'

#     mosaic = StructuredMosaic(
#         "/Users/robin/Pictures/soil-fauna-ai/full_images/A02-E/",
#         dim=8,                  # number of tiles in primary axis
#         origin="upper right",     # position of first tile
#         direction="horizontal",  # primary axis (i.e., the direction to traverse first)
#         pattern="snake"          # snake or raster
#     )
    
#     mosaic.load_params("params.json")
    
    
#     mosaic.align()
#     mosaic.build_out(from_placed=True)
#     mosaic.reset_tiles()
#     mosaic.save_params()

#     mosaic.smooth_seams()
#     mosaic.save("mosaic2.jpg")
    
    
# ##########
# # V2
# ##########

# import cv2
# import numpy as np
# import os
# import glob

# def read_images(image_folder):
#     """Read all images in the folder, assuming they are named sequentially."""
#     file_paths = sorted(glob.glob(os.path.join(image_folder, '*.png')), sort_key)  # adjust the pattern to your file format
#     images = [cv2.imread(file) for file in file_paths]
#     return images

# def stitch_images(images, rows, cols):
#     """Stitches images in a tile format."""
#     stitcher = cv2.Stitcher_create()
#     # Initiate empty list to hold results of each row stitching
#     row_images = []
    
#     # Stitch images row by row
#     for r in range(rows):
#         row_start = r * cols
#         row_end = row_start + cols
#         status, stitched_row = stitcher.stitch(images[row_start:row_end])
#         if status != cv2.Stitcher_OK:
#             print("Stitching row {} failed.".format(r))
#             return None
#         row_images.append(stitched_row)
    
#     # Stitch the rows vertically
#     status, stitched_image = stitcher.stitch(row_images)
#     if status != cv2.Stitcher_OK:
#         print("Vertical stitching failed.")
#         return None

#     return stitched_image

# def test2():
#     image_folder = 'path_to_your_folder_containing_images'
#     images = read_images(image_folder)
#     if images:
#         # Assumed parameters: adjust these based on your actual layout
#         num_rows = 5  # Number of rows in your grid
#         num_cols = 5  # Number of columns in your grid

#         final_image = stitch_images(images, num_rows, num_cols)
#         if final_image is not None:
#             cv2.imshow('Stitched Microscopy Image', final_image)
#             cv2.waitKey(0)
#             cv2.imwrite('final_stitched_output.tif', final_image)
#             cv2.destroyAllWindows()
#     else:
#         print("Error reading images.")