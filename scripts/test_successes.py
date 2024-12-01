import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
import cv2 
import matplotlib.patches as patches
import matplotlib.path as mpath

class PlotSuccess:
    def __init__(self, directory):
        self.directory = directory

    def check_red_in_region(self, image, xy=(130.136, 55.3258), width=36, height=4, angle=38, red_threshold=150, green_blue_max=100):
        image = np.transpose(image, (1,2,0))
        y = int(xy[1])
        x = int(xy[0])
        M = cv2.getRotationMatrix2D((x, y), angle, 1)
        rot_im = cv2.warpAffine(image, M, (image.shape[1], image.shape[0])) # shape of image?
        y_start = y - height // 2
        y_end = y_start + height
        x_start = x - width // 2
        x_end = x_start + width

        y_end = y_start + height
        x_end = x_start + width

        roi = rot_im[y_start:y_end, x_start:x_end, :]


        red_pixels = (roi[0, :, :] > red_threshold) & (roi[1, :, :] < green_blue_max) & (roi[2, :, :] < green_blue_max)
        return np.any(red_pixels)

    def find_object_centroid(self, og_image, red_threshold, green_blue_max):
        image = np.transpose(og_image, (1, 2, 0))
        red_mask = (image[:,:,0] > red_threshold) & (image[:,:,1] < green_blue_max) & (image[:,:,2] < green_blue_max)
        object_coords = np.argwhere(red_mask)
        if len(object_coords) == 0:
            return None  
        centroid = np.mean(object_coords, axis=0)
        return (centroid[1], centroid[0])
    
    def check_centroid_in_region(self, image, xy=(157.565, 47.158), width=45, height=65, angle=50):
        centroid = self.find_object_centroid(image, red_threshold=150, green_blue_max=100)
        rect = patches.Rectangle(xy, width, height, angle=angle)
        rect_coords = rect.get_patch_transform().transform(rect.get_path().vertices[:-1])
        path = mpath.Path(rect_coords)
        return path.contains_point(centroid), centroid
    
    def check_success(self, image):
        red_present, centroid = self.check_centroid_in_region(image)
        return red_present, centroid

    def process_images(self):
        for filename in os.listdir(self.directory):
            if filename.endswith('.npz'):
                with np.load(os.path.join(self.directory, filename)) as data:
                    image = data['obs.agent_image'][-25]
                    centroid_correct, centroid = self.check_success(image)
                    red_present = self.check_red_in_region(image)
                    red_present = red_present and centroid_correct
                    fig, ax = plt.subplots()
                    plt.scatter([centroid[0]], [centroid[1]], c='blue', marker='o')
                    ax.imshow(np.transpose(image, (1, 2, 0))) 
                    rect = Rectangle((157.565, 47.158), width=45, height=65, linewidth=1, edgecolor='r', facecolor='none', angle=50)
                    rect_peg = Rectangle((130.136, 55.3258), width=36, height=4, angle=38, linewidth=1, edgecolor='b', facecolor='none')
                    ax.add_patch(rect)
                    ax.add_patch(rect_peg)
                    plt.title(f"Successful?: {red_present}")
                    plt.show()

visualizer = PlotSuccess('directory')
visualizer.process_images()
