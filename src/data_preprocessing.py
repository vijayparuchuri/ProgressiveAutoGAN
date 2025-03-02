import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from skimage.transform import resize
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

from PIL import Image
import shutil


PATH = "/data/"
# Read the image files and return normalized images of size (128, 128, 3)
def read_images(filename):
    image_list = []
    for file in os.listdir(filename):
        image = Image.open(os.path.join(filename, file))
        image = image.convert('RGB')
        pixels = image.resize((128, 128))
        pixels = np.asarray(pixels)
        pixels = pixels/255
        image_list.append(pixels)
    return np.asarray(image_list)

# Display 100 images from the dataset
img_gen = ImageDataGenerator(zoom_range = 0.2, rescale=1/255, dtype='float32')
images = img_gen.flow_from_directory(PATH, target_size = (128, 128), classes=None, class_mode=None, batch_size=32)

fig, axes = plt.subplots(10, 10, figsize=(20, 20))
for i in range(len(axes)):
    for j in range(len(axes)):
        axes[i][j].imshow(images[j][i])
        axes[i][j].get_xaxis().set_visible(False)
        axes[i][j].get_yaxis().set_visible(False)
