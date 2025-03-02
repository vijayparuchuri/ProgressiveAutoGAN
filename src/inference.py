import tensorflow as tf
from math import sqrt
import matplotlib.pyplot as plt

from train import generate_latent_points
from model import PixelNormalization, MinibatchStdev, WeightedSum

cust = {'PixelNormalization': PixelNormalization, 'MinibatchStdev': MinibatchStdev, 'WeightedSum': WeightedSum}
model = tf.keras.models.load_model('checkpoints/model_008x008-tuned.h5', cust)

# size of the latent space
latent_dim = 100
# number of images to generate
n_images = 25
# generate images
latent_points = generate_latent_points(latent_dim, n_images)
# generate images
X = model.predict(latent_points)

# create a plot of generated images
def plot_generated(images, n_images):
    # plot images
    square = int(sqrt(n_images))
    # normalize pixel values to the range [0,1]
    images = (images - images.min()) / (images.max() - images.min())
    for i in range(n_images):
        # define subplot
        plt.subplot(square, square, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(images[i])
    plt.show()
