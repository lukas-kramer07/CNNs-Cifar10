import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def add_noise(image):
    noise = np.random.normal(loc=0, scale=0.02, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)


def create_data_augmenter(train_images):
    # Create an instance of the ImageDataGenerator class for data augmentation
    data_augmenter = ImageDataGenerator(
        rotation_range=10,  # rotate the image up to 10 degrees
        width_shift_range=0.05,  
        height_shift_range=0.05,  
        horizontal_flip=True,  # flip the image horizontally
        vertical_flip=False,  # do not flip the image vertically
        zoom_range=0.05,  # zoom in/out up to 5%
        fill_mode='nearest',  # fill gaps in the image with the nearest pixel
        preprocessing_function=add_noise  # Add the add_noise function as the preprocessing function
)
    return data_augmenter