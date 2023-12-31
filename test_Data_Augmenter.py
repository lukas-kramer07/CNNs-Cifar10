import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import utils

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
# Create a data generator with data augmentation

train_datagen = utils.create_data_augmenter(train_images)

# Display some of the original images
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(train_images[i], cmap='binary')
    ax.set_title(f'Label: {train_labels[i]}')
plt.suptitle('Original Images')
plt.show()

# Display some of the augmented images
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    augmented_image, _ = train_datagen.flow(
        tf.expand_dims(train_images[i],0),
        tf.expand_dims(train_labels[i],0)
    ).next()
    ax.imshow(augmented_image.squeeze(), cmap='binary')
    ax.set_title(f'Label: {train_labels[i]}')
plt.suptitle('Augmented Images')
plt.show()