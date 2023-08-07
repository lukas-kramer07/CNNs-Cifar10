import tensorflow as tf
from tensorflow.keras import datasets
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import utils

def augment(image):

  if tf.random.uniform((), minval=0, maxval=1)<0.15:
    if tf.random.uniform((), minval=0,maxval=1)<0.3:
        image = tf.image.rgb_to_hsv
    else:
      image = tf.image.rgb_to_grayscale(image)
      image = tf.tile(image, [1,1,3])
  image = tf.image.random_brightness(image, max_delta=0.25)
  image = tf.image.random_contrast(image, lower=0.85, upper=1)
  image = tf.image.random_flip_left_right(image)


  return image
def normalize_image(image, label):
  return tf.cast(image, tf.float32) / 255., label

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

'''for i in range(len(train_images)):  
  train_images[i] = utils.augment(train_images[i])  '''

plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i])
  # The CIFAR labels happen to be arrays, 
  # which is why you need the extra index
  plt.xlabel(class_names[train_labels[i][0]])
plt.show()
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(augment(train_images[i]))
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
plt.show()