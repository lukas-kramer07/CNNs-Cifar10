import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

def augment(image, label):

  if tf.random.uniform((), minval=0, maxval=1)<0.1:
    image = tf.image.rgb_to_grayscale(image)
    image = tf.tile(image, [1,1,3])
  image = tf.image.random_brightness(image, max_delta=0.1)
  image = tf.image.random_contrast(image, lower=0.1, upper=0.21)
  image = tf.image.random_flip_left_right(image)

  return image, label
def normalize_image(image, label):
  return tf.cast(image, tf.float32) / 255., label

(ds_train, ds_test), ds_info =tfds.load(
    "cifar10",
    split=["test","train"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
# Normalize pixel values to be between 0 and 1
ds_train = ds_train.map(normalize_image)
ds_train = ds_train.map(augment)
ds_test =  ds_test.map(normalize_image)
# Display the first image from the test set