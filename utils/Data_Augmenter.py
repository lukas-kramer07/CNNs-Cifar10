import tensorflow as tf

def augment(image):
    if tf.random.uniform((), minval=0, maxval=1)<0.1:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.tile(image, [1,1,3])
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.1, upper=0.21)
    image = tf.image.random_flip_left_right(image)

    return image