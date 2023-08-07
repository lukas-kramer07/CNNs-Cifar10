import tensorflow as tf

def augment(image):
    # Random rotation (up to 15 degrees)
    image = tf.keras.preprocessing.image.random_rotation(image, 15, row_axis=0, col_axis=1, channel_axis=2)

    # Random zoom (up to 20%)
    zoom_range = (0.8, 1.2)
    image = tf.keras.preprocessing.image.random_zoom(image, zoom_range, row_axis=0, col_axis=1, channel_axis=2)

    # Random shear (up to 20 degrees)
    shear_range = 0.2
    image = tf.keras.preprocessing.image.random_shear(image, shear_range, row_axis=0, col_axis=1, channel_axis=2)

    # Random shift (up to 10%)
    width_shift_range = 0.1
    height_shift_range = 0.1
    image = tf.keras.preprocessing.image.random_shift(image, width_shift_range, height_shift_range,
                                                      row_axis=0, col_axis=1, channel_axis=2)

    # Random flip (horizontal and vertical)
    image = tf.image.random_flip_left_right(image)
    # Random brightness adjustment
    image = tf.image.random_brightness(image, max_delta=0.2)

    # Random contrast adjustment
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # Random saturation adjustment
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

    # Clip pixel values to [0, 1] range
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image
