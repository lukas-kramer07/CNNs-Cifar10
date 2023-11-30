import tensorflow as tf
import albumentations as A
from functools import partial


def resize_rescale(Image, Label, IM_SIZE, class_names):
    Image = tf.image.resize(Image, (IM_SIZE, IM_SIZE))
    Label = tf.one_hot(Label, len(class_names))  # one_hot_encode
    return Image / 255.0, Label


@tf.function
def augment(image, label, IM_SIZE):
    aug_img = tf.numpy_function(
        func=aug_albument, inp=[image, IM_SIZE], Tout=tf.float32
    )
    return aug_img, label


def aug_albument(image, IM_SIZE):
    transform = create_transform(IM_SIZE)
    new_image = transform(image=image)["image"]
    return new_image


def create_transform(IM_SIZE):
    transforms = A.Compose(
        [
            # 'mechanical' transformations
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(height=IM_SIZE,width=IM_SIZE,scale=(0.78,1), p=0.75),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0, rotate_limit=12, p=0.65
            ),
        ]
    )
    return transforms


def preprocess_data(train_ds, test_ds, batch_size, IM_SIZE, class_names):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    augment_func = partial(augment, IM_SIZE=IM_SIZE)
    resize_rescale_func = partial(
        resize_rescale, IM_SIZE=IM_SIZE, class_names=class_names
    )
    train_ds = (
        train_ds.map(resize_rescale_func, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(8, reshuffle_each_iteration=True)
        .map(augment_func, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    test_ds = (
        test_ds.map(resize_rescale_func, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return train_ds, test_ds
