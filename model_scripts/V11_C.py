"""
  uses V2 model and V10 callbacks as well as augmentation using and augment 
  function and tf.image
"""
import datetime
from V11_A import augment
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import utils
from keras import backend as K
from keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from V2 import create_model

model_name = "V11_C"
IM_SIZE = 32
BATCH_SIZE = 32
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def mixup(train_ds1, train_ds2):
    """mixup function to mix two train_datasets

    Args:
        train_ds1 (tf dataset): first dataset
        train_ds2 (tf dataset): second dataset

    Returns:
        image: an image mixed up from the first and second dataset
        label: a mixup label equivalent to the ratio of both images
    """
    (image1, label1), (image2, label2) = train_ds1, train_ds2

    lamda = tfp.distributions.Beta(0.35, 0.35)
    lamda = lamda.sample(1)[0]

    image = lamda * tf.cast(image1, dtype=tf.float32) + (1 - lamda) * tf.cast(image2, dtype=tf.float32)
    label = lamda * float(label1) + (1 - lamda) * float(label2)
    return image, label


def main():
    # load dataset
    (train_ds, test_ds), ds_info = tfds.load("cifar10", split=["train", "test"], as_supervised=True, with_info=True)
    # preprocess
    train_ds, test_ds = preprocess_data(train_ds, test_ds)
    # visualize new data
    visualize_data(train_ds=train_ds, test_ds=test_ds, ds_info=ds_info)
    # define callbacks
    # custom TensorBoard callback
    Current_Time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = "./logs/fit/" + Current_Time

    class LRTensorBoard(TensorBoard):
        # add other arguments to __init__ if you need
        def __init__(self, log_dir, **kwargs):
            super().__init__(log_dir=log_dir, histogram_freq=3, **kwargs)

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            logs.update({"lr": K.eval(self.model.optimizer.lr)})
            super().on_epoch_end(epoch, logs)

    es_callback = EarlyStopping(
        monitor="val_accuracy",
        min_delta=0,
        patience=5,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    def scheduler(epoch, lr):
        if epoch <= 3:
            lr = lr
        elif epoch % 2 == 0:
            lr = (lr * tf.math.exp(-0.3)).numpy()
        return lr

    scheduler_callback = LearningRateScheduler(scheduler, verbose=1)
    checkpoint_callback = ModelCheckpoint(
        f"model_checkpoints/training_checkpoints/{model_name}",
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        mode="auto",
    )
    plateau_callback = ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.1,
        patience=3,
        verbose=1,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0,
    )

    # Create and compile model
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    # train for 20 epochs
    history = model.fit(
        train_ds,
        epochs=20,
        validation_data=test_ds,
        callbacks=[
            es_callback,
            scheduler_callback,
            checkpoint_callback,
            plateau_callback,
            LRTensorBoard(log_dir=LOG_DIR),
        ],
    )

    # model_evaluation
    utils.model_eval(
        history=history,
        model=model,
        model_name=model_name,
        test_ds=test_ds,
        class_names=class_names,
    )


## Preprocessing the dataset
def preprocess_data(train_ds, test_ds):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds_mixed = train_ds.map(resize_rescale)
    train_ds1 = train_ds_mixed.shuffle(buffer_size=32)
    train_ds2 = train_ds_mixed.shuffle(buffer_size=32)
    mixed_ds = tf.data.Dataset.zip((train_ds1, train_ds2))
    train_ds = mixed_ds.map(mixup, num_parallel_calls=AUTOTUNE)
    train_ds = (
        train_ds.shuffle(buffer_size=10000)
        .map(augment, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    test_ds = test_ds.map(resize_rescale, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return train_ds, test_ds


def resize_rescale(Image, Label):
    Image = tf.image.resize(Image, (IM_SIZE, IM_SIZE))
    Label = tf.one_hot(Label, len(class_names))  # one_hot_encode
    return Image / 255.0, Label


def visualize_data(train_ds, test_ds, ds_info):
    num_images_to_display = 15
    plt.figure(figsize=(num_images_to_display, num_images_to_display * 2))
    count = 0
    # Plot test samples
    for i in range(int(np.ceil(num_images_to_display / BATCH_SIZE))):
        image, label = next(iter(test_ds))
        for n in range(min(BATCH_SIZE, num_images_to_display - i * BATCH_SIZE)):
            plt.subplot(
                2 * int(tf.sqrt(float(num_images_to_display))) + 1,
                2 * int(tf.sqrt(float(num_images_to_display))) + 1,
                n + i + 1,
            )
            plt.imshow(image[n])
            plt.title(
                f"Test - {ds_info.features['label'].int2str(int(tf.argmax(label[n])))}",
                fontsize=8,
            )
            plt.axis("off")
            count += 1

    # Plot train samples
    for k in range(int(np.ceil(num_images_to_display / BATCH_SIZE))):
        image, label = next(iter(train_ds))
        for n in range(min(BATCH_SIZE, num_images_to_display - k * BATCH_SIZE)):
            plt.subplot(
                2 * int(tf.sqrt(float(num_images_to_display))) + 1,
                2 * int(tf.sqrt(float(num_images_to_display))) + 1,
                n + i + count + 1,
            )
            plt.imshow(image[n])
            # Compute the indices of the top two values in the label tensor
            top_k_values, top_k_indices = tf.math.top_k(label[n], k=2)

            # Extract the second largest label's index
            second_largest_index = top_k_indices[1]

            # Convert the index to the corresponding label
            second_largest_label = ds_info.features["label"].int2str(int(second_largest_index))
            plt.title(
                f"Train - {ds_info.features['label'].int2str(int(tf.argmax(label[n])))}: {label[n][top_k_indices[0]]:.3f}  \n and  {second_largest_label}: {label[n][second_largest_index]:.3f}",
                fontsize=8,
            )
            plt.axis("off")
    plt.tight_layout(w_pad=6, h_pad=4)
    plt.suptitle(
        'Train and Test Samples - class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]',
        fontsize=14,
    )
    plt.subplots_adjust(top=0.9)
    plt.show()


if __name__ == "__main__":
    main()
    plt.show()
