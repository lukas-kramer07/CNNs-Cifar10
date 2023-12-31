"""
  uses V2 model and includes more callbacks than V7
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import utils
from keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from V2 import create_model

model_name = "V9"
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


def main():
    # load dataset
    (train_ds, test_ds), ds_info = tfds.load("cifar10", split=["train", "test"], as_supervised=True, with_info=True)
    # preprocess
    train_ds, test_ds = preprocess_data(train_ds, test_ds)

    # define callbacks
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
        if epoch < 3:
            lr = lr
        else:
            lr = (lr * tf.math.exp(-0.1)).numpy()
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
    train_ds = (
        train_ds.map(resize_rescale, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(8, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    test_ds = test_ds.map(resize_rescale, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return train_ds, test_ds


def resize_rescale(Image, Label):
    Image = tf.image.resize(Image, (IM_SIZE, IM_SIZE))
    Label = tf.one_hot(Label, len(class_names))  # one_hot_encode
    return Image / 255.0, Label


if __name__ == "__main__":
    main()
    plt.show()
