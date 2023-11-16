"""
This script is a comparison of the hp models developed in V12 A to C
"""
"""
  uses V2 model and includes the tensorboard_callback
"""
import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import utils
from V11_E import preprocess_data
from V11_E import visualize_data
from keras import backend as K
from keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from keras.layers import (
    Conv2D,
    MaxPool2D,
    Dense,
    InputLayer,
    Flatten,
    BatchNormalization,
    Dropout,
)
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

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
    """main function that uses preprocess_data and visualize_data from V11_E to prepare the dataset. It then tests all V12 models."""
    # load dataset
    (train_ds, test_ds), ds_info = tfds.load(
        "cifar10", split=["train", "test"], as_supervised=True, with_info=True
    )
    # preprocess
    train_ds, test_ds = preprocess_data(train_ds, test_ds)
    # visualize new data
    visualize_data(train_ds=train_ds, test_ds=test_ds, ds_info=ds_info)

    # Test model A
    model_name = "V12_A"
    model_A = build_model_A()
    test_model(model=model_A, model_name=model_name, train_ds=train_ds, test_ds=test_ds)

    # Test model B
    model_name = "V12_B"
    model_B = build_model_B()
    test_model(model=model_B, model_name=model_name, train_ds=train_ds, test_ds=test_ds)
    # Test model C
    model_name = "V12_C"


def test_model(model, model_name, train_ds, test_ds):
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

    # Train model

    # train for 20 epochs
    history = model.fit(
        train_ds,
        epochs=30,
        validation_data=test_ds,
        callbacks=[
            es_callback,
            scheduler_callback,
            checkpoint_callback,
            plateau_callback,
            LRTensorBoard(log_dir=LOG_DIR),
        ],
    )

    # model evaluation
    utils.model_eval(
        history=history,
        model=model,
        model_name=model_name,
        test_ds=test_ds,
        class_names=class_names,
    )


def build_model_A():
    model = tf.keras.Sequential(
        [
            # Input
            InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),
            #
            # First Convolutional block
            Conv2D(
                filters=16,
                kernel_size=3,
                strides=1,
                padding="valid",
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.001),
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=2, strides=2),
            Dropout(rate=0.2),
            #
            # Second Convolutional block
            Conv2D(
                filters=32,
                kernel_size=3,
                strides=1,
                padding="valid",
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.001),
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=2, strides=2),
            Dropout(rate=0.2),
            #
            # Third Convolutional block
            Conv2D(
                filters=64,
                kernel_size=3,
                strides=1,
                padding="valid",
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.001),
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=2, strides=2),
            Dropout(rate=0.2),
            # Dense block
            Flatten(),
            Dense(
                128,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.001),
            ),
            BatchNormalization(),
            Dropout(rate=0.2),
            Dense(
                64,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.001),
            ),
            BatchNormalization(),
            Dropout(rate=0.2),
            Dense(
                16,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.001),
            ),
            BatchNormalization(),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def build_model_B():
    model = tf.keras.Sequential(
        [
            # Input
            InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),
            #
            # First Convolutional block
            Conv2D(
                filters=22,
                kernel_size=3,
                strides=1,
                padding="valid",
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.063145),
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=2, strides=2),
            Dropout(rate=0.18027),
            #
            # Second Convolutional block
            Conv2D(
                filters=48,
                kernel_size=3,
                strides=1,
                padding="valid",
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.063145),
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=2, strides=2),
            Dropout(rate=0.18027),
            #
            # Third Convolutional block
            Conv2D(
                filters=71,
                kernel_size=3,
                strides=1,
                padding="valid",
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.063145),
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=2, strides=2),
            Dropout(rate=0.18027),
            # Dense block
            Flatten(),
            Dense(
                159,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.063145),
            ),
            BatchNormalization(),
            Dropout(rate=0.18027),
            Dense(
                100,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.063145),
            ),
            BatchNormalization(),
            Dropout(rate=0.18027),
            Dense(
                27,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.063145),
            ),
            BatchNormalization(),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    main()
    plt.show()
