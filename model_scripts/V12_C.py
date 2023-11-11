"""
This is the third of three scripts to improve the model architecture using HP-search. This script will use the keras tuner
"""

import os
import random
import keras_tuner as kt
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import tensorflow_datasets as tfds
from V11_E import preprocess_data
from V11_E import visualize_data
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
MAX_EPOCHS = 10


def main():
    """main function that uses preprocess_data and visualize_data from V11_E to prepare the dataset. It then starts a grid search for the best hparams for the model."""
    # load dataset
    (train_ds, test_ds), ds_info = tfds.load(
        "cifar10", split=["train", "test"], as_supervised=True, with_info=True
    )
    # preprocess
    train_ds, test_ds = preprocess_data(train_ds, test_ds)
    # visualize new data
    visualize_data(train_ds=train_ds, test_ds=test_ds, ds_info=ds_info)
    runall()


def build_model_base(
    HP_NUM_FILTERS_1,
    HP_NUM_FILTERS_2,
    HP_NUM_FILTERS_3,
    HP_NUM_UNITS1,
    HP_NUM_UNITS2,
    HP_NUM_UNITS3,
    HP_REGULARIZATION_RATE,
    HP_DROPOUT,
    HP_LEARNING_RATE,
):
    model = tf.keras.Sequential(
        [
            # Input
            InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),
            #
            # First Convolutional block
            Conv2D(
                filters=HP_NUM_FILTERS_1,
                kernel_size=3,
                strides=1,
                padding="valid",
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(HP_REGULARIZATION_RATE),
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=2, strides=2),
            Dropout(rate=HP_DROPOUT),
            #
            # Second Convolutional block
            Conv2D(
                filters=HP_NUM_FILTERS_2,
                kernel_size=3,
                strides=1,
                padding="valid",
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(HP_REGULARIZATION_RATE),
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=2, strides=2),
            Dropout(rate=HP_DROPOUT),
            #
            # Third Convolutional block
            Conv2D(
                filters=HP_NUM_FILTERS_3,
                kernel_size=3,
                strides=1,
                padding="valid",
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(HP_REGULARIZATION_RATE),
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=2, strides=2),
            Dropout(rate=HP_DROPOUT),
            # Dense block
            Flatten(),
            Dense(
                HP_NUM_UNITS1,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(HP_REGULARIZATION_RATE),
            ),
            BatchNormalization(),
            Dropout(rate=HP_DROPOUT),
            Dense(
                HP_NUM_UNITS2,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(HP_REGULARIZATION_RATE),
            ),
            BatchNormalization(),
            Dropout(rate=HP_DROPOUT),
            Dense(
                HP_NUM_UNITS3,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(HP_REGULARIZATION_RATE),
            ),
            BatchNormalization(),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=HP_LEARNING_RATE),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def build_tuner(hp):
    # HPARAMS
    HP_NUM_FILTERS_1 = hp.HParam("num_filters_1", hp.IntInterval(4, 32))
    HP_NUM_FILTERS_2 = hp.HParam("num_filters_2", hp.IntInterval(8, 64))
    HP_NUM_FILTERS_3 = hp.HParam("num_filters_3", hp.IntInterval(16, 128))
    HP_NUM_UNITS1 = hp.HParam("num_units_1", hp.IntInterval(64, 256))
    HP_NUM_UNITS2 = hp.HParam("num_units_2", hp.IntInterval(32, 128))
    HP_NUM_UNITS3 = hp.HParam("num_units_3", hp.IntInterval(16, 64))
    HP_REGULARIZATION_RATE = hp.HParam(
        "regularization_rate", hp.RealInterval(0.001, 0.3)
    )
    HP_DROPOUT = hp.HParam("dropout", hp.RealInterval(0.01, 0.3))
    HP_LEARNING_RATE = hp.HParam("learning_rate", hp.RealInterval(0.0001, 0.01))
    model = build_model_base(HP_NUM_FILTERS_1=HP_NUM_FILTERS_1, 
                             HP_NUM_FILTERS_2=HP_NUM_FILTERS_2,
                             HP_NUM_FILTERS_3=HP_NUM_FILTERS_3,
                             HP_NUM_UNITS1=HP_NUM_UNITS1,
                             HP_NUM_UNITS2=HP_NUM_UNITS2,
                             HP_NUM_UNITS3=HP_NUM_UNITS3,
                             HP_LEARNING_RATE=HP_LEARNING_RATE,
                             HP_DROPOUT=HP_DROPOUT,
                             HP_REGULARIZATION_RATE=HP_REGULARIZATION_RATE)
    return model

def runall(num_sessions, base_logdir, train_ds, val_ds):
    tuner = kt.Hyperband(
        build_model_base,
        objective="val_accuracy",
        max_epochs=MAX_EPOCHS,
        factor=3,
        directory="logs",
        project_name="hp",
    )


if __name__ == "__main__":
    main()
