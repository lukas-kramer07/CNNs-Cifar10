"""
HP tuning for new Res-Net architecture
"""

from ast import Add
import os
import keras_tuner as kt
from pkg_resources import resource_listdir
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
    Layer,
)
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

IM_SIZE = 32
BATCH_SIZE = 32
MAX_TRIALS = 50


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
    runall(base_dir="logs", log_dir="hp", train_ds=train_ds, val_ds=test_ds)


def build_model_base(
    HP_NUM_FILTERS_1,
    HP_NUM_RESBLOCKS,
    HP_NUM_UNITS1,
    HP_NUM_UNITS2,
    HP_NUM_UNITS3,
    HP_REGULARIZATION_RATE,
    HP_DROPOUT,
    HP_LEARNING_RATE,
):
    class ResCell(Layer):
        def __init__(self, channels, strides=1, name="res_cell"):
            super(ResCell, self).__init__(name=name)

            self.res_conv = strides != 1
            self.conv1 = Conv2D(
                filters=channels, kernel_size=3, strides=strides, padding="same"
            )
            self.conv2 = Conv2D(filters=channels, kernel_size=3, padding="same")
            self.norm = BatchNormalization()
            self.activation = tf.keras.activations.relu
            if self.res_conv:
                self.conv3 = Conv2D(filters=channels, kernel_size=1, strides=strides)

        def call(self, input, training):
            x = self.conv1(input)
            x = self.norm(x, training)
            x = self.conv2(x)
            x = self.norm(x, training)

            if self.res_conv:
                residue = self.conv3(input)
                residue = self.norm(residue, training)
                result = Add()([x, residue])
            else:
                result = Add()([x, input])
            return self.activation(result)

    model = tf.keras.Sequential(
        [
            # Input
            InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),
            #
            # First Convolutional block
            Conv2D(
                filters=HP_NUM_FILTERS_1,
                kernel_size=7,
                strides=1,
                padding="same",
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(HP_REGULARIZATION_RATE),
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=2, strides=2),
            Dropout(rate=HP_DROPOUT),
            # Residual Blocks
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
    HP_NUM_FILTERS_1 = hp.Int("num_filters_1", 4, 32)
    HP_NUM_FILTERS_2 = hp.Int("num_filters_2", 8, 64)
    HP_NUM_FILTERS_3 = hp.Int("num_filters_3", 16, 128)
    HP_NUM_UNITS1 = hp.Int("num_units_1", 64, 256)
    HP_NUM_UNITS2 = hp.Int("num_units_2", 32, 128)
    HP_NUM_UNITS3 = hp.Int("num_units_3", 16, 64)
    HP_REGULARIZATION_RATE = hp.Float("regularization_rate", 0.001, 0.3)
    HP_DROPOUT = hp.Float("dropout", 0.01, 0.3)
    HP_LEARNING_RATE = hp.Float("learning_rate", 0.0001, 0.01)
    model = build_model_base(
        HP_NUM_FILTERS_1=HP_NUM_FILTERS_1,
        HP_NUM_FILTERS_2=HP_NUM_FILTERS_2,
        HP_NUM_FILTERS_3=HP_NUM_FILTERS_3,
        HP_NUM_UNITS1=HP_NUM_UNITS1,
        HP_NUM_UNITS2=HP_NUM_UNITS2,
        HP_NUM_UNITS3=HP_NUM_UNITS3,
        HP_LEARNING_RATE=HP_LEARNING_RATE,
        HP_DROPOUT=HP_DROPOUT,
        HP_REGULARIZATION_RATE=HP_REGULARIZATION_RATE,
    )
    return model


def runall(base_dir, log_dir, train_ds, val_ds):
    tuner = kt.BayesianOptimization(
        build_tuner,
        objective="val_accuracy",
        max_trials=MAX_TRIALS,
        directory=base_dir,
        executions_per_trial=2,
    )
    # callbacks
    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    logdir = os.path.join(base_dir, log_dir)
    t_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir, update_freq=500, profile_batch=0
    )
    tuner.search(
        train_ds,
        epochs=4,
        validation_data=(val_ds),
        callbacks=[stop_early, t_callback],
    )
    tuner.results_summary()


if __name__ == "__main__":
    main()
