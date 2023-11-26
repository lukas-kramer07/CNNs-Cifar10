"""
HP tuning for new Res-Net architecture
"""

import os
import keras_tuner as kt
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ReduceLROnPlateau,
    TensorBoard,
)
from V11_B import preprocess_data
from V11_E import visualize_data
from keras.layers import (
    Conv2D,
    MaxPool2D,
    Dense,
    InputLayer,
    BatchNormalization,
    Layer,
    GlobalAveragePooling2D,
    Add,
)
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

IM_SIZE = 32
BATCH_SIZE = 32
MAX_TRIALS = 40


class ResBlock(Layer):
    def __init__(self, channels, stride=1, name="res_block"):
        super(ResBlock, self).__init__(name=name)

        self.res_conv = stride != 1
        self.conv1 = Conv2D(
            filters=channels, kernel_size=3, strides=stride, padding="same"
        )
        self.norm1 = BatchNormalization()
        self.conv2 = Conv2D(filters=channels, kernel_size=3, padding="same")
        self.norm2 = BatchNormalization()
        self.relu = ReLU()
        if self.res_conv:
            self.norm3 = BatchNormalization()
            self.conv3 = Conv2D(filters=channels, kernel_size=1, strides=stride)

    def call(self, input, training):
        x = self.conv1(input)
        x = self.norm1(x, training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x, training)

        if self.res_conv:
            input = self.conv3(input)
            input = self.norm3(input, training)
        result = Add()([x, input])
        return self.relu(result)


def main():
    """main function that uses preprocess_data and visualize_data from V11_E to prepare the dataset. It then starts a bayesianOptimizer search for the best hparams for the model."""
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
    model = tf.keras.Sequential()

    # Input block
    model.add(InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)))
    model.add(
        Conv2D(
            filters=64,
            kernel_size=7,
            strides=1,
            padding="same",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=2, strides=2))

    # Residual blocks
    for reps, groups in enumerate(config):
        for n in range(groups):
            channels = 64 * (2**reps)
            if n == 0 and reps == 0:
                model.add(ResBlock(channels, name=f"res_cell-{reps}-{n}-1"))
            elif n == 0:
                model.add(ResBlock(channels, stride=2, name=f"res_cell-{reps}-{n}-1"))
            else:
                model.add(ResBlock(channels, name=f"res_cell-{reps}-{n}-2"))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(10, activation="softmax"))
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def build_tuner(hp):
    # HPARAMS
    HP_NUM_FILTERS_1 = hp.Int("num_filters_1", 4, 64)
    HP_NUM_RESBLOCKS = hp.Int("num_resblocks_1", 1, 4)
    HP_NUM_UNITS1 = hp.Int("num_units_1", 64, 128)
    HP_NUM_UNITS2 = hp.Int("num_units_2", 32, 128)
    HP_NUM_UNITS3 = hp.Int("num_units_3", 16, 64)
    HP_REGULARIZATION_RATE = hp.Float("regularization_rate", 0.001, 0.3)
    HP_DROPOUT = hp.Float("dropout", 0.01, 0.3)
    HP_LEARNING_RATE = hp.Float("learning_rate", 0.0001, 0.01)
    model = build_model_base(
        HP_NUM_FILTERS_1=HP_NUM_FILTERS_1,
        HP_NUM_RESBLOCKS=HP_NUM_RESBLOCKS,
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
        executions_per_trial=1,
    )
    # callbacks
    stop_early = EarlyStopping(monitor="val_loss", patience=4, verbose=1)

    def scheduler(epoch, lr):
        if epoch <= 3:
            lr = lr
        elif epoch % 3 == 0:
            lr = (lr * tf.math.exp(-0.35)).numpy()
        return lr

    scheduler_callback = LearningRateScheduler(scheduler, verbose=1)
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
    logdir = os.path.join(base_dir, log_dir)
    t_callback = TensorBoard(log_dir=logdir, update_freq=500, profile_batch=0)
    tuner.search(
        train_ds,
        epochs=20,
        validation_data=(val_ds),
        callbacks=[stop_early, t_callback, scheduler_callback, plateau_callback],
    )
    tuner.results_summary()


if __name__ == "__main__":
    main()
