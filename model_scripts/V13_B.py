"""
This script is a model version of the ResNet developed in V13_A
"""
import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import utils
from V11_D import preprocess_data
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
    Activation,
    BatchNormalization,
    Dropout,
    Layer,
    Add,
    GlobalMaxPooling2D,
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


class ResCell(Layer):
    def __init__(self, channels, strides=1, name="res_cell"):
        super(ResCell, self).__init__(name=name)

        self.res_conv = True if strides != 1 else False
        self.conv1 = Conv2D(
            filters=channels,
            kernel_size=3,
            strides=strides,
            padding="same",
        )
        self.conv2 = Conv2D(
            filters=channels,
            kernel_size=3,
            padding="same",
        )
        self.norm = BatchNormalization(axis=3)
        self.activation = Activation('relu')

        if self.res_conv:
            self.conv3 = Conv2D(
                filters=channels, kernel_size=1, strides=strides,
            )

    def call(self, input, training):
        x = self.conv1(input)
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm(x)

        if self.res_conv:
            residue = self.conv3(input)
            residue = self.norm(residue, training)
            result = Add()([x, residue])
        else:
            result = Add()([x, input])
        return self.activation(result)


def main():
    """main function that uses preprocess_data and visualize_data from V11_E to prepare the dataset. It then tests all V12 models."""
    # load dataset
    (train_ds, test_ds), ds_info = tfds.load(
        "cifar10", split=["train", "test"], as_supervised=True, with_info=True
    )
    # preprocess
    train_ds, test_ds = preprocess_data(train_ds, test_ds, batch_size=64)
    # visualize new data
    visualize_data(train_ds=train_ds, test_ds=test_ds, ds_info=ds_info)

    # Test model A
    model_name = "V13_A"
    config = [2,2,2,2]
    model_A = build_model_A(config)
    print("Model_A test starting:")
    test_model(model=model_A, model_name=model_name, train_ds=train_ds, test_ds=test_ds)
    model_A.summary()


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
    checkpoint_callback = ModelCheckpoint(
        f"model_checkpoints/training_checkpoints/{model_name}",
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        mode="auto",
    )
    # Train model

    # train for 20 epochs
    history = model.fit(
        train_ds,
        epochs=30,
        validation_data=test_ds,
        callbacks=[
            stop_early,
            scheduler_callback,
            plateau_callback,
            checkpoint_callback,
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


def build_model_A(config):
    model = tf.keras.Sequential()

    # Input block
    model.add(InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)))
    model.add(
        Conv2D(
            filters=32,
            kernel_size=7,
            strides=1,
            padding="same",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Dropout(rate=0.1))

    # Residual blocks
    for reps, groups in enumerate(config):
        for n in range(groups):
            channels = 32 * (2**reps)
            if n == 0 and reps == 0:
                model.add(ResCell(channels, name=f"res_cell-{reps}-{n}-1"))
            elif n== 0:
                model.add(ResCell(channels, strides=2, name=f"res_cell-{reps}-{n}-1"))
            else:
                model.add(ResCell(channels, name=f"res_cell-{reps}-{n}-2"))

    model.add(GlobalMaxPooling2D())
    model.add(Dense(10, activation="softmax"))
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    main()
    plt.show()
