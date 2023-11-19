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
    Flatten,
    BatchNormalization,
    Dropout,
    Layer,
    Add,
    AveragePooling2D,
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
    model_name = "V13_A"
    model_A = build_model_A()
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
            kernel_regularizer=tf.keras.regularizers.L2(0.01),
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=2, strides=2))
    model.add(Dropout(rate=0.1))

    # Residual blocks
    res_blocks = tf.keras.Sequential()
    for n in range(3):
        channels = 32 * 2**n
        res_blocks.add(ResCell(channels, strides=2, name=f"res_cell-{n}-1"))
        res_blocks.add(ResCell(channels, name=f"res_cell-{n}-2"))
        res_blocks.add(ResCell(channels, name=f"res_cell-{n}-3"))

    res_blocks.add(AveragePooling2D(pool_size=(2, 2), padding="same"))
    res_blocks.add(Flatten())
    model.add(res_blocks)

    # Output block
    output = tf.keras.Sequential(
        [
            Dense(
                300,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.01),
            ),
            BatchNormalization(),
            Dropout(rate=0.1),
            Dense(
                150,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.01),
            ),
            BatchNormalization(),
            Dropout(rate=0.1),
            Dense(
                50,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.01),
            ),
            BatchNormalization(),
            Dense(10, activation="softmax"),
        ]
    )

    model.add(output)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    main()
    plt.show()
