"""
This script is a model version of the ResNet developed in V13_A
"""
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import utils
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
    BatchNormalization,
    Layer,
    Add,
    ReLU,
    GlobalAveragePooling2D,
)
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

IM_SIZE = 32
BATCH_SIZE = 128
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


# TODO: Check for improved architecture
class ResBlock(Layer):
    def __init__(self, channels, stride=1, name="res_block", cut="pre"):
        super(ResBlock, self).__init__(name=name)

        # defining shortcut connection
        if cut == "pre":
            self.res_conv = False
        elif cut == "post":
            self.res_conv = True
        else:
            raise ValueError('Cut type not in ["pre", "post"]')
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


class ResBottleneck(Layer):
    def __init__(self, channels, stride=1, name="res_bottleneck_block", cut="pre"):
        super(ResBottleneck, self).__init__(name=name)

        # defining shortcut connection
        if cut == "pre":
            self.res_conv = False
        elif cut == "post":
            self.res_conv = True
        else:
            raise ValueError('Cut type not in ["pre", "post"]')
        self.conv1 = Conv2D(filters=channels, kernel_size=1, padding="same")
        self.norm1 = BatchNormalization()

        self.conv2 = Conv2D(
            filters=channels, kernel_size=3, strides=stride, padding="same"
        )
        self.norm2 = BatchNormalization()

        self.conv3 = Conv2D(filters=channels * 4, kernel_size=1, padding="same")
        self.norm3 = BatchNormalization()
        self.relu = ReLU()
        if self.res_conv:
            self.conv4 = Conv2D(filters=channels * 4, kernel_size=1, strides=stride)

    def call(self, input, training):
        x = self.conv1(input)
        x = self.norm1(x, training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x, training)
        x = self.relu(x)
        x = self.conv3(x)
        if self.res_conv:
            input = self.conv4(input)
        result = Add()([x, input])
        result = self.norm3(result, training)
        return self.relu(result)


def main():
    """main function that uses preprocess_data and visualize_data from V11_E to prepare the dataset. It then tests all V12 models."""
    # load dataset
    (train_ds, test_ds), ds_info = tfds.load(
        "cifar10", split=["train", "test"], as_supervised=True, with_info=True
    )
    # preprocess
    train_ds, test_ds = utils.preprocess_data(
        train_ds,
        test_ds,
        batch_size=BATCH_SIZE,
        IM_SIZE=IM_SIZE,
        class_names=class_names,
    )
    # visualize new data
    visualize_data(train_ds=train_ds, test_ds=test_ds, ds_info=ds_info)

    # Test model A
    model_name = "V13_A"
    config18 = ((2, 2, 2, 2), ResBlock)
    config34 = ((3, 4, 6, 3), ResBlock)
    config50 = ((3, 4, 6, 3), ResBottleneck)
    config101 = ((3, 4, 23, 3), ResBottleneck)
    config151 = ((3, 8, 36, 3), ResBottleneck)
    model_A = build_model_A(config151)
    print("Model_A test starting:")
    test_model(model=model_A, model_name=model_name, train_ds=train_ds, test_ds=test_ds)


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

    stop_early = EarlyStopping(monitor="val_loss", patience=10, verbose=1)

    def scheduler(epoch, lr):
        if epoch <= 4:
            lr = lr
        elif epoch % 4 == 0:
            lr = (lr * tf.math.exp(-0.35)).numpy()
        return lr

    scheduler_callback = LearningRateScheduler(scheduler, verbose=1)
    plateau_callback = ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.3,
        patience=8,
        verbose=1,
        mode="auto",
        min_delta=0.1,
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
        epochs=40,
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
    reslayer_config, ResBlock = config
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
    for reps, groups in enumerate(reslayer_config):
        for n in range(groups):
            channels = 64 * (2**reps)
            if n == 0 and reps == 0:
                model.add(ResBlock(channels, cut="post", name=f"res_cell-{reps}-{n}-1"))
            elif n == 0:
                model.add(
                    ResBlock(
                        channels, stride=2, cut="post", name=f"res_cell-{reps}-{n}-1"
                    )
                )
            else:
                model.add(ResBlock(channels, cut="pre", name=f"res_cell-{reps}-{n}-2"))

    model.add(GlobalAveragePooling2D())
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
