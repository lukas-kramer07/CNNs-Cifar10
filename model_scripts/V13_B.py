"""
This script is a model version of the ResNet developed in V13_A
"""
import datetime
import collections
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

ModelParams = collections.namedtuple(
    "ModelParams", ["model_name", "repetitions", "residual_block", "attention"]
)


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
    resnet50 = ModelParams(
        "resnet50", (3, 4, 6, 3), residual_bottleneck_block, None
    )  # ResNet34 or ResNet50
    model_A = ResNet(
        model_params=resnet50,
        input_shape=(IM_SIZE, IM_SIZE, 3),
        input_tensor=None,
        include_top=True,
        classes=10,
    )
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

    stop_early = EarlyStopping(monitor="val_loss", patience=5, verbose=1)

    def scheduler(epoch, lr):
        if epoch <= 3:
            lr = lr
        elif epoch % 3 == 0:
            lr = (lr * tf.math.exp(-0.35)).numpy()
        return lr

    scheduler_callback = LearningRateScheduler(scheduler, verbose=1)
    plateau_callback = ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.3,
        patience=5,
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


import keras.layers as layers
import keras.models as models
from keras import backend


def handle_block_names(stage, block):
    name_base = "stage{}_unit{}_".format(stage + 1, block + 1)
    conv_name = name_base + "conv"
    bn_name = name_base + "bn"
    relu_name = name_base + "relu"
    sc_name = name_base + "sc"
    return conv_name, bn_name, relu_name, sc_name


def get_conv_params(**params):
    default_conv_params = {
        "kernel_initializer": "he_uniform",
        "use_bias": False,
        "padding": "valid",
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    axis = 3 if backend.image_data_format() == "channels_last" else 1
    default_bn_params = {
        "axis": axis,
        "momentum": 0.99,
        "epsilon": 2e-5,
        "center": True,
        "scale": True,
    }
    default_bn_params.update(params)
    return default_bn_params


def residual_bottleneck_block(
    filters, stage, block, strides=None, attention=None, cut="pre"
):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        cut: one of 'pre', 'post'. used to decide where skip connection is taken
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):
        # get params and names of layers
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = layers.BatchNormalization(name=bn_name + "1")(input_tensor)
        x = layers.Activation("relu", name=relu_name + "1")(x)

        # defining shortcut connection
        if cut == "pre":
            shortcut = input_tensor
        elif cut == "post":
            shortcut = layers.Conv2D(
                filters * 4, (1, 1), name=sc_name, strides=strides
            )(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')

        # continue with convolution layers
        x = layers.Conv2D(filters, (1, 1), name=conv_name + "1")(x)

        x = layers.BatchNormalization(name=bn_name + "2")(x)
        x = layers.Activation("relu", name=relu_name + "2")(x)
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + "2")(x)

        x = layers.BatchNormalization(name=bn_name + "3")(x)
        x = layers.Activation("relu", name=relu_name + "3")(x)
        x = layers.Conv2D(filters * 4, (1, 1), name=conv_name + "3")(x)

        # use attention block if defined
        if attention is not None:
            x = attention(x)

        # add residual connection
        x = layers.Add()([x, shortcut])

        return x

    return layer


# -------------------------------------------------------------------------
#   Residual Model Builder
# -------------------------------------------------------------------------


def ResNet(
    model_params,
    input_shape=None,
    input_tensor=None,
    include_top=True,
    classes=10,
    weights="imagenet",
    **kwargs,
):
    print(type(model_params))
    # choose residual block type
    ResidualBlock = model_params.residual_block

    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()
    init_filters = 64

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, name="data")
    else:
        img_input = input_tensor
    # resnet bottom
    x = layers.BatchNormalization(name="bn_data", **no_scale_bn_params)(img_input)
    x = layers.ZeroPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(
        init_filters, (7, 7), strides=(2, 2), name="conv0", **conv_params
    )(x)
    x = layers.BatchNormalization(name="bn0", **bn_params)(x)
    x = layers.Activation("relu", name="relu0")(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="valid", name="pooling0")(x)

    # resnet body
    for stage, rep in enumerate(model_params.repetitions):
        for block in range(rep):
            filters = init_filters * (2**stage)

            # first block of first stage without strides because we have maxpooling before
            if block == 0 and stage == 0:
                x = ResidualBlock(filters,stage,block,strides=(1, 1),cut="post")(x)

            elif block == 0:
                x = ResidualBlock(
                    filters,
                    stage,
                    block,
                    strides=(2, 2),
                    cut="post",
                )(x)

            else:
                x = ResidualBlock(
                    filters,
                    stage,
                    block,
                    strides=(1, 1),
                    cut="pre",
                )(x)

    x = layers.BatchNormalization(name="bn1", **bn_params)(x)
    x = layers.Activation("relu", name="relu1")(x)

    # resnet top
    if include_top:
        x = layers.GlobalAveragePooling2D(name="pool1")(x)
        x = layers.Dense(classes, name="fc1")(x)
        x = layers.Activation("softmax", name="softmax")(x)

    inputs = img_input
    # Create model.
    model = models.Model(inputs, x)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    main()
    plt.show()
