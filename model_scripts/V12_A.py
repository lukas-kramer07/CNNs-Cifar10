"""
This is the first of three scripts to improve the model architecture using HP-search. This script will use the grid search method
"""
import tensorflow as tf
import tensorflow_datasets as tfds
from V11_E import preprocess_data
from V11_E import visualize_data
from keras.layers import Conv2D, MaxPool2D, Dense, InputLayer, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy

IM_SIZE = 32
BATCH_SIZE = 32


def main():
    """main function that uses preprocess_data and visualize_data from V11_E to prepare the dataset. It then starts a grid search for the best hparams for the model."""
    # load dataset
    (train_ds, test_ds), ds_info = tfds.load("cifar10", split=["train", "test"], as_supervised=True, with_info=True)
    # preprocess
    train_ds, test_ds = preprocess_data(train_ds, test_ds)
    # visualize new data
    visualize_data(train_ds=train_ds, test_ds=test_ds, ds_info=ds_info)


def build_model(hparams):
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
                kernel_regularizer=tf.keras.regularizers.L2(hparams[HP_REGULARIZATION_RATE]),
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=2, strides=2),
            Dropout(rate=hparams[HP_DROPOUT]),
            #
            # Second Convolutional block
            Conv2D(
                filters=32,
                kernel_size=3,
                strides=1,
                padding="valid",
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(hparams[HP_REGULARIZATION_RATE]),
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=2, strides=2),
            Dropout(rate=hparams[HP_DROPOUT]),
            #
            # Third Convolutional block
            Conv2D(
                filters=64,
                kernel_size=3,
                strides=1,
                padding="valid",
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(hparams[HP_REGULARIZATION_RATE]),
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=2, strides=2),
            Dropout(rate=hparams[HP_DROPOUT]),
            # Dense block
            Flatten(),
            Dense(
                hparams[HP_NUM_UNITS1],
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(hparams[HP_REGULARIZATION_RATE]),
            ),
            BatchNormalization(),
            Dropout(rate=hparams[HP_DROPOUT]),
            Dense(
                hparams[HP_NUM_UNITS2],
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(hparams[HP_REGULARIZATION_RATE]),
            ),
            BatchNormalization(),
            Dropout(rate=hparams[HP_DROPOUT]),
            Dense(
                hparams[HP_NUM_UNITS3],
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(hparams[HP_REGULARIZATION_RATE]),
            ),
            BatchNormalization(),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=hparams[HP_LEARNING_RATE]), loss=BinaryCrossentropy(), metrics=["accuracy"]
    )
    return model


def run(run_id, base_logdir, hparams):
    # TODO
    pass


def runall(base_logdir, METRICS, HPARAMS):
    pass


if __name__ == "__main__":
    main()
