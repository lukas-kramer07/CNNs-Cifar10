"""
This is the first of three scripts to improve the model architecture using HP-search. This script will use the grid search method
"""

import os
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import tensorflow_datasets as tfds
from V11_E import preprocess_data
from V11_E import visualize_data
from keras.layers import Conv2D, MaxPool2D, Dense, InputLayer, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

IM_SIZE = 32
BATCH_SIZE = 32

#HPARAMS
HP_NUM_UNITS1 = hp.HParam('num_units_1', hp.Discrete([64,128]))
HP_NUM_UNITS2 = hp.HParam('num_units_2', hp.Discrete([32,64]))
HP_NUM_UNITS3 = hp.HParam('num_units_3', hp.Discrete([16,32]))
HP_REGULARIZATION_RATE = hp.HParam('regularization_rate', hp.Discrete([0.001,0.1]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.2,0.3]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.001,0.01,0.0001]))

HPARAMS = [
    HP_NUM_UNITS1,
    HP_NUM_UNITS2,
    HP_NUM_UNITS3,
    HP_REGULARIZATION_RATE,
    HP_DROPOUT,
    HP_LEARNING_RATE,
]
METRICS = [
    hp.Metric(
        "epoch_accuracy",
        group="validation",
        display_name="accuracy (val.)",
    ),
    hp.Metric(
        "epoch_loss",
        group="validation",
        display_name="loss (val.)",
    ),
    hp.Metric(
        "batch_accuracy",
        group="train",
        display_name="accuracy (train)",
    ),
    hp.Metric(
        "batch_loss",
        group="train",
        display_name="loss (train)",
    ),
]

def main():
    """main function that uses preprocess_data and visualize_data from V11_E to prepare the dataset. It then starts a grid search for the best hparams for the model."""
    # load dataset
    (train_ds, test_ds), ds_info = tfds.load("cifar10", split=["train", "test"], as_supervised=True, with_info=True)
    # preprocess
    train_ds, test_ds = preprocess_data(train_ds, test_ds)
    # visualize new data
    visualize_data(train_ds=train_ds, test_ds=test_ds, ds_info=ds_info)
    runall(base_logdir='logs/hp', val_ds=test_ds, train_ds=train_ds)

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
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=hparams[HP_LEARNING_RATE]), loss=CategoricalCrossentropy(), metrics=["accuracy"]
    )
    return model


def run(run_id, base_logdir, hparams, train_ds, val_ds):
    logdir = os.path.join(base_logdir, run_id)
    t_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, update_freq=500, profile_batch=0)
    h_callback = hp.KerasCallback(logdir, hparams)
    model = build_model(hparams)
    result = model.fit(train_ds, validation_data=[val_ds], epochs=4, callbacks=[t_callback, h_callback])


def runall(base_logdir, train_ds, val_ds):
    with tf.summary.create_file_writer(base_logdir).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=METRICS)
    
    session_num = 0
    for num_units_1 in HP_NUM_UNITS1.domain.values:
        for num_units_2 in HP_NUM_UNITS2.domain.values:
            for nums_units_3 in HP_NUM_UNITS3.domain.values:
                for dropout in HP_DROPOUT.domain.values:
                    for regularization in HP_REGULARIZATION_RATE.domain.values:
                        for lr in HP_LEARNING_RATE.domain.values:
                            hparams = {
                                HP_NUM_UNITS1: num_units_1,
                                HP_NUM_UNITS2: num_units_2,
                                HP_NUM_UNITS3: nums_units_3,
                                HP_DROPOUT: dropout,
                                HP_REGULARIZATION_RATE: regularization,
                                HP_LEARNING_RATE: lr
                            } 
                            run_name = "run-%d" % session_num
                            print('--- Starting trial: %s' % run_name)
                            print({h.name: hparams[h] for h in hparams})
                            run(base_logdir=base_logdir, run_id=run_name, hparams=hparams, train_ds=train_ds, val_ds=val_ds)
                            session_num += 1

if __name__ == "__main__":
    main()
