"""
This is the first of three scripts to improve the model architecture using HP-search. This script will use the grid search method
"""

from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from V11_E import preprocess_data
from V11_E import visualize_data

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
    

if __name__ == '__main__':
    main()
    
