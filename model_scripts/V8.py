'''
  uses V2 model and tfds instead of keras.datasets. All subsequent models will use tfds
'''
import os
import utils
from V2 import create_model
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np
model_name = 'V8'
IM_SIZE = 32
BATCH_SIZE = 32
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']


def main():
    #load dataset
    (train_ds, test_ds), ds_info= tfds.load('cifar10', split=['train','test'], as_supervised=True, with_info=True)
    #preprocess
    train_ds, test_ds = preprocess_data(train_ds, test_ds)


    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    #train for 20 epochs
    history = model.fit(train_ds, epochs=20, validation_data=test_ds)

    #model_evaluation
    utils.model_eval(history=history, model=model,model_name=model_name, test_ds=test_ds, class_names=class_names)

## Preprocessing the dataset
def preprocess_data(train_ds, test_ds):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train_ds = (
             train_ds
             .map(resize_rescale, num_parallel_calls=AUTOTUNE)
             .cache()
             .shuffle(8, reshuffle_each_iteration=True)
             .batch(BATCH_SIZE)
             .prefetch(AUTOTUNE)  
            )
        test_ds = (
             test_ds
             .map(resize_rescale, num_parallel_calls=AUTOTUNE)
             .batch(BATCH_SIZE)
             .prefetch(AUTOTUNE) 
            )
        return train_ds, test_ds
def resize_rescale(Image, Label):
    Image = tf.image.resize(Image,(IM_SIZE,IM_SIZE))
    Label = tf.one_hot(Label, len(class_names)) #one_hot_encode
    return Image/255.0, Label

if __name__ == "__main__": 
  main()
  plt.show()    