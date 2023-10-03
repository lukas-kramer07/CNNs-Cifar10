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
    model_eval(history=history, model=model,model_name=model_name, test_ds=test_ds)

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

## Evaluation
def model_eval(model, model_name, history, test_ds):
    utils.change_plot(history)
    # Create the "plots" folder if it doesn't exist
    os.makedirs(f"plots/{model_name}", exist_ok=True)
    plt.savefig(f"plots/{model_name}/history_with_lr_and_change.png")

    y_probs = model.predict(test_ds)
    y_preds = tf.argmax(y_probs, axis=1)
    print(type(y_preds))
    print(y_preds)
    print(test_ds)
    y_true = np.concatenate([y for x, y in test_ds], axis=0) # extract labels from test_ds
    y_true = tf.argmax(y_true, axis=1) # revert from one_hot
    utils.make_confusion_matrix(y_true=y_true,
                        y_pred=y_preds,
                        classes=class_names,
                        figsize=(13,13),
                        text_size=8,
                        model_name=model_name)
    os.makedirs(f"plots/{model_name}", exist_ok=True)  # Create the "models" folder if it doesn't exist
    plt.savefig(f"plots/{model_name}/confusion_matrix")

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"test_acc: {test_acc}; test_loss: {test_loss}")
    model.summary()
    os.makedirs(f"model_checkpoints", exist_ok=True)  # Create the "models" folder if it doesn't exist
    model.save(f"model_checkpoints/{model_name}")#-> move to utils


if __name__ == "__main__": 
  main()
  plt.show()    