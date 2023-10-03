'''
  uses V2 model and tfds instead of keras.datasets. All subsequent models will use tfds
'''
import os
import utils
from V2 import create_model
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
model_name = 'V8'
IM_SIZE = 32
BATCH_SIZE = 32

def main():
    #load dataset
    (train_ds, test_ds), ds_info= tfds.load('cifar10', split=['train','test'], as_supervised=True, with_info=True)
    #preprocess
    train_ds, test_ds = preprocess_data(train_ds, test_ds)
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    history = model.fit(train_ds, epochs=3, validation_data=test_ds)


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
    return Image/255.0, Label

def model_eval(model, history, test_ds):
    '''# Calculate the change in accuracy from the previous epoch
    accuracy_changes = [0] + [history.history['accuracy'][i] - history.history['accuracy'][i-1] for i in range(1, len(history.history['accuracy']))]

    plt.figure(figsize=(10, 6))

    # Plot accuracy and validation accuracy
    accuracy_line, = plt.plot(history.history['accuracy'], label='accuracy', color='b')
    val_accuracy_line, = plt.plot(history.history['val_accuracy'], label='val_accuracy', color='g')

    # Plot change in accuracy
    accuracy_change_line, = plt.plot(accuracy_changes, label='Accuracy Change', color='r', linestyle='dashed')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.1])

    # Create a twin axis for the learning rate
    ax2 = plt.gca().twinx()
    lr_line, = ax2.plot(history.history['lr'], label='Learning Rate', color='m', linestyle='dotted')
    ax2.set_ylabel('Learning Rate')

    # Combine the legend entries from both axes
    lines = [accuracy_line, val_accuracy_line, accuracy_change_line, lr_line]
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, loc='upper left')

    plt.title('Accuracy, Validation Accuracy, Accuracy Change, and Learning Rate')
    plt.tight_layout()

    # Create the "plots" folder if it doesn't exist
    os.makedirs(f"plots/{model_name}", exist_ok=True)

    plt.savefig(f"plots/{model_name}/history_with_lr_and_change.png")''' 


    '''y_probs = model.predict(test_images)
    y_preds = tf.argmax(y_probs, axis=1)
    utils.make_confusion_matrix(y_true=test_labels,
                        y_pred=y_preds,
                        classes=class_names,
                        figsize=(13,13),
                        text_size=8,
                        model_name=model_name)
    os.makedirs(f"plots/{model_name}", exist_ok=True)  # Create the "models" folder if it doesn't exist
    plt.savefig(f"plots/{model_name}/confusion_matrix")
    test_loss, test_acc = model.evaluate(test_images,  test_labels_hot, verbose=1)
    print(f"test_acc: {test_acc}; test_loss: {test_loss}")
    model.summary()
    os.makedirs(f"model_checkpoints", exist_ok=True)  # Create the "models" folder if it doesn't exist
    model.save(f"model_checkpoints/{model_name}")''' #-> move to utils


if __name__ == "__main__": 
  main()
  #plt.show()    