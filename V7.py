'''
  uses V6 data augmentation and V6 model, but also a callbacker
'''
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import os
import utils
from V6 import create_model
from tensorflow.keras.callbacks import LearningRateScheduler
model_name = 'V7'

def scheduler(epochs,lr):
    return lr * (1/3) if epochs % 5 == 0 and epochs > 5 else lr

def main():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    # One_hot encode the labels
    train_labels_hot = to_categorical(train_labels, 10)
    test_labels_hot = to_categorical(test_labels, 10)

    data_augmenter = utils.create_data_augmenter(train_images)
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    
    # Define the learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    history = model.fit(data_augmenter.flow(train_images, train_labels_hot, batch_size=32), epochs=5,
                        callbacks=[lr_scheduler],
                        validation_data=(test_images, test_labels_hot))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    os.makedirs(f"plots/{model_name}", exist_ok=True)  # Create the "models" folder if it doesn't exist
    plt.savefig(f"plots/{model_name}/history")

    y_probs = model.predict(test_images)
    y_preds = tf.argmax(y_probs, axis=1)
    utils.make_confusion_matrix(y_true=test_labels,
                        y_pred=y_preds,
                        classes=class_names,
                        figsize=(13,13),
                        text_size=8)
    os.makedirs(f"plots/{model_name}", exist_ok=True)  # Create the "models" folder if it doesn't exist
    plt.savefig(f"plots/{model_name}/confusion_matrix")

    test_loss, test_acc = model.evaluate(test_images,  test_labels_hot, verbose=1)
    print(f"test_acc: {test_acc}; test_loss: {test_loss}")
    model.summary()
    os.makedirs(f"models", exist_ok=True)  # Create the "models" folder if it doesn't exist
    model.save(f"models/{model_name}")

if __name__ == "__main__":
  main()
  plt.show()