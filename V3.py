'''
  uses data augmentation and V2 model architecture
'''
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
import utils
model_name = 'V3'

def main():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Data augmentation: horizontal flipping
    train_images = np.concatenate((train_images, np.flip(train_images, axis=2)), axis=0)
    train_labels = np.concatenate((train_labels, train_labels), axis=0)

    # Randomize the training data order
    permutation = np.random.permutation(len(train_images))
    train_images, train_labels = train_images[permutation], train_labels[permutation]

    # Define the CNN model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',  input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])
    # Compile the model with a lower learning rate
    history = model.fit(train_images, train_labels, epochs=20, 
                      validation_data=(test_images, test_labels))

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
    
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(f"test_acc: {test_acc}; test_loss: {test_loss}")
    model.summary()
    os.makedirs(f"models", exist_ok=True)  # Create the "models" folder if it doesn't exist
    model.save(f"models/{model_name}")

if __name__ == "__main__":
  main()
  plt.show()
