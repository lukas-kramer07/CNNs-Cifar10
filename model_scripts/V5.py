"""
  uses further data augmentation than V4 and V2 model architecture. The data augmentation is applied during training
"""
import os

import matplotlib.pyplot as plt
import tensorflow as tf
import utils
from keras.utils import to_categorical
from keras import datasets, layers, models

model_name = "V5"


def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))

    return model


def main():
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = datasets.cifar10.load_data()
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
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    # One_hot encode the labels
    train_labels_hot = to_categorical(train_labels, 10)
    test_labels_hot = to_categorical(test_labels, 10)

    data_augmenter = utils.create_data_augmenter(train_images)
    model = create_model()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    history = model.fit(
        data_augmenter.flow(train_images, train_labels_hot, batch_size=32),
        epochs=20,
        validation_data=(test_images, test_labels_hot),
    )

    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1])
    plt.legend(loc="lower right")
    os.makedirs(f"plots/{model_name}", exist_ok=True)  # Create the "models" folder if it doesn't exist
    plt.savefig(f"plots/{model_name}/history")

    y_probs = model.predict(test_images)
    y_preds = tf.argmax(y_probs, axis=1)
    utils.make_confusion_matrix(
        y_true=test_labels,
        y_pred=y_preds,
        classes=class_names,
        figsize=(13, 13),
        text_size=8,
        model_name=model_name,
    )
    os.makedirs(f"plots/{model_name}", exist_ok=True)  # Create the "models" folder if it doesn't exist
    plt.savefig(f"plots/{model_name}/confusion_matrix")

    test_loss, test_acc = model.evaluate(test_images, test_labels_hot, verbose=1)
    print(f"test_acc: {test_acc}; test_loss: {test_loss}")
    model.summary()
    os.makedirs("model_checkpoints", exist_ok=True)  # Create the "models" folder if it doesn't exist
    model.save(f"model_checkpoints/{model_name}")


if __name__ == "__main__":
    main()
    plt.show()
