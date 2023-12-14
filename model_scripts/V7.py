"""
  uses V6 data augmentation and V6 model as well as a callbacker to progressively reduce lr
"""
import os

import matplotlib.pyplot as plt
import tensorflow as tf
import utils
from keras.utils import to_categorical
from keras import datasets
from keras.callbacks import LearningRateScheduler
from V6 import create_model

model_name = "V7"


def scheduler(epochs, lr):
    return lr * (1 / 4) if epochs % 7 == 0 and epochs > 7 else lr


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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # Define the learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    history = model.fit(
        data_augmenter.flow(train_images, train_labels_hot, batch_size=32),
        epochs=3,
        callbacks=[lr_scheduler],
        validation_data=(test_images, test_labels_hot),
    )

    # Calculate the change in accuracy from the previous epoch
    accuracy_changes = [0] + [
        history.history["accuracy"][i] - history.history["accuracy"][i - 1]
        for i in range(1, len(history.history["accuracy"]))
    ]

    plt.figure(figsize=(10, 6))

    # Plot accuracy and validation accuracy
    (accuracy_line,) = plt.plot(history.history["accuracy"], label="accuracy", color="b")
    (val_accuracy_line,) = plt.plot(history.history["val_accuracy"], label="val_accuracy", color="g")

    # Plot change in accuracy
    (accuracy_change_line,) = plt.plot(accuracy_changes, label="Accuracy Change", color="r", linestyle="dashed")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1.1])

    # Create a twin axis for the learning rate
    ax2 = plt.gca().twinx()
    (lr_line,) = ax2.plot(history.history["lr"], label="Learning Rate", color="m", linestyle="dotted")
    ax2.set_ylabel("Learning Rate")

    # Combine the legend entries from both axes
    lines = [accuracy_line, val_accuracy_line, accuracy_change_line, lr_line]
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, loc="upper left")

    plt.title("Accuracy, Validation Accuracy, Accuracy Change, and Learning Rate")
    plt.tight_layout()

    # Create the "plots" folder if it doesn't exist
    os.makedirs(f"plots/{model_name}", exist_ok=True)

    plt.savefig(f"plots/{model_name}/history_with_lr_and_change.png")

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
