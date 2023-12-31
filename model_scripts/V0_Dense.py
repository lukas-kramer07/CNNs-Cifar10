"""
  referential Dense model
"""
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras import datasets, layers

model_name = "V0_Dense"


# Create a confusion matrix
# Note: Adapted from scikit-learn's plot_confusion_matrix()
def make_confusion_matrix(
    y_true,
    y_pred,
    model_name,
    classes=None,
    figsize=(
        10,
        10,
    ),
    text_size=15,
):
    # Create the confusion matrix
    cm = confusion_matrix(y_true, tf.round(y_pred))
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]

    # Let's prettify it
    fig, ax = plt.subplots(figsize=figsize)
    # Create a matrix plot
    cax = ax.matshow(cm, cmap=plt.cm.Purples)
    fig.colorbar(cax)

    # set labels to be classes
    labels = classes if classes else np.arange(cm.shape[0])
    # Label the axes
    ax.set(
        title=f"Confusion matrix - Fashion_model{model_name}",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels,
    )

    # Set x-labels to the bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Adjust label size
    ax.xaxis.label.set_size(text_size)
    ax.yaxis.label.set_size(text_size)
    ax.title.set_size(text_size)

    # Set coluor threshhold
    threshold = (cm.max() + cm.min()) / 2.0

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            f"{cm[i, j]} ({cm_norm[i,j]*100:.1f}%)",
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
            size=text_size,
        )


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

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()

    model = tf.keras.Sequential(
        [
            layers.Flatten(input_shape=(32, 32, 3)),
            layers.Dense(128, activation="relu"),
            layers.Dense(200, activation="relu"),
            layers.Dense(80, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # Train the model
    history = model.fit(
        train_images,
        train_labels,
        epochs=20,
        validation_data=(test_images, test_labels),
    )

    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.2, 1])
    plt.legend(loc="lower right")
    os.makedirs(f"plots/{model_name}", exist_ok=True)  # Create the "models" folder if it doesn't exist
    plt.savefig(f"plots/{model_name}/history")

    y_probs = model.predict(test_images)
    y_preds = tf.argmax(y_probs, axis=1)
    make_confusion_matrix(
        y_true=test_labels,
        y_pred=y_preds,
        classes=class_names,
        figsize=(13, 13),
        text_size=8,
        model_name=model_name,
    )
    os.makedirs(f"plots/{model_name}", exist_ok=True)  # Create the "models" folder if it doesn't exist
    plt.savefig(f"plots/{model_name}/confusion_matrix")

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"test_acc: {test_acc}; test_loss: {test_loss}")
    model.summary()
    os.makedirs("model_checkpoints", exist_ok=True)  # Create the "models" folder if it doesn't exist
    model.save(f"model_checkpoints/{model_name}")


if __name__ == "__main__":
    main()
    # plt.show()
