'''
  uses further data augmentation than V3 and V2 model architecture. The data augmentation is applied before training
'''
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
from sklearn.metrics import confusion_matrix
model_name = 'V4'
# Create a confusion matrix
# Note: Adapted from scikit-learn's plot_confusion_matrix()
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10,10,), text_size=15):
  #Create the confusion matrix
  cm = confusion_matrix(y_true, tf.round(y_pred))
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
  n_classes= cm.shape[0]

  # Let's prettify it
  fig, ax = plt.subplots(figsize=figsize)
  #Create a matrix plot
  cax = ax.matshow(cm, cmap=plt.cm.Purples)
  fig.colorbar(cax)

#set labels to be classes
  labels = classes if classes else np.arange(cm.shape[0])
  # Label the axes
  ax.set(title="Confusion matrix - Fashion_modelV1",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels)

  # Set x-labels to the bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  #Adjust label size
  ax.xaxis.label.set_size(text_size)
  ax.yaxis.label.set_size(text_size)
  ax.title.set_size(text_size)

  # Set coluor threshhold
  threshold = (cm.max() + cm.min())/2.

  #Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i,j]*100:.1f}%)",
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
            size=text_size)


def augment(image, label):

  if tf.random.uniform((), minval=0, maxval=1)<0.1:
    image = tf.image.rgb_to_grayscale(image)
    image = tf.tile(image, [1,1,3])
  image = tf.image.random_brightness(image, max_delta=0.1)
  image = tf.image.random_contrast(image, lower=0.1, upper=0.21)
  image = tf.image.random_flip_left_right(image)

  return image, label
def normalize(image,label):
  return tf.cast(image,tf.float32)/255, label
def main():
  (ds_train, ds_test), ds_info =tfds.load(
    "cifar10",
    split=["test","train"],
    shuffle_files=True,
    with_info=True
  )
  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
  # Normalize pixel values to be between 0 and 1
  ds_train, ds_test = ds_train.map(normalize), ds_test.map(normalize)
  ds_train = ds_train.map(augment)

  test_images = np.concatenate([y for x, y in ds_test], axis=0)
  test_labels = np.concatenate([x for x, y in ds_test], axis=0)
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
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

  history = model.fit(ds_train, epochs=20, 
                      validation_data=ds_test)

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
  make_confusion_matrix(y_true=test_labels,
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