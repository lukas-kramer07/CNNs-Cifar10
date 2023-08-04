import tensorflow as tf
from tensorflow.keras import layers, models, datasets, utils
import numpy as np

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Data augmentation: horizontal flipping
train_images = np.concatenate((train_images, np.flip(train_images, axis=2)), axis=0)
train_labels = np.concatenate((train_labels, train_labels), axis=0)

# Randomize the training data order
permutation = np.random.permutation(len(train_images))
train_images, train_labels = train_images[permutation], train_labels[permutation]

# Convert labels to one-hot encoding
num_classes = 10
train_labels = utils.to_categorical(train_labels, num_classes)
test_labels = utils.to_categorical(test_labels, num_classes)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 20
batch_size = 64
model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy}")
