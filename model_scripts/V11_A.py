'''
  uses V2 model and V10 callbacks as well as augmentation 
'''
import utils
from V2 import create_model
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import datetime
model_name = 'V8'
IM_SIZE = 32
BATCH_SIZE = 32
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']


def main():
    #load dataset
    (train_ds, test_ds), ds_info= tfds.load('cifar10', split=['train','test'], as_supervised=True, with_info=True)
    #preprocess
    train_ds, test_ds = preprocess_data(train_ds, test_ds)
    #visualize data
    visualize_data(train_ds=train_ds, ds_info=ds_info)
    #define callbacks
    #custom TensorBoard callback
    Current_Time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = './logs/fit/'+ Current_Time
    class LRTensorBoard(TensorBoard):
        # add other arguments to __init__ if you need
        def __init__(self, log_dir, **kwargs):
            super().__init__(log_dir=log_dir, histogram_freq=3, **kwargs)

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            logs.update({'lr': K.eval(self.model.optimizer.lr)})
            super().on_epoch_end(epoch, logs)

    
    es_callback = EarlyStopping(
        monitor = 'val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto', baseline=None, restore_best_weights = True 
    )
    def scheduler(epoch, lr):
        if epoch < 3:
            lr = lr
        else:
            lr = (lr * tf.math.exp(-0.1)).numpy()
        return lr
    scheduler_callback = LearningRateScheduler(scheduler, verbose=1) 
    checkpoint_callback = ModelCheckpoint(
        f'model_checkpoints/training_checkpoints/{model_name}', monitor = 'val_loss', verbose=0, save_best_only=True, mode='auto'
    ) 
    plateau_callback = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.1,
        patience=3,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0,
    )

    # Create and compile model
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    #train for 20 epochs
    history = model.fit(train_ds, epochs=20, validation_data=test_ds, callbacks=[es_callback, scheduler_callback, checkpoint_callback, plateau_callback,  LRTensorBoard(log_dir=LOG_DIR)])

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
             .map(augment, num_parallel_calls=AUTOTUNE)
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

@tf.function
def augment(Image, Label):
    Image = tf.image.random_flip_left_right(Image)
    Image = tf.image.random_hue(Image, 0.15)
    if tf.random.uniform((), maxval=1, minval=0) < 0.1:
        Image = tf.image.rgb_to_grayscale(Image)
        Image = tf.image.grayscale_to_rgb(Image)
    Image = tf.image.random_brightness(Image, 0.15)
    Image = tf.image.random_contrast(Image, 0, 0.15)
    Image = tf.image.random_jpeg_quality(Image, 90, 100)
    Image = tf.image.random_saturation(Image, 0,0.5)
    return Image, Label
def visualize_data(train_ds, ds_info):
    for i, (image, label) in enumerate(train_ds.take(16)):
        ax = plt.subplot(7,7, i+1)
        plt.imshow(image[0])
        plt.title(ds_info.features['label'].int2str(int(tf.argmax(label[0]))), fontsize=30)
        
        plt.axis("off")
        plt.subplots_adjust(right=6, top=6) 
if __name__ == "__main__": 
  main()
  plt.show()  