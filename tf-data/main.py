# Important
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

print('Tensorflow version: ' + str(tf.__version__))

# Load images from the Keras dataset.
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Pre-process the images (normalize)
@tf.function
def preprocess(images, labels):
    """I am using @tf.function here for higher efficiency (optional)"""
    images = images / 255
    return images, labels


# Training pipeline
train_data = (
    tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        .shuffle(100)
        .map(preprocess)
        .batch(100)
)

# Test pipeline
test_data = (
    tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        .map(preprocess)
        .batch(10000)
)

# Inspect data.
df_show = next(iter(train_data))
tf.print(df_show)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


# Define Tensorboard callbacks
log_dir="./tensorboard/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)
# Ignore the warnings: https://github.com/tensorflow/tensorflow/issues/31509
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
model.fit(train_data,
          epochs=100,
          validation_data=test_data,
          callbacks=[tensorboard_callback])

# Evaluate output
# Evaluate output.
test_acc = model.evaluate(test_data, verbose=2)

# go to the folder where your <logdir> is located and run tensorboard --logdir <tensorboard>