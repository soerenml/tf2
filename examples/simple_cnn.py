from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

epochs=1
batch_size=250
buffer_size=10000

###########################
# Preprocessing / pipelines
###########################
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

img = test_images[1]
img = img.reshape((1, 28, 28, 1))
print(img.shape)


# Plot image
'''Reshape image to 28x28. Because we are using only one image, dimension one
does not need to be defined. Otherwise, it can be defined as:
x = test_images.reshape((10000,28,28))
plt.imshow(x[0], cmap='gray')
'''
plt.imshow(test_images[1].reshape((28,28)), cmap='gray')
plt.show()

'''categorical_crossentropy is used when labels are one-hot encoded (i.e.,
labels become categorical), sparse_categorical_crossentropy is used when
labels do not use one-hot encoding'''

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Use tf.data for pipelining
train_dataset=(tf.data.Dataset.from_tensor_slices((train_images, train_labels))
               .batch(1000)
               .shuffle(10000))

test_dataset=(tf.data.Dataset.from_tensor_slices((test_images, test_labels))
              .batch(10000)
              .shuffle(10000))

#################
# Build the model
#################

# Tensorboard preparations
logdir="./tboard/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# Add callbacks for tensorboard
tensorboard_callback = (keras.callbacks.
                        TensorBoard(log_dir=logdir,write_graph=True))

# Build model
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile model
model1.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Fit model
model1.fit(train_dataset,
           epochs=epochs,
           validation_data=(test_dataset),
           callbacks=[tensorboard_callback])

# Show model architecture
model1.summary()

# Evaluate model
model1.evaluate(test_dataset)

### Visualize network
layer = 2
layer_outputs = [layer.output for layer in model1.layers[:layer]]
activation_model = tf.keras.models.Model(inputs=model1.input, outputs=layer_outputs)
activations = activation_model.predict(img)
first_layer_activation = activations[0]
plt.matshow(first_layer_activation[0, :, :, layer], cmap='viridis')
plt.show()