import tensorflow as tf
import numpy as np
from sklearn.datasets import load_sample_image
china=load_sample_image("china.jpg")/255
flower=load_sample_image("flower.jpg")/255
images=np.array([china, flower])
batch_size, height, width, channels = images.shape
# Create 2 filters
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1 # vertical line
filters[3, :, :, 1] = 1 # horizontal line
outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")
import matplotlib.pyplot as plt
plt.imshow(outputs[0, :, :, 1], cmap="gray") # plot 1st image's 2nd feature map
plt.show()


X_train, X_test = tf.keras.datasets.fashion_mnist.load_data()
conv = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="SAME", activation="relu")

#Convolutional Layer
conv = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,padding="SAME", activation="relu") 

"""If training crashes because of an out-of-memory error, you can try
reducing the mini-batch size. Alternatively, you can try reducing
dimensionality using a stride, or removing a few layers. Or you can
try using 16-bit floats instead of 32-bit floats. Or you could distrib‐
ute the CNN across multiple devices. """

#Max Pool
max_pool = tf.keras.layers.MaxPool2D(pool_size=2)
output = tf.nn.max_pool(images,ksize=(1, 1, 1, 3), strides=(1, 1, 1, 3), padding="VALID")

from functools import partial
DefaultConv2D=partial(tf.keras.layers.Conv2D, kernel_size=3 , activation='relu', padding="SAME")


model = tf.keras.models.Sequential([
DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
tf.keras.layers.MaxPooling2D(pool_size=2),
DefaultConv2D(filters=128),
DefaultConv2D(filters=128),
tf.keras.layers.MaxPooling2D(pool_size=2),
DefaultConv2D(filters=256),
DefaultConv2D(filters=256),
tf.keras.layers.MaxPooling2D(pool_size=2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(units=128, activation='relu'),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(units=64, activation='relu'),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(units=10, activation='softmax'),
])

"""In TensorFlow, each input image is typically represented as a 3D tensor of shape
[height, width, channels] . A mini-batch is represented as a 4D tensor of shape
[mini-batch size, height, width, channels] . The weights of a convolutional
layer are represented as a 4D tensor of shape [f h , f w , f n′ , f n ]. The bias terms of a convo‐
lutional layer are simply represented as a 1D tensor of shape [f n ]"""    