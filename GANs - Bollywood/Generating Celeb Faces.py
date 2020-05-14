import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt

"""
Make it fast and be able to run in memory
Connect GPU
Define right architecture
Run for large number of epochs
"""

#path="/home/test/Desktop/Projects/GANs - Bollywood/IMFDB_final/**/**/*.jpg"
data_dir="/home/test/Desktop/Projects/Machine-Learning-Content/GANs - Bollywood"
#configfiles = glob.glob(path,recursive=True)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
BATCH_SIZE=32
IMG_HEIGHT=128
IMG_WIDTH=128
train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    )

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        #plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')
        
image_batch, label_batch = next(train_data_gen)
show_batch(image_batch, label_batch) 

def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        print("Epoch is " + str(epoch))
        for X_batch in dataset:
            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, tf.convert_to_tensor(X_batch[0], dtype=tf.float32)], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)
    return gan    

#Defining DCGAN Architecture
def DCGAN(codings_size,IMG_HEIGHT,IMG_WIDTH):
    generator = keras.models.Sequential([
    keras.layers.Dense( 32* 32 * 128, input_shape=[codings_size]),
    keras.layers.Reshape([32, 32, 128]),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same",
    activation="selu"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding="same",
    activation="tanh")
    ])

    discriminator = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="same",
    activation=keras.layers.LeakyReLU(0.2),
    input_shape=[IMG_HEIGHT, IMG_WIDTH, 3]),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="same",
    activation=keras.layers.LeakyReLU(0.2)),
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation="sigmoid")
    ])

    gan = keras.models.Sequential([generator, discriminator])
    return gan

# Complete
def learn_dcgan(X_train, epochs=10, batch_size=32,codings_size = 100):
    gan = DCGAN(codings_size,next(X_train)[0].shape[1],next(X_train)[0].shape[2])
    generator, discriminator = gan.layers
    discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
    discriminator.trainable = False
    gan.compile(loss="binary_crossentropy", optimizer="rmsprop")
    batch_size = batch_size
    #dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
    #dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
    dataset=X_train
    gan = train_gan(gan, dataset, batch_size, codings_size, epochs)
    return gan


gan = learn_dcgan(train_data_gen,epochs=2)    

#Add code to save images 
def save_images(gan,n_images=100,codings_size=100):
    generator, discriminator = gan.layers
    for i in range(n_images):
        noise = tf.random.normal(shape=[1, codings_size])
        generated_images = generator(noise)[0].numpy()*255
        cv2.imwrite('genimage_num_'+str(i)+'.jpg',generated_images)

save_images(gan)