import tensorflow as tf
from tensorflow import keras
import cv2

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()

X_train = X_train.astype('float32') / 255

def GAN(codings_size):
    generator = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[codings_size]),
    keras.layers.Dense(150, activation="selu"),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
    ])

    discriminator = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(150, activation="selu"),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(1, activation="sigmoid")
    ])

    gan = keras.models.Sequential([generator, discriminator])
    return gan


def learn_gan(X_train,epochs=10,batch_size=32,codings_size = 30):
    gan = GAN(codings_size)
    generator, discriminator = gan.layers
    discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
    discriminator.trainable = False
    gan.compile(loss="binary_crossentropy", optimizer="rmsprop")
    batch_size = 32
    dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
    gan = train_gan(gan, dataset, batch_size, codings_size,epochs)
    return gan

def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        print("Epoch is " + str(epoch))
        for X_batch in dataset:
            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, tf.expand_dims(X_batch,axis=3)], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)
    return gan    

gan = learn_gan(X_train,epochs=2)    

#Add code to save images 
def save_images(gan,n_images=100,codings_size=100):
    generator, discriminator = gan.layers
    for i in range(n_images):
        noise = tf.random.normal(shape=[1, codings_size])
        generated_images = generator(noise)[0].numpy()*255
        cv2.imwrite('genimage_num_'+str(i)+'.jpg',generated_images)

save_images(gan) 