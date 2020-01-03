from tensorflow import keras
import tensorflow as tf
print(tf.__version__)

#Loading Dataset
(train_images,train_labels),(test_images,test_labels)=keras.datasets.fashion_mnist.load_data()

#Visualisation
import matplotlib.pyplot as plt
plt.imshow(train_images[0])
print(train_labels[0])
print(train_images[0])

#Normalising
training_images  = train_images / 255.0
test_images = test_images / 255.0

#Model Definition - Architecture
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                      tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                      tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
#Model Compilation - Optimizer, loss, metrics
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Training the Model
model.fit(train_images,train_labels,epochs=10)

#Testing the Model
model.evaluate(test_images,test_labels)

#Using Callbacks
class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.6):
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks=myCallback()

model.fit(train_images,train_labels,epochs=10,callbacks=[callbacks])

