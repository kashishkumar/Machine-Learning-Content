# Perceptron Model
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
iris = load_iris()
X = iris.data[:, (2, 3)] # petal length, petal width
y = (iris.target == 0).astype(np.int) # Iris Setosa?
per_clf = Perceptron()
per_clf.fit(X, y) # Training
y_pred = per_clf.predict([[2, 0.5]]) # Inference


import tensorflow as tf2
from tensorflow import keras
tf2.__version__
keras.__version__


#MultiClass Classification using ANN 
# Building an Image Classifier Using the Sequential API
# MNIST (70,000 grayscale images of 28×28 pixels each, with 10 classes)
fashion_mnist=tf2.keras.datasets.fashion_mnist
(X_train,Y_train),(X_test,Y_test)=fashion_mnist.load_data()
X_valid, X_train_s = X_train[:5000]/255, X_train[5000:]/255
Y_valid, Y_train_s = Y_train[:5000], Y_train[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
class_names[Y_train[0]]

#Model Using Keras
model=keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(keras.layers.Dense(300,'relu'))
model.add(keras.layers.Dense(100,'relu'))
model.add(keras.layers.Dense(10,'softmax'))

model.summary()
optimizer=keras.optimizers.SGD()
loss=keras.losses.sparse_categorical_crossentropy
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

# Sparse Categorical Crossentropy because we have [0-9] vectors (sparse labels) and not one hot encoded vectors
# Softmax because we have multiple classes, 10

"""If you want to convert sparse labels (i.e., class indices) to one-hot
vector labels, you can use the keras.utils.to_categorical()
function. To go the other way round, you can just use the np.arg
max() function with axis=1."""

history=model.fit(X_train_s,Y_train_s,epochs=30,validation_data=[X_valid,Y_valid])

""" If the training set was very skewed, with some classes being overrepresented and oth‐
ers underrepresented, it would be useful to set the class_weight argument when
calling the fit() method, giving a larger weight to underrepresented classes, and a
lower weight to overrepresented classes. """

"""Similarly we have an argument called as sample weight used for weighting the loss 
function. TO apply a different weight to every timestamp of every sample. More like a 
temporal component to the whole thing, only supported when X is an array"""

history.params  # Training parameters
history.epoch   # Epochs
history.history # Loss and extra metrics at each epoch

import pandas as pd

import matplotlib.pyplot as plt 

training=pd.DataFrame(history.history)
training.plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

# Tune the model hyperparamters for better performance
# Number of layers, number of neurons, activation functions, batch size, epochs
# There are automated ways of doing hyperparameter tuning as well

model.evaluate(X_test,Y_test)

# Do not optimize hyperparameters on test set, or you would have a biased estimate of the model's true accuracy

X_new=X_test[0:10]
model.predict(X_new)
model.predict_classes(X_new)


# Regression Model
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

model2=keras.models.Sequential([keras.layers.Dense(30,activation='relu',input_shape=(X_train_full.shape[1:])),keras.layers.Dense(1)])
model2.summary()

model2.compile('sgd','mse')

history2 = model2.fit(X_train_scaled,y_train,epochs=30,validation_data=[X_valid_scaled,y_valid])

mse_test = model.evaluate(X_test_scaled, y_test)
X_new = X_test[:3] # pretend these are new instances
y_pred = model.predict(X_new)


model.save("my_keras_model.h5")

# Callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid), callbacks=[checkpoint_cb,early_stopping_cb])
model = keras.models.load_model("my_keras_model.h5") # rollback to best model

# Custom Callback
class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))
callback=PrintValTrainRatioCallback.on_batch_end(epoch=100,logs=10)

#Using Tensorbaord - callbacks = [tensorboard_cb]
import os
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)        
run_logdir=get_run_logdir()        
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

history=model.fit(X_train_s,Y_train_s,epochs=30,validation_data=[X_valid,Y_valid],callbacks=[tensorboard_cb])

#Hyperparameter Tuning
# Write a function for scikitlearn wrapper
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=(X_train.shape[1],X_train.shape[2])):
    model = keras.models.Sequential()
    options = {"input_shape": input_shape}
    model.add(keras.layers.Flatten(input_shape=(X_train.shape[1],X_train.shape[2])))
    for layer in range(n_hidden-1):
        options = {}
        model.add(keras.layers.Dense(n_neurons, activation="relu", **options))
    model.add(keras.layers.Dense(1, **options))
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# Keras Wrapper
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

# Hyperparameter Dictionary
param_distribs = {"n_hidden": [0, 1, 2, 3], "n_neurons": np.arange(1, 100, 5), "learning_rate": reciprocal(3e-4, 3e-2)}
rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train_s, Y_train_s, epochs=20, validation_data=(X_valid,Y_valid), callbacks=[keras.callbacks.EarlyStopping(patience=10)])
rnd_search_cv.best_params_
rnd_search_cv.best_score_

"""
• Hyperopt: a popular Python library for optimizing over all sorts of complex
search spaces (including real values such as the learning rate, or discrete values
such as the number of layers).
• Hyperas, kopt or Talos: optimizing hyperparameters for Keras model (the first
two are based on Hyperopt).
• Scikit-Optimize (skopt): a general-purpose optimization library. The Bayes
SearchCV class performs Bayesian optimization using an interface similar to Grid
SearchCV .
• Spearmint: a Bayesian optimization library.
• Sklearn-Deap: a hyperparameter optimization library based on evolutionary
algorithms, also with a GridSearchCV -like interface.     
""" 

# Initialisation for neural networks
keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal") # Variance based on number of input neurons 

# Variance based on average number of neurons
he_avg_init = keras.initializers.VarianceScaling(scale=2., mode='fan_avg', distribution='uniform')
keras.layers.Dense(10, activation="sigmoid", kernel_initializer=he_avg_init)

#Leaky ReLU
leaky_relu = keras.layers.LeakyReLU(alpha=0.2)
layer = keras.layers.Dense(10, activation=leaky_relu, kernel_initializer="he_normal")

# SELU Activation Fuction
layer = keras.layers.Dense(10, activation="selu", kernel_initializer="lecun_normal")

# Batch Normalisation 
keras.layers.BatchNormalization() # For each layer

[(var.name, var.trainable) for var in model.layers[1].variables] # Gives the parameter description along with the fact that it is trainable or not

#Gradient Clipping
optimizer = keras.optimizers.SGD(clipvalue=1.0)
model.compile(loss="mse", optimizer=optimizer)


# Transfer Learning from model A
model_A = keras.models.load_model("my_model_A.h5")
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))

#Cloning model since model_A may get modified as it shares the same memory as model_B_on_A
model_A_clone=keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights)

#Freezing the previous layers
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False
model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

#Its a good idea to freeze the layers for few epochs so the last layer could learn 

history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
validation_data=(X_valid_B, y_valid_B))
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True
optimizer = keras.optimizers.SGD(lr=1e-4) # the default lr is 1e-3
model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
validation_data=(X_valid_B, y_valid_B))


#Implementing Dropout for training machine learning models
model = keras.models.Sequential([keras.layers.Flatten(input_shape=[28, 28]),
keras.layers.Dropout(rate=0.2),
keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
keras.layers.Dropout(rate=0.2),
keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
keras.layers.Dropout(rate=0.2),
keras.layers.Dense(10, activation="softmax")])
    
#Momentum Optimisation
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)

#nesterov Acceleration Gradient
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)    

#AdaGrad
keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

#RMSProp
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)

#Adam
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

#Learning rate decay
optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)

#Reguralisation 
layer = keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.01))
    
    