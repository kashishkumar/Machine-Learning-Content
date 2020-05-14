#Data - MNIST
#Model - Linear model with sigmoid
#Algorithm - Gradient Descent
#Cost Function - Cross Entropy
import numpy as np
from tensorflow import keras 
(train_images,train_labels),(test_images,test_labels)=keras.datasets.fashion_mnist.load_data()


def cross_entropy():
