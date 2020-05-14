import gym 
env = gym.make("CartPole-v1")
obs = env.reset()

"""You can get the list of all available environments by running
gym.envs.registry.all()"""

import tensorflow as tf
from tensorflow import keras
n_inputs = 4 # == env.observation_space.shape[0]
model = keras.models.Sequential([
keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
keras.layers.Dense(1, activation="sigmoid"),
])