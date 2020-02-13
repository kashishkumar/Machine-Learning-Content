import tensorflow as tf
Dense = tf.keras.layers.Dense
fc_model = tf.keras.Sequential(
    [
     tf.keras.Input(shape=(2,)),
     Dense(1024, activation=tf.nn.swish),
     Dense(1024, activation=tf.nn.swish),
     Dense(1, activation=None)])

     