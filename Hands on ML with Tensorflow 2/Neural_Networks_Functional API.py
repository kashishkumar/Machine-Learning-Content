# Functional APIs give the flexibility to build any sort of model as we may envision


from tensorflow import keras

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

input_A = keras.layers.Input(shape=[5])
input_B = keras.layers.Input(shape=[6])
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])
model.summary()
model.compile(loss="mse", optimizer="sgd")

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

X_train_A, X_train_B = X_train_scaled[:, :5], X_train_scaled[:, 2:]
X_valid_A, X_valid_B = X_valid_scaled[:, :5], X_valid_scaled[:, 2:]
X_test_A, X_test_B = X_test_scaled[:, :5], X_test_scaled[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]
history = model.fit((X_train_A, X_train_B), y_train, epochs=20, validation_data=((X_valid_A, X_valid_B), y_valid))
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((X_new_A, X_new_B))

"""There are also many use cases in which you may want to have multiple outputs:
• The task may demand it, for example you may want to locate and classify the
main object in a picture. This is both a regression task (finding the coordinates of
the object’s center, as well as its width and height) and a classification task.
• Similarly, you may have multiple independent tasks to perform based on the
same data. Sure, you could train one neural network per task, but in many cases
you will get better results on all tasks by training a single neural network with
one output per task. This is because the neural network can learn features in the
data that are useful across tasks.
• Another use case is as a regularization technique (i.e., a training constraint whose
objective is to reduce overfitting and thus improve the model’s ability to general‐
ize). For example, you may want to add some auxiliary outputs in a neural net‐
work architecture (see Figure 10-15) to ensure that the underlying part of the
network learns something useful on its own, without relying on the rest of the
network."""

# For Multiple Outputs
output = keras.layers.Dense(1)(concat)
aux_output = keras.layers.Dense(1)(hidden2)
model = keras.models.Model(inputs=[input_A, input_B],
outputs=[output, aux_output])

model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer="sgd")
# Different losses and corresponding weights to each loss can be given.
# Training labels and validation labels are also needed to be given separately
  
history = model.fit([X_train_A, X_train_B], [y_train, y_train], epochs=20, validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))

total_loss, main_loss, aux_loss = model.evaluate([X_test_A, X_test_B], [y_test, y_test])

y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])

model.save("my_keras_model.h5")

model2 = keras.models.load_model("my_keras_model.h5")

model2.summary()

# Callbacks are used to tell the fit method to save checkpoints

