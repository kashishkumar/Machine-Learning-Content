# Regression Model
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

input = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.Concatenate(axis=-1)([input, hidden2])
output = keras.layers.Dense(1)(concat)
model2 = keras.models.Model(inputs=[input], outputs=[output])

#model2=keras.models.Sequential([keras.layers.Dense(30,activation='relu',input_shape=(X_train_full.shape[1:])),keras.layers.Dense(1)])
model2.summary()

model2.compile('sgd','mse')

history2 = model2.fit(X_train_scaled,y_train,epochs=30,validation_data=[X_valid_scaled,y_valid])

mse_test = model2.evaluate(X_test_scaled, y_test)

