#Regression problem = predicting a numerical variable from some any other variabkles
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


##Creating data to view and fit
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
Y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

X = tf.constant(X)
Y = tf.constant(Y)

plt.scatter(X,Y)

model = tf.keras.Sequential()
print(Y == X + 10)
#print(np.array(([1,2,3],[4,5,6])))

input_shape = X[0].shape
output_shape = Y[0].shape

##Steps in modelling w tensorFlow
tf.random.set_seed(42)

#1. Create a miodel using sequential
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])


#2.
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"]) #MAE Is mean absolute ewrror, SGD is stochastic gradient descent

#r. Fit the model
model.fit(tf.expand_dims(X, axis = -1), Y, epochs = 5)

print(tf.expand_dims(X, axis = -1))