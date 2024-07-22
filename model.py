import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

#Getting the handwritten digits dataset
#The dataset is already split into training and test datasets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#Normalize the dataset
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#Creating the neural network model
model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(units=128, activation='relu'),
    Dense(units=128, activation='relu'),
    Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4) 

model.save('DigitRecognizer/model/digitrecognizer.keras')


#Evaluating the model
loss,accuracy= model.evaluate(x_test, y_test)
print(loss)
print(accuracy)

