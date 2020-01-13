import numpy as np
import keras
import random

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import to_categorical
from matplotlib import pyplot

from data_proc import data

def simple_rnn(x, y, batch_size):
	model = Sequential()
	model.add(LSTM(100, input_shape=(batch_size, 6)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(2, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

def simple_cnn(x, y, batch_size):
	model = Sequential()
	model.add(Conv1D(32, kernel_size=10, activation='relu',input_shape=(batch_size, 6)))
	model.add(Conv1D(32, 10, activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

	return model

def train_rnn(filepath, batch_size, epochs):
	x_train, x_test, y_train, y_test = data.load_data(filepath)

	model = simple_rnn(x_train, y_train, batch_size)

	model.fit_generator(data.rnn_data_batch_generator(filepath, batch_size, x_train, y_train), steps_per_epoch=x_train.shape[1], epochs=epochs)

	x_test, y_test = data.batch_test_data(x_test, y_test, batch_size)

	print(model.evaluate(x_test, y_test))

	return model

def train_cnn(filepath, batch_size, epochs):
	x_train, x_test, y_train, y_test = data.load_data(filepath)

	model = simple_cnn(x_train, y_train, batch_size)

	model.fit_generator(data.cnn_data_batch_generator(filepath, batch_size, x_train, y_train), steps_per_epoch=x_train.shape[1], epochs=epochs)

	x_test, y_test = data.batch_test_data(x_test, y_test, batch_size)

	print(model.evaluate(x_test, y_test))

	return model

def save_model(model):
	path = 'model_data/'+str(random.randrange(1000))+'.h5'
	# Save the model
	print("saved model under: " + path)
	model.save(path)

def test_predict(model, batch_size, filepath):
	print(filepath + ": ")
	x_train, x_test, y_train, y_test = data.load_data(filepath)
	x_test, y_test = data.batch_test_data(x_test, y_test, batch_size)

	prediction = model.predict(x_test)

	print("prediction: ")
	print(prediction.argmax(axis=-1))

	print("test data: ")
	print(y_test.argmax(axis=-1))

def main():
	epochs = 2
	batch_size = 30

	# model = load_model("model_data/.h5")

	# model = train_rnn("data/punch_data/recorded_data.csv", batch_size, epochs)

	model = train_cnn("data/punch_data/recorded_data.csv", batch_size, epochs)

	test_predict(model, batch_size, "data/punch_data/sampled_data/punch1.csv")
	# test_predict(model, batch_size, "data/punch_data/sampled_data/invalid.csv")

	save_model(model)

if __name__ == '__main__':
    main()
