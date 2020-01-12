import numpy as np
import keras

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import to_categorical
from matplotlib import pyplot
from datetime import date

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
	model.add(Conv1D(32, kernel_size=2, activation='relu',input_shape=(batch_size, 6)))
	model.add(Conv1D(32, 2, activation='relu'))
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

	model.fit_generator(data.data_batch_generator(filepath, batch_size, x_train, y_train), steps_per_epoch=x_train.shape[1], epochs=epochs)

	x_test, y_test = data.batch_test_data(x_test, y_test, batch_size)

	print(model.evaluate(x_test, y_test))

	return model

def train_cnn(filepath, batch_size, epochs):
	x_train, x_test, y_train, y_test = data.load_data(filepath)

	model = simple_cnn(x_train, y_train, batch_size)

	model.fit_generator(data.data_batch_generator(filepath, batch_size, x_train, y_train), steps_per_epoch=x_train.shape[1], epochs=epochs)

	# x_test, y_test = data.batch_test_data(x_test, y_test, batch_size)

	# print(model.evaluate(x_test, y_test))

	return model

def save_model(model):
	path = 'model_data/'+str(date.today())+'.h5'
	# Save the model
	model.save(path)

	# data.export_lite_model(model)


def load_model(path):
	# Recreate the exact same model purely from the file
	return load_model(model)


def test_predict(model, batch_size, filepath):
	x_train, x_test, y_train, y_test = data.load_data(filepath)
	x_test, y_test = data.batch_test_data(x_test, y_test, batch_size)

	prediction = model.predict(x_test)

	print("prediction: ")
	print(np.around(prediction))

	print("test data: ")
	print(y_test)

def main():
	epochs = 1
	batch_size = 10

	# model = train("data/punch_data/recorded_data.csv", batch_size, epochs)

	model = train_cnn("data/punch_data/test.csv", batch_size, epochs)

	test_predict(model, batch_size, "data/punch_data/sampled_data/punch1.csv")

	save_model(model)

if __name__ == '__main__':
    main()
