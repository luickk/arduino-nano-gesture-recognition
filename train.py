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
	model.add(Conv1D(40, kernel_size=int(batch_size/2), activation='sigmoid',input_shape=(batch_size, 6)))
	model.add(Conv1D(40, int(batch_size/2), activation='sigmoid'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(40, activation='tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

	return model

# by https://github.com/arduino/ArduinoTensorFlowLiteTutorials/blob/master/GestureToEmoji/arduino_tinyml_workshop.ipynb
def simple_dnn(x,y, batch_size):
	model = Sequential()
	model.add(Dense(50, input_dim=6, activation='relu')) # relu is used for performance
	model.add(Dense(15, activation='relu'))
	model.add(Dense(2, activation='softmax')) # softmax is used, because we only expect one gesture to occur per input

	model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

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


def train_dnn(filepath, batch_size, epochs):
	x_train, x_test, y_train, y_test = data.load_data(filepath)

	model = simple_dnn(x_train, y_train, batch_size)

	model.fit_generator(data.dnn_data_generator(filepath, batch_size, x_train, y_train), steps_per_epoch=x_train.shape[1], epochs=epochs)

	x_test, y_test = data.batch_test_data(x_test, y_test, batch_size)

	print(model.evaluate(x_test.reshape(1,6), y_test))

	return model

def save_model(model):
	path = 'model_data/'+str(random.randrange(1000))+'.h5'
	# Save the model
	print("saved model under: " + path)
	model.save(path)

def test_predict(model, batch_size, filepath, batched):
	print(filepath + ": ")
	x_train, x_test, y_train, y_test = data.load_data(filepath)
	x_test, y_test = data.batch_test_data(x_test, y_test, batch_size)

	if not batched:
		x_test = x_test.reshape(1, 6)

	prediction = model.predict(x_test)

	print("prediction: ")
	print(prediction.argmax(axis=-1))

	print("test data: ")
	print(y_test.argmax(axis=-1))

def main():
	cnn_epochs = 2
	dnn_epochs = 1

	cnn_batch_size = 75
	dnn_batch_size = 1

	# model = load_model("model_data/.h5")

	# model = train_rnn("data/flex_data/recorded_data.csv", batch_size, epochs)

	model = train_cnn("data/flex_data/recorded_data.csv", cnn_batch_size, dnn_epochs)

	# model = train_dnn("data/flex_data/recorded_data2.csv", dnn_batch_size, dnn_epochs)

	test_predict(model, cnn_batch_size, "data/flex_data/sampled_data/flex1.csv", True)

	save_model(model)

if __name__ == '__main__':
    main()
