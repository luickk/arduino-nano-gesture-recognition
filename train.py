import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
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

def train(filepath, batch_size, epochs):
	x_train, x_test, y_train, y_test = data.load_data(filepath)

	model = simple_rnn(x_train, y_train, batch_size)

	model.fit_generator(data.data_batch_generator(filepath, batch_size, x_train, y_train), steps_per_epoch=x_train.shape[1], epochs=epochs)

	x_test, y_test = data.batch_test_data(x_test, y_test, batch_size)

	print(model.evaluate(x_test, y_test))

	return model, x_test, y_test

def test_predict(model, batch_size, filepath):
	x_train, x_test, y_train, y_test = data.load_data(filepath)
	x_test, y_test = data.batch_test_data(x_test, y_test, batch_size)

	prediction = model.predict(x_test)

	print("prediction: ")
	print(prediction)

	print("test data: ")
	print(y_test)

def main():
	epochs = 1
	batch_size = 2

	# model = train("data/punch_data/recorded_data.csv", batch_size, epochs)

	model, x_test, y_test = train("data/punch_data/test.csv", batch_size, epochs)

	test_predict(model, batch_size, "data/punch_data/sampled_data/punch1.csv")

if __name__ == '__main__':
    main()
