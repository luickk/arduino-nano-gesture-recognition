import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot

from data_proc import data

def simple_rnn():
	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

def train(train_x, train_y, test_x, test_y):
	verbose, epochs, batch_size = 0, 15, 64
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

	model = simple_renn()

	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)

	_, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)

	return accuracy

def main():
	x_train, x_test, y_train, y_test = data.load_data("data/test_data.csv")
	train(x_train, x_test, y_train, y_test)

if __name__ == '__main__':
    main()
