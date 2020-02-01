import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from pandas import read_csv


# float mapping function ported from arduino c++ code
def mapf(val, in_min, in_max, out_min, out_max):
	return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;

# float mapping function ported from arduino c++ code
# cuts off add given mask value to achieve higher sensitivity
def masked_mapf(val, mask_min, mask_max, in_min, in_max, out_min, out_max):
	if val >= mask_min and val <= mask_max:
		return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
	elif val < mask_min:
		return mask_min
	elif val > mask_max:
		return mask_max

# normalizing raw acc/ gyro data to 0-1 because of tensorflow lite implementation
# returns normalized array
def normalize_data(gx, gy, gz, ax, ay, az):
	ngyro_x = masked_mapf(gx, -1000, 1000, -2000, 2000, -4, 4);
	ngyro_y = masked_mapf(gy, -1000, 1000, -2000, 2000, -4, 4);
	ngyro_z = masked_mapf(gz, -1000, 1000, -2000, 2000, -4, 4);

	return [ngyro_x, ngyro_y, ngyro_z, ax, ay, az]

# splits x and y data to x,y arrays
# normalizing gyro data to acc. min/max
def parse_raw_csv(data_stack):
	stack_x = []
	stack_y = []
	for index, row in data_stack.iterrows():
		if row[6] == 0:
			stack_y.append(0)
		else:
			stack_y.append(1)

		stack_x.append(normalize_data(row[0],row[1],row[2],row[3],row[4],row[5]))

		#print(stack_x)

	x = np.array(stack_x)
	y = np.array(stack_y)

	return x,y

# sample data to final x/y test/ train
def load_data(filepath):
	print("- loading data")
	print("Path: " + filepath)

	data_stack = read_csv(filepath)

	x, y = parse_raw_csv(data_stack)

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

	x_train = np.reshape(x_train, (1, x_train.shape[0], x_train.shape[1])) # X.reshape(samples, timesteps, features)
	x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1])) # X.reshape(samples, timesteps, features)

	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)

	print("Raw Data: ")
	print(data_stack)

	print("x train:")
	print(x_train.shape)
	print("x test:")
	print(x_test.shape)
	print("y train:")
	print(y_train.shape)
	print("y test:")
	print(y_test.shape)

	return x_train, x_test, y_train, y_test

def dnn_data_generator(filepath, batch_size, x_train, y_train):
	x_train = np.reshape(x_train, (x_train.shape[1], 6))
	i=0

	while True:
		if i > x_train.shape[1]:
			i=0
		i += 1
		x_data = x_train[i].reshape(1, 6)
		y_data = y_train[i].reshape(1, 2)
		
		yield x_data, y_data

def batch_test_data(test_x, test_y, batch_size):
	for i in range(batch_size):
		# ugly resizing for the test input data but is required for the shifting window technique when using a rnn
		return np.resize(np.array(test_x[:, i*batch_size:i*batch_size+batch_size, :]), (batch_size, batch_size, 6)), np.array(test_y[i*batch_size:i*batch_size+batch_size, :])

def rnn_data_batch_generator(filepath, batch_size, x_train, y_train):
	batch_x, batch_y = [], []
	while True:
		for i in range(batch_size):
			# ugly resizing for the train input data but is required for the shifting window technique when using a rnn
			batch_x = np.array(x_train[:, i*batch_size:i*batch_size+batch_size, :])
			batch_x = np.resize(batch_x, (batch_size, batch_size, 6))

			batch_y = np.array(y_train[i*batch_size:i*batch_size+batch_size, :])

		# print("batch x: " + str(batch_x.shape))
		# print("batch y: " + str(batch_y.shape))
		yield batch_x, batch_y


def cnn_data_batch_generator(filepath, batch_size, x_train, y_train):
	batch_x, batch_y = [], []
	while True:
		for i in range(batch_size):
			batch_x = np.array(x_train[:, i*batch_size:i*batch_size+batch_size, :])

			batch_x = np.resize(batch_x, (batch_size, batch_size, 6))

			batch_y = np.array(y_train[i*batch_size:i*batch_size+batch_size, :])

		# print("batch x: " + str(batch_x.shape))
		# print("batch y: " + str(batch_y.shape))
		yield batch_x, batch_y
