import numpy as np
from sklearn.model_selection import train_test_split
from pandas import read_csv

# splits x and y data to x,y arrays
def parse_raw_csv(data_stack):
	stack_x = []
	stack_y = []
	for index, row in data_stack.iterrows():
		if row[6] == 0:
			stack_y.append(0)
		else:
			stack_y.append(1)

		stack_x.append([row[0],row[1],row[2],row[3],row[4],row[5]])

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

	print("Data: ")
	print(data_stack)

	print("x train, test:")
	print(x_train.shape)
	print(x_test.shape)
	print("y train, test:")
	print(y_train.shape)
	print(y_test.shape)

	return x_train, x_test, y_train, y_test
