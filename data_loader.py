import numpy as np
import cv2
import glob


def load_data_wrapper(IMAGE_SIZE):

	num_train = 1000
	num_test = 100

	train_face_path = 'Data/train_face_images/'
	train_nonface_path = 'Data/train_nonface_images/'
	test_face_path = 'Data/test_face_images/'
	test_nonface_path = 'Data/test_nonface_images/'

	train_x, train_y = [], []
	test_x, test_y = [], []
	
	files_train_face = glob.glob(train_face_path+'*.jpg')
	files_train_nonface = glob.glob(train_nonface_path+'*.jpg')
	files_test_face = glob.glob(test_face_path+'*.jpg')
	files_test_nonface = glob.glob(test_nonface_path+'*.jpg')

	row,col,channel = cv2.imread(files_train_face[0]).shape
	row, col, channel = IMAGE_SIZE,IMAGE_SIZE,1
	print("Shape of Image:", row, col, channel)

	for f in files_train_face[0:num_train]:
		img = cv2.imread(f,0)
		img = cv2.resize(img,(row,col))
		train_x.append(img.reshape(row*col*channel))
		train_y.append(1)

	for f in files_train_nonface[0:num_train]:
		img = cv2.imread(f,0)
		img = cv2.resize(img,(row,col))
		train_x.append(img.reshape(row*col*channel))
		train_y.append(0)

	for f in files_test_face[0:num_test]:
		img = cv2.imread(f,0)
		img = cv2.resize(img,(row,col))
		test_x.append(img.reshape(row*col*channel))
		test_y.append(1)

	for f in files_test_nonface[0:num_test]:
		img = cv2.imread(f,0)
		img = cv2.resize(img,(row,col))
		test_x.append(img.reshape(row*col*channel))
		test_y.append(0)


	return train_x, train_y, test_x, test_y

if __name__ == '__main__':
	data_loader_wrapper()