#!/usr/bin/env python3

from keras import backend
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
class imageProcess:
	'''
		Class that is used to build a convolutional neural network on images
	'''
	def __init__(self):
		'''
			Initializes the object
		'''
		
		# dimensions of our images.
		self.img_width, self.img_height = 1280, 720

		self.train_data_dir = '/home/ryan/Documents/ML10/train/'
		self.validation_data_dir = '/home/ryan/Documents/ML10/validate'

	def trainData(self):
		'''
			sets up the parameters for the model
		'''
		# used to rescale the pixel values from [0, 255] to [0, 1] interval
		datagen = ImageDataGenerator(rescale=1./255)

		# automagically retrieve images and their classes for train and validation sets
		
		'''
			The nice thing about both of these functions is that they give the the number of images
			found and the number of classes
		'''
		train_generator = datagen.flow_from_directory(
			self.train_data_dir,
			classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
			target_size=(self.img_width, self.img_height),
			batch_size=64,
			class_mode='categorical')

	


if __name__ =='__main__':
	myObj = imageProcess()

	myObj.trainData()
