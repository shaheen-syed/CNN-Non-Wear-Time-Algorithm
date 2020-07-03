"""
	IMPORT PACKAGES
"""
import logging
import time
import psutil
import os
import sys
import glob2
import numpy as np
import csv
from datetime import datetime

def set_logger(folder_name = 'logs'):
	"""
	Set up the logging to console layout

	Parameters
	----------
	folder_name : string, optional
		name of the folder where the logs can be saved to

	Returns
	--------
	logger: logging
		logger to console and file
	"""

	# create the logging folder if not exists
	create_directory(folder_name)

	# define the name of the log file
	log_file_name = os.path.join(folder_name, '{:%Y%m%d%H%M%S}.log'.format(datetime.now()))

	# create a new logger but use root
	logger = logging.getLogger('')

	# clear existing handlers to avoid duplicated output
	logger.handlers.clear()

	# set logging level, DEBUG means everything, also INFO, WARNING, EXCEPTION, ERROR etc
	logger.setLevel(logging.DEBUG)

	# define the format of the message
	formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

	# write log to filehandler
	file_handler = logging.FileHandler(log_file_name)
	file_handler.setFormatter(formatter)

	# write to console
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)

	# add stream handler and file handler to logger
	logger.addHandler(stream_handler)
	logger.addHandler(file_handler)
	
	return logger


def set_start():
	"""
	Make sure the logger outputs to the console in a certain format
	Define the start time and get the process so we know how much memory we are using

	Returns
	----------
	tic : timestamp
		time the program starts
	process : object
		process id
	logger : logging object
		logger that outputs to console and save log file to disk
	"""

	# create logging to console
	logger = set_logger()

	# define start time
	tic = time.time()

	# define process ID
	process = psutil.Process(os.getpid())

	return tic, process, logger


def set_end(tic, process):
	"""
	Verbose function to display the elapsed time and used memory

	Parameters
	----------
	tic : timestamp
		time the program has started
	"""

	# print time elapsed
	logging.info('-- executed in {} seconds'.format(time.time()-tic))
	logging.info('-- used {} MB of memory'.format(process.memory_info().rss / 1024 / 1024))


def create_directory(name):
	"""
	Create directory if not exists

	Parameters
	----------
	name : string
		name of the folder to be created
	"""

	try:
		if not os.path.exists(name):
			os.makedirs(name)
			logging.info('Created directory: {}'.format(name))
	except Exception as e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def read_directory(directory):

	"""
	Read file names from directory recursively

	Parameters
	----------
	directory : string
		directory/folder name where to read the file names from

	Returns
	---------
	files : list of strings
		list of file names
	"""
	
	try:
		return glob2.glob(os.path.join( directory, '**' , '*.*'))
	except Exception as e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def calculate_vector_magnitude(data, minus_one = False, round_negative_to_zero = False, dtype = np.float32):
	"""
	Calculate vector magnitude of acceleration data
	the vector magnitude of acceleration is calculated as the Euclidian Norm

	sqrt(y^2 + x^2 + z^2)

	if minus_one is set to True then it it is the Euclidian Norm Minus One

	sqrt(y^2 + x^2 + z^2) - 1

	Parameters
	----------
	data : numpy array (acceleration values, axes)
		numpy array with acceleration data
	minus_one : Boolean (optional)
		If set to True, the calculate the vector magnitude minus one, also known as the ENMO (Euclidian Norm Minus One)
	round_negative_to_zero : Boolean (optional)
		If set to True, round negative values to zero
	dtype = mumpy data type (optional)
		set the data type of the return array. Standard float 16, but can be set to better precision
	
	Returns
	-------
	vector_magnitude : numpy array (acceleration values, 1)(np.float)
		numpy array with vector magnitude of the acceleration
	"""

	# change dtype of array to float32 (also to hold scaled data correctly). The original unscaled data is stored as int16, but when we want to calculate the vector we exceed the values that can be stored in 16 bit
	data = data.astype(dtype = np.float32)

	try:

		# calculate the vector magnitude on the whole array
		vector_magnitude = np.sqrt(np.sum(np.square(data), axis=1)).astype(dtype=dtype)

		# check if minus_one is set to True, if so, we need to calculate the ENMO
		if minus_one:
			vector_magnitude -= 1

		# if set to True, round negative values to zero
		if round_negative_to_zero:
			vector_magnitude = vector_magnitude.clip(min=0)

		# reshape the array into number of acceleration values, 1 column
		return vector_magnitude.reshape(data.shape[0], 1)
		

	except Exception as e:
		
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)


def save_csv(data, name, folder):
	"""
	Save list of list as CSV (comma separated values)

	Parameters
	----------
	data : list of list
		A list of lists that contain data to be stored into a CSV file format
	name : string
		The name of the file you want to give it
	folder: string
		The folder location
	"""
	
	try:

		# create folder name as directory if not exists
		create_directory(folder)

		# create the path name (allows for .csv and no .csv extension to be handled correctly)
		suffix = '.csv'
		if name[-4:] != suffix:
			name += suffix

		# create the file name
		path = os.path.join(folder, name)

		# save data to folder with name
		with open(path, "w") as f:
			writer = csv.writer(f, lineterminator='\n')
			writer.writerows(data)

	except Exception as e:
		logging.error('[{}] : {}'.format(sys._getframe().f_code.co_name,e))
		exit(1)
