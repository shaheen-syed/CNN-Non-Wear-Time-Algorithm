# import packages
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser

# import functions
from functions.helper_functions import set_start, set_end, read_directory, create_directory, save_csv
from functions.raw_non_wear_functions import cnn_nw_algorithm
from functions.gt3x_functions import create_time_array, rescale_log_data, extract_info

def parse_arguments():
	"""
	Parse out the arguments from the command line

	Returns
	--------
	args : ArgumentParser()
		object with arguments now available as attributes
	"""

	# create command line parser
	parser = ArgumentParser()
	# folder where raw acceleration is stored. If this hasn't been created yet, then run read_raw_gt3x.py first
	parser.add_argument('-fd', '--folder', dest = 'folder', help = 'Folder location where raw acceleration data in numpy format is saved in subfolders.')

	# parse out the arguments and return
	return parser.parse_args()



"""
	SCRIPT STARTS HERE
"""
if __name__ == "__main__":

	# set the logger and start time
	tic, process, logging = set_start()
	
	# pars command line arguments
	args = parse_arguments()

	"""
		DEFINE CNN NON-WEAR TIME ALGORITHM HYPERPARAMETERS
	"""
	# standard deviation threshold in g
	std_threshold = 0.004
	# merge distance to group two nearby candidate nonwear episodes
	distance_in_min = 5
	# define window length to create input features for the CNN model
	episode_window_sec = 7
	# default classification when an episode does not have a starting or stop feature window (happens at t=0 or at the end of the data)
	edge_true_or_false = True
	# logical operator to see if both sides need to be classified as non-wear time (AND) or just a single side (OR)
	start_stop_label_decision = 'and'

	# load cnn model
	cnn_model_file = os.path.join('cnn_models', f'cnn_v2_{str(episode_window_sec)}.h5')

	"""
		PROCESS ALL FILES IN -FD
	"""
	# check if folder argument is provided
	if args.folder is not None:

		# read all numpy files in folder argument
		F = [f for f in read_directory(args.folder) if f[-4:] == '.npz']

		logging.info(f'Found {len(F)} raw acceleration files to process')

		# process each file
		for i, file in enumerate(F):

			logging.info(f'Processing file {file} {i}/{len(F)}')

			"""
				PREPARE DATA
			"""

			# read file as numpy array
			data = np.load(file)
			# read meta data from file
			meta_data = extract_info(os.path.join(os.path.dirname(file), 'info.txt'))
			# extract raw acceleration data from numpy.
			actigraph_acc = data['raw_data']
			# convert acceleration values to g values
			actigraph_acc = rescale_log_data(log_data = actigraph_acc, acceleration_scale = meta_data['Acceleration_Scale'])

			# extract time data 
			actigraph_time = data['time_data']
			# convert time data to correct time series array with correct miliseconds values
			actigraph_time = create_time_array(actigraph_time)
			
			"""
				INFER NON-WEAR TIME
			"""
			# call function to infer non-wear time
			nw_vector, nw_data = cnn_nw_algorithm(	raw_acc = actigraph_acc,
													hz = int(meta_data['Sample_Rate']),
													cnn_model_file = cnn_model_file, 
													std_threshold = std_threshold,
													distance_in_min = distance_in_min,
													episode_window_sec = episode_window_sec,
													edge_true_or_false = edge_true_or_false,
													start_stop_label_decision = start_stop_label_decision
													)
			
			"""
				POST PROCESS DATA
			"""

			# convert nw_indexes to timestamps
			nw_data_timestamps = []
			for row in nw_data:
				# get start timestamps
				start_timestamp = actigraph_time[row[0]]
				# get stop timestamps
				stop_timestamp = actigraph_time[row[1]]
				# verbose
				logging.info(f'Found non wear episode at Start : {start_timestamp}, Stop : {stop_timestamp}')
				# add to list
				nw_data_timestamps.append([start_timestamp, stop_timestamp])

			"""
				SAVE DATA
			"""

			# save non-wear vector to the same folder as where the numpy data was stored
			logging.info('Saving data to disk')

			# save non-wear vector as numpy array
			np.save(file = os.path.join(os.path.dirname(file), 'nw_vector'), arr = nw_vector)
			# save start and stop indexes
			save_csv(data = nw_data, name = 'non_wear_data_indexes', folder = os.path.dirname(file))
			# save human readable start and stop timestamps as CSV file
			save_csv(data = nw_data_timestamps, name = 'non_wear_data_timestamps', folder = os.path.dirname(file))
		
	else:
		logging.warning('Folder argument not provided. Please specify the -fd or --folder argument which specifies where raw acceleration data is stored in numpy format.')

	# verbose
	set_end(tic, process)