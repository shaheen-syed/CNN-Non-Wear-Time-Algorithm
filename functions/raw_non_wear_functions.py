"""
	IMPORT PACKAGES
"""
import numpy as np
import pandas as pd
import logging
import os
from tensorflow.keras import models

from functions.helper_functions import calculate_vector_magnitude
from functions.signal_processing_functions import resample_acceleration

def find_candidate_non_wear_segments_from_raw(acc_data, std_threshold, hz, min_segment_length = 1, sliding_window = 1, use_vmu = False):
	"""
	Find segements within the raw acceleration data that can potentially be non-wear time (finding the candidates)

	Parameters
	---------
	acc_data : np.array(samples, axes)
		numpy array with acceleration data (typically YXZ)
	std_threshold : int or float
		the standard deviation threshold in g
	hz : int
		sample frequency of the acceleration data (could be 32hz or 100hz for example)
	min_segment_length : int (optional)
		minimum length of the segment to be candidate for non-wear time (default 1 minutes, so any shorter segments will not be considered non-wear time)
	hz : int (optional)
		sample frequency of the data (necessary so we know how many data samples we have in a second window)
	sliding_window : int (optional)
		sliding window in minutes that will go over the acceleration data to find candidate non-wear segments
	"""

	# adjust the sliding window to match the samples per second (this is encoded in the samplign frequency)
	sliding_window *= hz * 60
	# adjust the minimum segment lenght to reflect minutes
	min_segment_length*= hz * 60

	# define new non wear time vector that we initiale to all 1s, so we only have the change when we have non wear time as it is encoded as 0
	non_wear_vector = np.ones((len(acc_data), 1), dtype = np.uint8)
	non_wear_vector_final = np.ones((len(acc_data), 1), dtype = np.uint8)

	# loop over slices of the data
	for i in range(0,len(acc_data), sliding_window):

		# slice the data
		data = acc_data[i:i + sliding_window]

		# calculate VMU if set to true
		if use_vmu:
			# calculate the VMU of XYZ
			data = calculate_vector_magnitude(data)
	
		# calculate the standard deviation of each column (YXZ)
		std = np.std(data, axis=0)

		# check if all of the standard deviations are below the standard deviation threshold
		if np.all(std <= std_threshold):

			# add the non-wear time encoding to the non-wear-vector for the correct time slices
			non_wear_vector[i:i+sliding_window] = 0

	# find all indexes of the numpy array that have been labeled non-wear time
	non_wear_indexes = np.where(non_wear_vector == 0)[0]

	# find the min and max of those ranges, and increase incrementally to find the edges of the non-wear time
	for row in find_consecutive_index_ranges(non_wear_indexes):

		# check if not empty
		if row.size != 0:

			# define the start and end of the index range
			start_slice, end_slice = np.min(row), np.max(row)

			# backwards search to find the edge of non-wear time vector
			start_slice = backward_search_non_wear_time(data = acc_data, start_slice = start_slice, end_slice = end_slice, std_max = std_threshold, hz = hz) 
			# forward search to find the edge of non-wear time vector
			end_slice = forward_search_non_wear_time(data = acc_data, start_slice = start_slice, end_slice = end_slice, std_max = std_threshold, hz = hz)

			# calculate the length of the slice (or segment)
			length_slice = end_slice - start_slice

			# minimum length of the non-wear time
			if length_slice >= min_segment_length:

				# update numpy array by setting the start and end of the slice to zero (this is a non-wear candidate)
				non_wear_vector_final[start_slice:end_slice] = 0

	# return non wear vector with 0= non-wear and 1 = wear
	return non_wear_vector_final


def find_consecutive_index_ranges(vector, increment = 1):
	"""
	Find ranges of consequetive indexes in numpy array

	Parameters
	---------
	data: numpy vector
		numpy vector of integer values
	increment: int (optional)
		difference between two values (typically 1)

	Returns
	-------
	indexes : list
		list of ranges, for instance [1,2,3,4],[8,9,10], [44]
	"""

	return np.split(vector, np.where(np.diff(vector) != increment)[0]+1)


def forward_search_non_wear_time(data, start_slice, end_slice, std_max, hz, time_step = 60):
	"""
	Increase the end_slice to obtain more non_wear_time (used when non-wear range has been found but due to window size, the actual non-wear time can be slightly larger)

	Parameters
	----------
	data: numpy array of time x 3 axis 
		raw log data
	start_slice: int
		start of known non-wear time range
	end_slice: int
		end of known non-wear time range
	std_max : int or float
		the standard deviation threshold in g
	time_step : int (optional)
		value to add (or subtract in the backwards search) to find more non-wear time
	"""

	# adjust time step on number of samples per time step window
	time_step *= hz

	# define the end of the range
	end_of_data = len(data)

	# Do-while loop
	while True:

		# define temporary end_slice variable with increase by step
		temp_end_slice = end_slice + time_step

		# check condition range still contains non-wear time
		if temp_end_slice <= end_of_data and np.all(np.std(data[start_slice:temp_end_slice], axis=0) <= std_max):
			
			# update the end_slice with the temp end slice value
			end_slice = temp_end_slice

		else:
			# here we have found that the additional time we added is not non-wear time anymore, stop and break from the loop by returning the updated slice
			return end_slice


def backward_search_non_wear_time(data, start_slice, end_slice, std_max, hz, time_step = 60):
	"""
	Decrease the start_slice to obtain more non_wear_time (used when non-wear range has been found but the actual non-wear time can be slightly larger, so here we try to find the boundaries)

	Parameters
	----------
	data: numpy array of time x 3 axis 
		raw log data
	start_slice: int
		start of known non-wear time range
	end_slice: int
		end of known non-wear time range
	std_max : int or float
		the standard deviation threshold in g
	time_step : int (optional)
		value to add (or subtract in the backwards search) to find more non-wear time
	"""

	# adjust time step on number of samples per time step window
	time_step *= hz

	# Do-while loop
	while True:

		# define temporary end_slice variable with increase by step
		temp_start_slice = start_slice - time_step

		# logging.debug('Decreasing temp_start_slice to: {}'.format(temp_start_slice))

		# check condition range still contains non-wear time
		if temp_start_slice >= 0 and np.all(np.std(data[temp_start_slice:end_slice], axis=0) <= std_max):
			
			# update the start slice with the new temp value
			start_slice = temp_start_slice

		else:
			# here we have found that the additional time we added is not non-wear time anymore, stop and break from the loop by returning the updated slice
			return start_slice


def group_episodes(episodes, distance_in_min = 3, correction = 3, hz = 100, training = False):
	"""
	Group episodes that are very close together

	Parameters
	-----------
	episodes : pd.DataFrame()
		dataframe with episodes that need to be grouped
	distance_in_min = int
		maximum distance two episodes can be apart and need to be grouped together
	correction = int
		due to changing from 100hz to 32hz we need to allow for a small correction to capture full minutes
	hz = int
		sample frequency of the data (necessary when working with indexes)

	Returns
	--------
	grouped_episodes : pd.DataFrame()
		dataframe with grouped episodes
	"""

	# check if there is only 1 episode in the episodes dataframe, if so, we need not to do anything since we cannot merge episodes if we only have 1
	if episodes.empty or len(episodes) == 1:
		# transpose back and return
		return episodes.T

	# create a new dataframe that will contain the grouped rows
	grouped_episodes = pd.DataFrame()

	# get all current values from the first row
	current_start = episodes.iloc[0]['start']
	current_start_index = episodes.iloc[0]['start_index']
	current_stop = episodes.iloc[0]['stop']
	current_stop_index = episodes.iloc[0]['stop_index']
	current_label = None if not training else episodes.iloc[0]['label']
	current_counter = episodes.iloc[0]['counter']


	# loop over each next row (note that we skip the first row)
	for _, row in episodes.iloc[1:].iterrows():

		# define all next values
		next_start = row.loc['start']
		next_start_index = row.loc['start_index']
		next_stop = row.loc['stop']
		next_stop_index = row.loc['stop_index']
		next_label = None if not training else row.loc['label']
		next_counter = row.loc['counter']

		# check if there are 'distance_in_min' minutes apart from current and next ( + correction for some adjustment)
		if next_start_index - current_stop_index <= hz * 60 * distance_in_min + correction:
			
			# here the two episodes are close to eachother, we update the values and continue the next row to see if we can group more. If it's the last row, we need to add it to the dataframe
			current_stop_index = next_stop_index
			current_stop = next_stop

			# check if row is the last row
			if next_counter == episodes.iloc[-1]['counter']:

				# create the counter label
				counter_label = f'{current_counter}-{next_counter}'

				# save to new dataframe
				grouped_episodes[counter_label] = pd.Series({ 	'counter' : counter_label,
																'start_index' : current_start_index,
																'start' : current_start,
																'stop_index' : current_stop_index,
																'stop' : current_stop,
																'label' : None if not training else current_label })
		else:			
			
			# create the counter label
			counter_label = current_counter if (next_counter - current_counter == 1) else f'{current_counter}-{next_counter - 1}'

			# save to new dataframe
			grouped_episodes[counter_label] = pd.Series({ 	'counter' : counter_label,
															'start_index' : current_start_index,
															'start' : current_start,
															'stop_index' : current_stop_index,
															'stop' : current_stop,
															'label' : None if not training else current_label})

			# update tracker variables
			current_start = next_start
			current_start_index = next_start_index
			current_stop = next_stop
			current_stop_index = next_stop_index
			current_label = next_label
			current_counter = next_counter

			# check if last row then also include by itself
			if next_counter == episodes.iloc[-1]['counter']:

				# save to new dataframe
				grouped_episodes[next_counter] = pd.Series({ 	'counter' : next_counter,
																'start_index' : current_start_index,
																'start' : current_start,
																'stop_index' : current_stop_index,
																'stop' : current_stop,
																'label' : None if not training else current_label })

	return grouped_episodes



def cnn_nw_algorithm(raw_acc, hz, cnn_model_file, std_threshold = 0.004, distance_in_min = 5, episode_window_sec = 7, edge_true_or_false = True,\
								start_stop_label_decision = 'and', nwt_encoding = 1, wt_encoding = 0,
								min_segment_length = 1, sliding_window = 1, verbose = False):
	"""
	Infer non-wear time from raw 100Hz triaxial data. Data at different sample frequencies will be resampled to 100hz.

	Paper:
	A novel algorithm to detect non-wear time from raw accelerometer data using convolutional neural networks

	The steps are described in the paper but are summarized here too:


	Detect candidate non-wear episodes:
		Perform a forward pass through the raw acceleration signal and calculate the SD for each 1-minute interval and for each individual axis. 
		If the standard deviation is <= 0.004 g for all axes, record this 1-minute interval as a candidate non-wear interval. After all 1-minute 
		intervals have been processed, merge consecutive 1-minute intervals into candidate non-wear episodes and record their start and stop timestamps. 

	Merge bordering candidate non-wear episodes:
		Merge candidate non-wear episodes that are no more than 5 minutes apart and record their new start and stop timestamps. This step is required 
		to capture artificial movement that would typically break up two or more candidate non-wear episodes in close proximity.

	Detect the edges of candidate non-wear episodes:
		Perform a backward pass with a 1-second step size through the acceleration data from the start timestamp of a candidate non-wear episode and 
		calculate the SD for each individual axis. The same is applied for the stop timestamps with a forward pass and a step size of 1 second. 
		If the standard deviation of all axes is  <= 0.004 g, include the 1-second interval into the candidate non-wear episode and record the new 
		start or stop timestamp. Repeat until the standard deviation of the 1-second interval does not satisfy <= 0.004 g. As a result, the resolution 
		of the edges is now recorded on a 1-second resolution.

	Classifying the start and stop windows: 
		For each candidate non-wear episode, extract the start and stop segment with a window length of 3 seconds to create input features 
		for the CNN classification model. For example, if a candidate non-wear episode has a start timestamp of tstart a feature matrix is 
		created as (tstart – w  , tstart) x 3 axes with w = 3 seconds, resulting in an input feature with dimensions (300 x 3) for 100Hz data. 
		If both (i.e., logical ‘AND’) start and stop features are classified (through the CNN model) as non-wear time, the candidate non-wear 
		episode can be considered true non-wear time. If tstart is at t = 0, or tend is at the end of the acceleration data—meaning that 
		those candidate non-wear episodes do not have a preceding or following window to extract features from—classify the start or stop 
		as non-wear time by default.


	Parameters
	--------------
	raw_acc : np.array(n_samples, 3 axes)
		numpy array that contains raw triaxial data at 100hz. Size of the array should be (n_samples, 3)
	hz : int
		sample frequency of the data. The CNN model was trained for 100Hz of data. If the data is at a different sampling frequency it will be resampled to 100Hz
	cnn_model_file : os.path
		file location of the trained CNN model
	std_threshold : float (optional)
		standard deviation threshold to find candidate non-wear episodes. Default 0.004 g  
	distance_in_min : int (optional)
		causes two nearby candidate non-wear episodes not more than 'distance_in_min' apart to be grouped/merged together. This results in capturing artificial movement
		that would otherwise break up a longer candidate non-wear time. Defaults to 5 minutes.
	episode_window_sec : int (optional)
		length of the window to extract features from the start or the end of a candidate non-wear episode. If a non-wear episodes starts at time t, then a feature
		will be extracted from the raw data t-'episode_window_sec' to t. The same happens at the end of a candidate non-wear episode. So t-end untill t-end + 'episode_window_sec'
		Default to 7 seconds. Also note that a different value will need a different trained CNN model. 
	edge_true_or_false : Bool (optional)
		default classification if a candidate non-wear episode starts at the start of the acceleration data, so at t=0, or ends at the end of the acceleration data.
		In such cases, we can't obtain the feature at t-'episode_window_sec' since there is no data before t=0. In these cases, the start or stop of the candidate non-wear
		episode will be defaulted to True (non-wear time) or False (wear-time). Default value is True
	start_stop_label_decision : string ('or','and') (optional)
		Logical operator OR or AND to determine if a candidate non-wear episode should be considered non-wear time if only one side, either the start or the stop parts, is 
		inferred as non-wear time, or if both sides need to be inferred as non-wear time for the candidate non-wear time to be considered true non-wear time. Default to AND, meaning
		that both the start and the stop parts of the candidate non-wear time need to be inferred as non-wear time to allow the candidate non-wear time to be true non-wear time.
		In all other cases, the candidate non-wear time is then wear-time. 
	nwt_encoding : int (optional)
		encoding of non-wear time for the returning vector. Defaults to 1 
	wt_encoding : int (optional)
		encoding of wear time for the returning vector. Defaults to 0
	min_segment_length : int (optional)
		minimum length of the segment to be candidate for non-wear time
	sliding_window : int (optional)
		sliding window in minutes that will go over the acceleration data to find candidate non-wear segments
	verbose : Bool (optional)
		set to True if debug messages should be printed to the console and log file. Default False.

	Returns
	---------
	nw_vector : np.array(n_samples, 1)
		non-wear vector which has the same number of samples as 'raw_acc'. Every element has either a non-wear-time encoding or a wear-time encoding.
	nw_start_stop_indexes : list
		list of start and stop indexes that are considered non-wear time.

	Important
	---------
	-	If the data is not 100hz, then it will be resampled to 100hz. However, how the inference of non-wear time is affected by this has not been investigated.
	-	CNN models were trained with a hip worn accelerometer.
	"""

	# check if data is triaxial
	if raw_acc.shape[1] != 3:
		logging.error(f'Acceleration data should be triaxial/3 axes. Number of axes found is {raw_acc.shape[1]}')
		exit(1)

	# check if wear time encoding and non-wear time encoding are different
	if wt_encoding == nwt_encoding:
		logging.error('Wear time encoding and non-wear time encoding are the same, whereas they should be different.')
		exit(1)

	# check if data needs to be resampled to 100hz
	if hz != 100:
		logging.info(f'Sampling frequency of the data is {hz}Hz, should be 100Hz, starting resampling....')
		# call resampling function
		raw_acc = resample_acceleration(data = raw_acc, from_hz = hz, to_hz = 100, verbose = verbose)
		logging.info('Data resampled to 100hz')
		# set sampling frequency to 100hz
		hz = 100

	
	# create new non-wear vector that is prepopulated with wear-time encoding. This way we only have to record the non-wear time
	nw_vector = np.full(shape = [raw_acc.shape[0], 1], fill_value = wt_encoding, dtype = 'uint8')
	# empty list to keep track of non-wear time start and stop indexes.
	nw_start_stop_indexes = []

	"""
		FIND CANDIDATE NON-WEAR SEGMENTS ACTIGRAPH ACCELERATION DATA
	"""

	# get candidate non-wear episodes (note that these are on a minute resolution). Also note that it returns wear time as 1 and non-wear time as 0
	nw_episodes = find_candidate_non_wear_segments_from_raw(acc_data = raw_acc, std_threshold = std_threshold, 
															min_segment_length = min_segment_length,
															sliding_window = sliding_window, hz = hz)

	"""
		GET START AND END TIME OF NON WEAR SEGMENTS
	"""

	# find all indexes of the numpy array that have been labeled non-wear time
	nw_indexes = np.where(nw_episodes == 0)[0]
	# find consecutive ranges
	non_wear_segments = find_consecutive_index_ranges(nw_indexes)
	# empty dictionary where we can store the start and stop times
	dic_segments = {}

	# check if segments are found
	if len(non_wear_segments[0]) > 0:
		
		# find start and stop times (the indexes of the edges and find corresponding time)
		for i, row in enumerate(non_wear_segments):

			# find start and stop
			start, stop = np.min(row), np.max(row)

			# add the start and stop times to the dictionary
			# note that start and stop timestamps are not given. 
			dic_segments[i] = {'counter' : i, 'start' : start, 'start_index': start, 'stop' : stop, 'stop_index' : stop}
	
	# create dataframe from segments
	episodes = pd.DataFrame.from_dict(dic_segments)
	
	"""
		MERGE EPISODES THAT ARE CLOSE TO EACH OTHER
	"""				
	grouped_episodes = group_episodes(episodes = episodes.T, distance_in_min = distance_in_min, correction = 3, hz = hz, training = False).T
	
	"""
		LOAD CNN MODEL
	"""	

	# load CNN model
	cnn_model = models.load_model(cnn_model_file)

	"""
		FOR EACH EPISODE, EXTEND THE EDGES, CREATE FEATURES, AND INFER LABEL
	"""
	for _, row in grouped_episodes.iterrows():

		start_index = int(row.loc['start_index'])
		stop_index = int(row.loc['stop_index'])

		if verbose:
			logging.debug(f'Processing episode start_index : {start_index}, stop_index : {stop_index}')
	
		# forward search to extend stop index
		stop_index = _forward_search_episode(raw_acc, stop_index, hz = hz, max_search_min = 5, std_threshold = std_threshold, verbose = verbose)
		# backwar search to extend start index
		start_index = _backward_search_episode(raw_acc, start_index, hz = hz, max_search_min = 5, std_threshold = std_threshold, verbose = verbose)

		# get start episode
		start_episode = raw_acc[start_index - (episode_window_sec * hz) : start_index]
		# get stop episode
		stop_episode = raw_acc[stop_index : stop_index + (episode_window_sec * hz)]

		# default label for start and stop combined. The first False will turn into True if the start of the episode is inferred as non-wear time. The same happens to the
		# second False when the end is inferred as non-weaer time
		start_stop_label = [False, False]

		"""
			START EPISODE
		""" 
		if start_episode.shape[0] == episode_window_sec * hz:

			# reshape into num feature x time x axes
			start_episode = start_episode.reshape(1, start_episode.shape[0], start_episode.shape[1]) 
			
			# get binary class from model
			start_label = cnn_model.predict_classes(start_episode).squeeze()

			# if the start label is 1, this means that it is wear time, and we set the first start_stop_label to 1
			if start_label == 1:
				start_stop_label[0] = True	
		
		else:
			# there is an episode right at the start of the data, since we cannot obtain a full epsisode_window_sec array
			# here we say that True for nw-time and False for wear time
			start_stop_label[0] = edge_true_or_false
		

		"""
			STOP EPISODE
		""" 
		if stop_episode.shape[0] == episode_window_sec * hz:
			
			# reshape into num feature x time x axes
			stop_episode = stop_episode.reshape(1, stop_episode.shape[0], stop_episode.shape[1]) 
			
			# get binary class from model
			stop_label = cnn_model.predict_classes(stop_episode).squeeze()

			# if the start label is 1, this means that it is wear time, and we set the first start_stop_label to 1
			if stop_label == 1:
				start_stop_label[1] = True	
		else:
			# there is an episode right at the END of the data, since we cannot obtain a full epsisode_window_sec array
			# here we say that True for nw-time and False for wear time
			start_stop_label[1] = edge_true_or_false
		
		# check the start_stop_label.
		if start_stop_label_decision == 'or':
			# use logical OR to determine if episode is true non-wear time
			if any(start_stop_label):
				# true non-wear time, record start and stop in nw-vector
				nw_vector[start_index:stop_index] = nwt_encoding
				# add start and stop to nw_data (this is the human readable start and stop)
				nw_start_stop_indexes.append([start_index, stop_index])
				# verbose
				if verbose:
					logging.info(f'Found non-wear time: start_index : {start_index}, Stop_index: {stop_index}')

		elif start_stop_label_decision == 'and':

			# use logical and to determine if episode is true non-wear time
			if all(start_stop_label):
				# true non-wear time, record start and stop in nw-vector
				nw_vector[start_index:stop_index] = nwt_encoding
				# add start and stop to nw_data
				nw_start_stop_indexes.append([start_index, stop_index])
				# verbose
				if verbose:
					logging.info(f'Found non-wear time: start_index : {start_index}, Stop_index: {stop_index}')

		else:
			logging.error(f'Start/Stop decision unknown, can only use or/and, given: {start_stop_label_decision}')
			exit(1)

	return nw_vector, nw_start_stop_indexes


"""
	INTERNAL HELPER FUNCTIONS
"""

def _forward_search_episode(acc_data, index, hz, max_search_min, std_threshold, verbose = False):
	"""
	When we have an episode, this was created on a minute resolution, here we do a forward search to find the edges of the episode with a second resolution
	"""

	# calculate max slice index
	max_slice_index = acc_data.shape[0]

	for i in range(hz * 60 * max_search_min):

		# create new slices
		new_start_slice = index
		new_stop_slice = index + hz

		if verbose:
			logging.info(f'i : {i}, new_start_slice : {new_start_slice}, new_stop_slice : {new_stop_slice}')

		# check if the new stop slice exceeds the max_slice_index
		if new_stop_slice > max_slice_index:
			if verbose:
				logging.info(f'Max slice index reached : {max_slice_index}')
			break
			
		# slice out new activity data
		slice_data = acc_data[new_start_slice:new_stop_slice]

		# calculate the standard deviation of each column (YXZ)
		std = np.std(slice_data, axis=0)
		
		# check if all of the standard deviations are below the standard deviation threshold
		if np.all(std <= std_threshold):
			
			# update index
			index = new_stop_slice
		else:
			break

	if verbose:
		logging.info(f'New index : {index}, number of loops : {i}')
	return index

def _backward_search_episode(acc_data, index, hz, max_search_min, std_threshold, verbose = False):
	"""
	When we have an episode, this was created on a minute resolution, here we do a backward search to find the edges of the episode with a second resolution
	"""

	# calculate min slice index
	min_slice_index = 0

	for i in range(hz * 60 * max_search_min):

		# create new slices
		new_start_slice = index - hz
		new_stop_slice = index

		if verbose:
			logging.info(f'i : {i}, new_start_slice : {new_start_slice}, new_stop_slice : {new_stop_slice}')

		# check if the new start slice exceeds the max_slice_index
		if new_start_slice < min_slice_index:
			if verbose:
				logging.debug(f'Minimum slice index reached : {min_slice_index}')
			break
			
		# slice out new activity data
		slice_data = acc_data[new_start_slice:new_stop_slice]

		# calculate the standard deviation of each column (YXZ)
		std = np.std(slice_data, axis=0)
		
		# check if all of the standard deviations are below the standard deviation threshold
		if np.all(std <= std_threshold):
			
			# update index
			index = new_start_slice
		else:
			break

	if verbose:
		logging.info(f'New index : {index}, number of loops : {i}')
	return index


