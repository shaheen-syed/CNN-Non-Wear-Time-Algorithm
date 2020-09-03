# import packages
import os
import numpy as np
from argparse import ArgumentParser

# parallel processing
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

# import functions
from functions.helper_functions import set_start, set_end, read_directory, create_directory
from gt3x import read_gt3x

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
	# pars out the folder where multiple gt3x files are stored
	parser.add_argument('-fd', '--folder', dest = 'folder', help = 'Folder location where one or several .gt3x files are stored')
	# define save folder where data should be stored. If not given, then save folder will be the working directory
	parser.add_argument('-s', '--save', dest = 'save_folder', help = 'Folder location where extracted raw data should be saved. If folder does not exist, it will be created. If not provided, the same folder as defined by -fd will be used.')
	# delete original .gt3x file (source file) after it has been unzipped
	parser.add_argument('-ds', '--delete_source', dest = 'delete_source_file', action = 'store_true', help = 'Delete the original .gt3x source file after its content is unzipped.')
	# delete .zip files after extracting data from zip file
	parser.add_argument('-dz', '--delete_zip', dest = 'delete_zip_file', action='store_true', help = 'When the .gt3x files is unzipped, it creates a log.bin data. This file contains the raw acceleration data. After this data has been converted to a numpy array, it can be deleted by provided this argument.')
	# use parallel processing
	parser.add_argument('-up', '--use_parallel', dest = 'use_parallel', action = 'store_true', help = 'When this argument is given, all .gt3x files will be processed in parallel.')

	# parse out the arguments and return
	return parser.parse_args()

def process_gt3x_file(idx, total, file, save_folder, delete_source_file, delete_zip_file):
	"""
	Processing of a single .gt3x file with the following steps
		1: unzip .gt3x to get the log.bin and info.txt file and save to disk
		2: read info.txt file and parse the content to a dictionary
		3: extract raw acceleration data from the log.bin binary file
		4: save all data to disk


	Parameters
	-----------
	file : os.path
		file location if a single .gt3x file. This is the raw format of the ActiGraph
	save_folder : os.path
		folder location where extracted raw data should be saved. If folder does not exist, it will be created. If not provided, the same folder as defined by -fd will be used.
	delete_source_file : Bool
		delete the original .gt3x source file after its content is unzipped. If not provided, then default to False
	delete_zip_file : Bool
		when the .gt3x file is unzipped, it creates a log.bin data. This file contains the raw acceleration data. After this data has been converted to a numpy array, it can be deleted when set to True. Default False
	
	"""

	logging.info(f'Processing file {file} {idx + 1}/{total}')

	# extract name for subfolder based on file name without extension
	subfolder = os.path.splitext(file)[0].split(os.sep)[-1]
	
	# if save folder is not sent, then use the same folder as where the .gt3x file is located
	if save_folder is None:
		save_folder = os.path.splitext(file)[0]
	else:
		save_folder = os.path.join(save_folder, subfolder)

	
	# unzip .gt3x file and get the file location of the binary log.bin (which contains the raw data) and the info.txt which contains the meta-data
	# log_bin, info_txt = unzip_gt3x_file(f = file, save_location = save_folder, delete_source_file = delete_source_file)

	# # get meta data from info.txt file
	# meta_data = extract_info(info_txt)

	# # read raw data from binary data
	# log_data, time_data = extract_log(log_bin = log_bin, acceleration_scale = float(meta_data['Acceleration_Scale']), sample_rate = int(meta_data['Sample_Rate']), use_scaling = False)

	# if 'delete_zip_file' is set to True, then remove the unpacked log.bin data
	# if delete_zip_file:
	# 	os.remove(log_bin)

	"""
		The following code will use the package gt3x which contains updated code for older .gt3x formats. In case you want to use the functions within this repo, please use the above lines
		which are now uncommented.
	"""

	log_data, time_data, meta_data = read_gt3x(f = file, save_location = save_folder, create_time = False, rescale_data = False, verbose = False)

	# save log_data and time_data as numpy array
	np.savez(file = os.path.join(save_folder, subfolder), raw_data = log_data, time_data = time_data, meta_data = meta_data)


"""
	SCRIPT STARTS HERE
"""
if __name__ == "__main__":

	# set the logger and start time
	tic, process, logging = set_start()
	
	# pars command line arguments
	args = parse_arguments()

	# if --use_parallel is provided, set num_jobs to the number of cores
	if args.use_parallel:
		# set number of jobs to number of cpu cores
		num_jobs = cpu_count()
		logging.info(f'Parallel processing enabled. Using {num_jobs} cores')
	else:
		num_jobs = 1

	# check if folder is sent as argument, if so, then process each file within that folder 
	if args.folder is not None:
		
		# read all gt3x files within folder
		F = [x for x in read_directory(args.folder) if x[-4:] == 'gt3x']
		logging.info(f'Found a total of {len(F)} .gt3x files to process')

		# use parallel processing to speed up processing time
		executor = Parallel(n_jobs = num_jobs, backend = 'multiprocessing')
		# create tasks so we can execute them in parallel
		tasks = (delayed(process_gt3x_file)(idx = i, total = len(F), file = f, save_folder = args.save_folder, delete_source_file = args.delete_source_file, delete_zip_file = args.delete_zip_file) for i, f in enumerate(F))
		# execute task
		executor(tasks)

	else:
		logging.warning('No folder argument given. Please specify a folder location where the .gt3x files are located. This can be done with the -fd or --folder argument')
	
	# verbose duration
	set_end(tic, process)