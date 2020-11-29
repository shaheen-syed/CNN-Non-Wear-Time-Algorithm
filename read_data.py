import os
import numpy as np
import tempfile as tmp

# import functions
from functions.helper_functions import set_start, set_end, read_directory, create_directory
from gt3x import unzip_gt3x_file, extract_info, extract_log, create_time_array, rescale_log_data


file = "data/AI12_NEO1F09120034_2017-09-25.gt3x"
save_folder = tmp.TemporaryDirectory()
save_folder = save_folder.name
delete_source_file = False
# unzip .gt3x file and get the file location of the binary log.bin (which contains the raw data) and the info.txt which contains the meta-data
log_bin, info_txt, _, _ = unzip_gt3x_file(f = file, save_location = save_folder, delete_source_file = delete_source_file)

# get meta data from info.txt file
meta_data = extract_info(info_txt)

# read raw data from binary data
log_data, time_data = extract_log(log_bin = log_bin, acceleration_scale = float(meta_data['Acceleration_Scale']), sample_rate = int(meta_data['Sample_Rate']), use_scaling = False)

actigraph_acc = rescale_log_data(log_data = log_data, acceleration_scale = meta_data['Acceleration_Scale'])

# convert time data to correct time series array with correct miliseconds values
actigraph_time = create_time_array(time_data)
