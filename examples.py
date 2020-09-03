"""
Example Python code to get non-wear vectors from several published algorithms

If you have a raw .gt3x file, please first use the above code to extract the raw data
"""

import os
import numpy as np
from functions.raw_non_wear_functions import cnn_nw_algorithm, hees_2013_calculate_non_wear_time, raw_baseline_calculate_non_wear_time

# read raw acceleration from numpy array. Since the data was not scaled, we are dividing it by the acceleration scale to obtain acceleration values in gravity units
raw_acc = np.load(file = os.path.join(os.sep, 'Users', 'shaheen.syed', 'PA', 'acceleration.npz'))['raw_data'] / 256.
# how to encode wear time
wt_encoding = 0
# how to encode non-wear time
nwt_encoding = 1
# sample frequency of the data
hz = 100


"""
Get non-wear vector from the CNN non-wear algorithm

Paper: A novel algorithm to detect non-wear time from raw accelerometer data using convolutional neural networks
DOI : https://doi.org/10.1101/2020.07.08.20148015
Authors : Shaheen Syed, Bente Morseth, Laila A Hopstock, Alexander Horsch
"""

# path for trained CNN model (best performing cnn model with v2 architecture and 7 seconds window)
cnn_model_file = os.path.join('cnn_models', f'cnn_v2_7.h5')
# obtain cnn non-wear vector
cnn_nw, _ = cnn_nw_algorithm(raw_acc = raw_acc, hz = hz, cnn_model_file = cnn_model_file, nwt_encoding = nwt_encoding, wt_encoding = wt_encoding)


"""
Get non-wear vector from v. Hees 2013 non-wear algorithm. Note that the method is identical to the paper published in 2011 but the minimum non-wear time window was increased from 30 to 60 minutes.
If you want to apply the 2011 method, simply change the 'min_non_wear_time_window' to 30

Papers:
van Hees, V. T. et al. Estimation of Daily Energy Expenditure in Pregnant and Non-Pregnant Women Using a Wrist-Worn Tri-Axial Accelerometer. PLoS ONE 6, e22922, DOI: 10.1371/journal.pone.0022922 (2011).
van Hees, V. T. et al. Separating Movement and Gravity Components in an Acceleration Signal and Implications for the Assessment of Human Daily Physical Activity. PLoS ONE 8, e61691, DOI: 10.1371/journal.pone.0061691 (2013).
"""

# v. Hees 2011 algorithm
hees_2011_nw = hees_2013_calculate_non_wear_time(data = raw_acc, hz = hz, min_non_wear_time_window = 30, nwt_encoding = nwt_encoding, wt_encoding = wt_encoding)
# v. Hees 2013 algorithm
hees_2013_nw = hees_2013_calculate_non_wear_time(data = raw_acc, hz = hz, min_non_wear_time_window = 60, nwt_encoding = nwt_encoding, wt_encoding = wt_encoding)


"""
Hees non-wear method with optimized hyperparameters

Paper: Evaluating the performance of raw and epoch non-wear algorithms using multiple accelerometers and electrocardiogram recordings
DOI: https://doi.org/10.1038/s41598-020-62821-2
Authors: Shaheen Syed, Bente Morseth, Laila A Hopstock, Alexander Horsch
"""
# hees non-wear vector with optimized hyperparameters
hees_optimized_nw = hees_2013_calculate_non_wear_time(data = raw_acc, hz = hz, min_non_wear_time_window = 135, std_mg_threshold = 7.0, std_min_num_axes = 1, value_range_mg_threshold = 1.0, value_range_min_num_axes = 1, nwt_encoding = nwt_encoding, wt_encoding = wt_encoding)


"""
Best performing baseline non-wear algorithms

Paper: A novel algorithm to detect non-wear time from raw accelerometer data using convolutional neural networks
DOI : https://doi.org/10.1101/2020.07.08.20148015
Authors : Shaheen Syed, Bente Morseth, Laila A Hopstock, Alexander Horsch
"""

# XYZ baseline method
xyz_nw = raw_baseline_calculate_non_wear_time(raw_acc = raw_acc, std_threshold = 0.004, min_interval = 90, hz = hz, use_vmu = False, nwt_encoding = nwt_encoding, wt_encoding = wt_encoding)
# VMY baseline method
vmu_nw = raw_baseline_calculate_non_wear_time(raw_acc = raw_acc, std_threshold = 0.004, min_interval = 105, hz = hz, use_vmu = True, nwt_encoding = nwt_encoding, wt_encoding = wt_encoding)
