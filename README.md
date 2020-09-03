# CNN non-wear time algorithm
A novel algorithm to detect non-wear time from raw acceleration data that can detect non-wear time episodes of any duration.

See paper:

A novel algorithm to detect non-wear time from raw accelerometer data using convolutional neural networks

Shaheen Syed, Bente Morseth, Laila A Hopstock, Alexander Horsch

doi: https://doi.org/10.1101/2020.07.08.20148015

### Installing Requirements

First, make sure the requirements are installed.

```bash
pip install -r requirements.txt
```


## Step 1) Read Actigraph .gt3x file to extract raw acceleration data
The script read_raw_gt3x.py contains code to extract raw acceleration data from .gt3x files. Each .gt3x file is basically a zip file containing a log.bin and a info.txt file. The log.bin is a binary file which contains the actual acceleration values. The info.txt file contains the meta-data in text form. When the script is executed, it will create a numpy file that contains the raw data and a time vector.

### Usage
```bash
python3 read_raw_gt3x.py -fd /path to folder with .gt3x files
```

The script accepts the following arguments

| Argument  short| Argument long  | Description  |
| :---:   | :-: | :-: |
| -fd | --folder | Folder location where one or several .gt3x files are stored. |
| -s | --save | Folder location where extracted raw data should be saved. If folder does not exist, it will be created. If not provided, the same folder as defined by -fd will be used. |
| -ds | --delete_source | Delete the original .gt3x source file after its content is unzipped. |
| -dz | --delete_zip | When the .gt3x files is unzipped, it creates a log.bin data. This file contains the raw acceleration data. After this data has been converted to a numpy array, it can be deleted by provided this argument.|
| -up | --use_parallel| When this argument is given, all .gt3x files will be processed in parallel.|

For example, process all .gt3x files in folder /users/username/gt3x, delete the original .gt3x file, delete the extracted zip file, and process all files in parallel:

```bash
python3 read_raw_gt3x.py -fd /users/username/gt3x -ds -dz -up
```

## Step 2) Infer non-wear time vectors from raw acceleration data using CNN method
The script infer_nw_time.py reads the raw data that was extracted from the .gt3x files (see step 1) and uses the CNN non-wear time algorithm to infer non-wear vectors and two files containing the start and stop indexes and timestamps of each non-wear episode.

Note that the CNN model was trained with an accelerometer placed on the hip. Furthermore, it works with triaxial data sampled at 100hz. If the data has a different sampling frequency, let's say 30Hz, then the acceleration data will be resampled to 100hz. Please also note that resampled acceleration values and the effect of the inferred non-wear vectors have not been tested.

```bash
python3 infer_nw_time.py -fd /path to folder with .gtx files
```

The script accepts the following arguments

| Argument  short| Argument long  | Description  |
| :---:   | :-: | :-: |
| -fd | --folder | Folder location where raw acceleration data in numpy format is saved in subfolders|

## Examples
Example Python code to get non-wear vectors from several published algorithms

```python
"""


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

```