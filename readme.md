# CNN non-wear time algorithm
A novel algorithm to detect non-wear time from raw acceleration data that can detect non-wear time episodes of any duration.

See paper:

A novel algorithm to detect non-wear time from raw accelerometer data using convolutional neural networks

Shaheen Syed, Bente Morseth, Laila A Hopstock, Alexander Horsch

doi: https://doi.org/10.1101/2020.07.08.20148015

### Installing Requirements

First, make sure the requirements are installed.

The `gt3x` package needs to be first installed via:

```bash
pip install git+https://github.com/muschellij2/gt3x.git#egg=gt3x
````

and then the rest of the packages can be installed via:

```bash
pip install -r requirements.txt
```


## Step 1) Read Actigraph .gt3x file to extract raw acceleration data
The script read_raw_gt3x.py contains code to extract raw acceleration data from .gt3x files. Each .gt3x file is basically a zip file containing a log.bin and a info.txt file. The log.bin is a binary file which contains the actual acceleration values. The info.txt file contains the meta-data in text form. When the script is executed, it will create a numpy file that contains the raw data and a time vector.

### Usage
```bash
python3 read_raw_gt3x.py -fd /path to folder with .gtx files
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

## Installation

```bash
pip3 install -r requirements.txt
```

