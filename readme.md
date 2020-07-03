# CNN non-wear time algorithm
A novel algorithm to detect non-wear time from raw acceleration data that can detect non-wear time episodes of any duration.

See paper:
'A novel algorithm to detect non-wear time from raw accelerometer data using convolutional neural networks'

## Step 1) Read Actigraph .gt3x file to extract raw acceleration data
The script read_raw_gt3x.py contains code to extract raw acceleration data from .gt3x files. Each .gt3x file is basically a zip file containing a log.bin and a info.txt file. The log.bin is a binary file which contains the actual acceleration values. The info.txt file contains the meta-data in text form. When the script is executed, it will create a numpy file that contains the raw data and a time vector.

### Usage
```bash
python3 read_raw_gt3x.py --fd /path to folder with .gtx files
```

## Installation

```bash
pip3 install -r requirements.txt
```

