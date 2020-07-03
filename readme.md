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

The script also accepts the following arguments
| Argument  short| Argument long  | Description  |
| :---:   | :-: | :-: |
| -fd | --folder | Folder location where one or several .gt3x files are stored. |
| -s | -save | Folder location where extracted raw data should be saved. If folder does not exist, it will be created. If not provided, the same folder as defined by -fd will be used. |
| | --delete_source | Delete the original .gt3x source file after its content is unzipped. |
| | --delete_zip | When the .gt3x files is unzipped, it creates a log.bin data. This file contains the raw acceleration data. After this data has been converted to a numpy array, it can be deleted by provided this argument.|
| | --use_parallel| When this argument is given, all .gt3x files will be processed in parallel.|


## Installation

```bash
pip3 install -r requirements.txt
```

