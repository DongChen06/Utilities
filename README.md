# Utilities
This repository is used to store utility codes which may be useful in the near future.

## Contents

* [Numpy](#numpy)
  * [Save and read numpy array](#save-and-read-numpy-array)
  * [Save and read pkl](#save-and-read-pkl)
* [Folder Operation](#folder-operation)
  * [Move up several folders](#move-up-several-folders)
* [Python](#python)
  * [Calculating run time](#calculating-run-time)
* [Tensorflow](#tensorflow)
  * [Tensorboard](#tensorboard)
  * [Tensorflow GPU Statement](-tensorflow-gPU-statement)

# Numpy
## Save and read numpy array


## Save and read pkl
```
data = (1, 2, 3, [[1,2,3],[4,5,6]], ['a','b','c'])
with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)
 ```
 read data
 ```
 with open('data.pickle', 'rb') as f:
     data = pickle.load(f)
 ```

# Folder Operation

## Move up several folders
```
from pathlib import Path

full_path = "path/to/directory"
str(Path(full_path).parents[0])  # "path/to"
str(Path(full_path).parents[1])  # "path"
str(Path(full_path).parents[2])  # "."
```

# Python

## Calculating run time
[Reference](https://stackoverflow.com/questions/5622976/how-do-you-calculate-program-run-time-in-python)

```
from datetime import datetime
start=datetime.now()

#Statements

print(datetime.now()-start)
```


# Tensorflow
## Tensorboard
- On linux: `tensorboard --logdir=logs`
 
- On windows: `tensorboard --logdir=log/ --host localhost --port 8088`

## Tensorflow GPU Statement
```
from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```
If we donot want to use GPU, we can set it as -1.

