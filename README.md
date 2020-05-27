# Utilities
This repository is used to store utility codes which may be useful in the near future.

## Contents

* [Numpy](#numpy)
  * [Save and read numpy array](#save-and-read-numpy-array)
  * [Save and read pkl](#save-and-read-pkl)
* [Ubuntu](#ubuntu)
  * [Package Installation](#package-installation)
  * [Folder Operation](#folder-operation)
    * [Move up several folders](#move-up-several-folders)
* [Python](#python)
  * [Calculating run time](#calculating-run-time)
* [Deep Learning Related](#deep-learning-related)
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

# Ubuntu
## Package Installation
- to install the packages in the requirement.txt file: `pip install -r requirements.txt`.

- Install box2D
```
conda install swig
pip install pocketsphinx
pip install gym[box2d]
```

## Folder Operation

### Move up several folders
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


# Deep Learning Related

## Anaconda
- create a new environment In Anaconda: `conda create -n py3.5 python=3.5`

## Tensorflow
### Tensorboard
- On linux: `tensorboard --logdir=logs`
 
- On windows: `tensorboard --logdir=log/ --host localhost --port 8088`

### Tensorflow GPU Statement
```
from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```
If we donot want to use GPU, we can set it as -1.

