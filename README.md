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
    * [Remove files in a directory with a specific extension](#remove-files-in-a-directory-with-a-specific-extension)
    * [Move or Copy files from one directory to another with specific extensions](#move-or-copy-files-from-one-directory-to-another-with-specific-extensions)
  * [Basic Operation](#basic-operation)
    * [Screenshots](#screenshots)
    * [Create new sudo user](#create-new-sudo-user)
* [Python](#python)
  * [Calculating run time](#calculating-run-time)
  * [Obtain file extension](#obtain-file-extension)
* [Deep Learning Related](#deep-learning-related)
  * [Anaconda](#anaconda)
  * [Tensorflow](#tensorflow)
    * [Tensorboard](#tensorboard)
    * [Tensorflow GPU Statement](#tensorflow-gpu-statement)

# Numpy
## Save and read numpy array


## Save and read pkl
- Saving datat
```
data = (1, 2, 3, [[1,2,3],[4,5,6]], ['a','b','c'])
with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)
 ```
 - Reading data
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

### Remove files in a directory with a specific extension
```
import os

dir_name = "C:\PycharmProjects\Icon_Detector_v1\data_160_yolo1\images/"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".xml"):
        os.remove(os.path.join(dir_name, item))
```

### Move or Copy files from one directory to another with specific extensions
[Reference](https://stackoverflow.com/questions/45136427/python-moving-files-based-on-extensions)
```
import glob
import shutil
source = '/home/xxx/randomdir/'
mydict = {
    '/home/xxx/Pictures': ['jpg','png','gif'],
    '/home/xxx/Documents': ['doc','docx','pdf','xls']
}
for destination, extensions in mydict.items():
    for ext in extensions:
        for file in glob.glob(source + '*.' + ext):
            print(file)
            shutil.move(file, destination)
```

My implementation: move files with xml extension from source directory to the destination directory.  (Windows)
```
import glob
import shutil

source = 'C:\\PycharmProjects\\Icon_Detector_v1\\data_160\\'
mydict = {
    r'C:\Users\Windows\Downloads\tensorflow-yolov3-master\data\image_160\Annotations': ['xml']
}
for destination, extensions in mydict.items():
    for ext in extensions:
        for file in glob.glob(source + '*.' + ext):
            shutil.copy(file, destination)
```



## Basic Operation

### Screenshots
`Alt+Prt` Scrn to take a screenshot of a window.

`Shift+Prt` Scrn to take a screenshot of an area you select.

`Shift+Ctrl+Prt` Scrn to take a screenshot of an area you select.

In windows: `shift+Win+S` Scrn to take a screenshot of an area you select.

### Create new sudo user
[Reference](https://linuxize.com/post/how-to-create-a-sudo-user-on-ubuntu/)

- Create the user and enter the password, `sudo adduser username`

- Press ENTER to accept the defaults user information.

- add the user to sudo, `sudo usermod -aG sudo username`

- switch users, `su - username`
 

# Python

## Calculating run time
[Reference](https://stackoverflow.com/questions/5622976/how-do-you-calculate-program-run-time-in-python)

```
from datetime import datetime
start=datetime.now()

#Statements

print(datetime.now()-start)
```

## Obtain file extension

```
def getImageName(file_location):
    filename = file_location.split('/')[-1]  
    location = file_location.split('/')[0:-1] 
    filename = filename.split('.')
    filename[0] += "_resized"
    filename = '.'.join(filename)
    new_path = '/'.join(location) + '/' + filename
    return new_path
```

Given an input as, `file_location = '/home/dong/Downloads/data_30_new/OnePlus_7T_Jun_26_2020_17_08_30.png'`

Then we can get the output as,
`filename = 'OnePlus_7T_Jun_26_2020_17_08_30.png'`

`location = ['', 'home', 'dong', 'Downloads', 'data_30_new']`

`new_path = '/home/dong/Downloads/data_30_new/OnePlus_7T_Jun_26_2020_17_08_30_resized.png'`



# Deep Learning Related

## Anaconda
- create a new environment In Anaconda: `conda create -n py3.5 python=3.5`
- check the environment info: `conda info --envs`

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


