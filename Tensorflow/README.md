## Tensorboard:
- On linux: `tensorboard --logdir=logs`
 
- On windows: `tensorboard --logdir=log/ --host localhost --port 8088`

- Tensorflow GPU Statement:
```
from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```
If we donot want to use GPU, we can set it as -1.
