import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv


def readcsv(files):
    """Read csv file """
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append((row[2]))
        x.append((row[1]))
    return [float(i) for i in x[1:]], [float(i) * 20 for i in y[1:]]

plt.figure()
x, y = readcsv("ma2c_dial_i7/log_cnet.csv")
plt.plot(x, y, 'g', label='CNet')

x1, y1 = readcsv("ma2c_dial_i7/log_dial.csv")
plt.plot(x1, y1, color='b', label='DIAL')

x2, y2 = readcsv("ma2c_dial_i7/log_dial_i6.csv")
plt.plot(x2, y2, color='red', label='I6')

x3, y3 = readcsv("ma2c_dial_i7/log_i7.csv")
plt.plot(x3, y3, color='black', label='I7')

# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)

# plt.ylim(0, 16)
# plt.xlim(0, 104800)
plt.xlabel('Steps', fontsize=10)
plt.ylabel('Rewards', fontsize=10)
plt.legend(fontsize=8)
plt.show()
