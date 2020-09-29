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
    return [float(i) for i in x[1:]], [float(i) for i in y[1:]]

plt.figure()
x, y = readcsv("log_cnet.csv")
x1, y1 = readcsv("log_dial.csv")
# x2, y2 = readcsv("log_i5.csv")
# x3, y3 = readcsv("log_i6.csv")
# x4, y4 = readcsv("log_i7.csv")
# x5, y5 = readcsv("log_i8.csv")
x6, y6 = readcsv("log_i9.csv")

plt.plot(x, y, label='CNet')
plt.plot(x1, y1, label='DIAL')
# plt.plot(x2, y2, label='I5')
# plt.plot(x3, y3, label='I6')
# plt.plot(x4, y4, label='I7')
# plt.plot(x5, y5, label='I8')
plt.plot(x6, y6, label='I9')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.ylim(0.35, 0.58)
plt.xlim(0, 40000)
plt.xlabel('Steps', fontsize=12)
plt.ylabel('Rewards', fontsize=12)
leg = plt.legend(fontsize=12, loc='lower right')

# set the linewidth of each legend object
for legobj in leg.legendHandles:
    legobj.set_linewidth(3.0)
    
plt.show()
