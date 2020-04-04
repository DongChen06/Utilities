import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd


a = np.random.rand(2000, 3)*10


def update_graph(num):
    a = np.random.rand(2000 - num*200, 3)*10
    graph._offsets3d = (a[:, 0], a[:, 1], a[:, 2])
    title.set_text('3D Test, time={}'.format(num))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')

graph = ax.scatter(a[:, 0], a[:, 1], a[:, 2])

ani = matplotlib.animation.FuncAnimation(fig, update_graph, 10,
                                         interval=40, blit=False)

plt.show()
