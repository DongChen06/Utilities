# https://stackoverflow.com/questions/29384358/solving-system-of-ode-using-matlab/29386582#29386582
import numpy as np
import matplotlib.pyplot as plt
from math import exp


def dydt(t, y):
    dy = np.array(y)

    G = 3.16
    g = 0.1 * exp(-((t - 200) / 90) ** 2)

    dy[0] = -2 * 2 * y[0] + 2 * G * y[4] + 2 * g * y[6]
    dy[1] = 2 * y[0] - 2 * G * y[4]
    dy[2] = 2 * y[0] - 2 * g * y[6]
    dy[3] = -2 * y[3] + g * y[8]
    dy[4] = -2 * y[4] + G * (y[1] - y[0]) + g * y[7]
    dy[5] = -2 * y[5] - G * y[8]
    dy[6] = -2 * y[6] + g * (y[2] - y[0]) + G * y[7]
    dy[7] = -G * y[6] - g * y[4]
    dy[8] = G * y[5] - g * y[3]
    return dy


def RK4Step(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(x + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(x + h, y + h * k3)
    return (k1 + 2 * (k2 + k3) + k4) / 6.0


t = np.linspace(0, 500, 200 + 1)
dt = t[1] - t[0]
y0 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])

y = [y0]

for t0 in t[0:-1]:
    N = 200
    h = dt / N
    for i in range(N):
        y0 = y0 + h * RK4Step(dydt, t0 + i * h, y0, h)
    y.append(y0)

y = np.array(y)
plt.subplot(121)
plt.title("y(1)")
plt.plot(t, y[:, 0], "b.--")
plt.subplot(122)
plt.title("y(2)")
plt.plot(t, y[:, 1], "b-..")
plt.show()
