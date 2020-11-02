import urllib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imread
import os
from os.path import join, isdir

input_dir = '/home/dong/PycharmProjects/DeepAutoTest/Dataset/message'
img = cv2.imread(input_dir + '/Samsung_Galaxy A10e_Jun_30_2020_10_07_22.png')
out_dir = '/home/dong/PycharmProjects/DeepAutoTest/Dataset/label'
os.makedirs(out_dir, exist_ok=True)
# the [x, y] for each right-click event will be stored here
right_clicks = list()


# this function will be called whenever the mouse is right-clicked
def mouse_callback(event, x, y, flags, params):
    # right-click event value is 2
    if event == 2:
        global right_clicks
        # store the coordinates of the right-click event
        right_clicks.append([x, y])
        # print coordinates of right clicks
        # print(right_clicks)


scale = 1
window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', window_width, window_height)

# set mouse callback function for window
cv2.setMouseCallback('image', mouse_callback)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

selected_points = np.asarray(right_clicks)
X = selected_points[:, 0]
Y = selected_points[:, 1]
# center of the clicked buttons
x_center = np.mean(X)
y_center = np.mean(Y)

# X = X.reshape((X.shape[0], 1))
# Y = Y.reshape((Y.shape[0], 1))

# # Formulate and solve the least squares problem ||Ax - b ||^2
# A = np.hstack([X ** 2, X * Y, Y ** 2, X, Y])
# b = np.ones_like(X)
# x = np.linalg.lstsq(A, b)[0].squeeze()
#
# # Print the equation of the ellipse in standard form
# print('The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'.format(x[0], x[1], x[2], x[3], x[4]))
#
# # Plot the least squares ellipse
# x_coord = np.linspace(0, (img.shape[1] - 1), img.shape[1])
# y_coord = np.linspace(0, (img.shape[0] - 1), img.shape[0])
# X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
# Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord ** 2 + x[3] * X_coord + x[4] * Y_coord
# # orientation_rad = 1/float(2*np.arctan(x[1]/(x[2]-x[0])))
# orientation_rad = (1 / 2.0) * float(np.arctan(x[1] / (x[2] - x[0])))
# cos_phi = np.cos(orientation_rad)
# sin_phi = np.sin(orientation_rad)
# x_new_0 = x[0] * cos_phi * cos_phi - x[1] * cos_phi * sin_phi + x[2] * sin_phi * sin_phi
# x_new_1 = 0
# x_new_2 = x[0] * sin_phi * sin_phi + x[1] * cos_phi * sin_phi + x[2] * cos_phi * cos_phi
# x_new_3 = x[3] * cos_phi - x[4] * sin_phi
# x_new_4 = x[3] * sin_phi + x[4] * cos_phi
# Z_coord_new = x_new_0 * X_coord ** 2 + x_new_1 * X_coord * Y_coord + x_new_2 * Y_coord ** 2 + x_new_3 * X_coord + x_new_4 * Y_coord
# mean_x = cos_phi * np.mean(X) - sin_phi * np.mean(Y)
# mean_y = sin_phi * np.mean(X) + cos_phi * np.mean(Y)
#
# x_0 = (-x_new_3) / (2 * x_new_0)
# x_1 = (-x_new_4) / (2 * x_new_2)
# x_0_tilted = cos_phi * x_0 + sin_phi * x_1
# x_1_tilted = -sin_phi * x_0 + cos_phi * x_1
#
# # convert to interger
# x_0_tilted = round(x_0_tilted)
# x_1_tilted = round(x_1_tilted)
# plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=1)
#
# plt.legend()
# plt.imshow(img)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.plot(x_0_tilted, x_1_tilted, marker='o', markersize=3, color="red")
# plt.show()

# generate gaussian label
sigma = 30
image = (np.zeros((img.shape[0], img.shape[1]))).astype('float')
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        r_square = (i - y_center) ** 2 + (j - x_center) ** 2
        image[i, j] = np.exp(-r_square / (2 * sigma * sigma))

image = image * 255
cv2.imwrite(os.path.join(out_dir, 'label.png'), image.astype('uint8'))
