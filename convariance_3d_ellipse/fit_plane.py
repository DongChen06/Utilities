import urllib
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imread
import os
from mpl_toolkits.mplot3d import Axes3D
import math
from os.path import join, isdir
import scipy.linalg

img_temp = cv2.imread('959_Intensity.png')
XYZ_range_intensity_temp = np.load('959_XYZ_range_intensity.npy')

img = np.zeros_like(img_temp)
img[:, 0:200, :] = img_temp[:, 824:1024, :]
img[:, 200:400, :] = img_temp[:, 0:200, :]
img[:, 400:1024, :] = img_temp[:, 200:824, :]

XYZ_range_intensity = np.zeros_like(XYZ_range_intensity_temp)
XYZ_range_intensity[:, 0:200, :] = XYZ_range_intensity_temp[:, 824:1024, :]
XYZ_range_intensity[:, 200:400, :] = XYZ_range_intensity_temp[:, 0:200, :]
XYZ_range_intensity[:, 400:1024, :] = XYZ_range_intensity_temp[:, 200:824, :]

synch_dirc = 'large_sphere_whiteboard/lidar_camera_rail/4/lidar_camera'
intensity_image = img
img2_temp = cv2.imread('959_range.png')
img2 = np.zeros_like(img2_temp)
img2[:, 0:200, :] = img2_temp[:, 824:1024, :]
img2[:, 200:400, :] = img2_temp[:, 0:200, :]
img2[:, 400:1024, :] = img2_temp[:, 200:824, :]
range_image = img2
np.save('_XYZ_range_intensity.npy', XYZ_range_intensity)
cv2.imwrite('_Intensity.png', intensity_image)
cv2.imwrite('_range.png', range_image)

##------------- fill the nan values
range_img = XYZ_range_intensity[:, :, 3:4]
col_mean_ran = np.nanmean(range_img, axis=0)
inds_ran = np.where(np.isnan(range_img))
range_img[inds_ran] = np.take(col_mean_ran, inds_ran[1])

intensity_img = XYZ_range_intensity[:, :, 4:5]
col_mean_inten = np.nanmean(range_img, axis=0)
inds_inten = np.where(np.isnan(intensity_img))
intensity_img[inds_inten] = np.take(col_mean_inten, inds_inten[1])

X = XYZ_range_intensity[:, :, 0:1]
col_mean_x = np.nanmean(X, axis=0)
inds_x = np.where(np.isnan(X))
X[inds_x] = np.take(col_mean_x, inds_x[1])

Y = XYZ_range_intensity[:, :, 1:2]
col_mean_y = np.nanmean(Y, axis=0)
inds_y = np.where(np.isnan(Y))
Y[inds_y] = np.take(col_mean_y, inds_y[1])

Z = XYZ_range_intensity[:, :, 2:3]
col_mean_z = np.nanmean(Z, axis=0)
inds_z = np.where(np.isnan(Z))
Z[inds_z] = np.take(col_mean_z, inds_z[1])

XYZ_range_intensity[:, :, 0:1] = X
XYZ_range_intensity[:, :, 1:2] = Y
XYZ_range_intensity[:, :, 2:3] = Z
XYZ_range_intensity[:, :, 3:4] = range_img
XYZ_range_intensity[:, :, 4:5] = intensity_img

# the [x, y] for each right-click event will be stored here
right_clicks = list()


# this function will be called whenever the mouse is right-clicked
def mouse_callback(event, x, y, flags, params):
    # right-click event value is 2
    if event == 2:
        global right_clicks
        # store the coordinates of the right-click event
        right_clicks.append([x, y])
        # this just verifies that the mouse data is being collected
        # you probably want to remove this later
        print(right_clicks)


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

# extracted the sphere part by a plane. and generate a label
region_x1 = selected_points[:, 1].min()
region_x2 = selected_points[:, 1].max()
region_y1 = selected_points[:, 0].min()
region_y2 = selected_points[:, 0].max()
xs = XYZ_range_intensity[region_x1:region_x2, region_y1:region_y2, 0]
xs = np.reshape(xs, np.product(xs.shape))
ys = XYZ_range_intensity[region_x1:region_x2, region_y1:region_y2, 1]
ys = np.reshape(ys, np.product(ys.shape))
zs = XYZ_range_intensity[region_x1:region_x2, region_y1:region_y2, 2]
zs = np.reshape(zs, np.product(zs.shape))

data = np.c_[ys,zs, xs]
# regular grid covering the domain of the data
mn = np.min(data, axis=0)
mx = np.max(data, axis=0)
X,Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
# X,Y = np.meshgrid(np.arange(-3.0, 3.0, 0.5), np.arange(-3.0, 3.0, 0.5))
XX = X.flatten()
YY = Y.flatten()

# best-fit linear plane
A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients

# evaluate it on grid
Z = C[0]*X + C[1]*Y + C[2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(ys, zs, xs, marker='o')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.7)

plt.legend()
ax.set_xlabel('Y Label')
ax.set_ylabel('Z Label')
ax.set_zlabel('X Label')
plt.show()
