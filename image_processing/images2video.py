import os
import cv2
from os import walk
from shutil import copyfile

input_dir = '/home/dong/PycharmProjects/DeepAutoTest/images'
out_dir = '/home/dong/PycharmProjects/DeepAutoTest/images_renamed'
os.makedirs(out_dir, exist_ok=True)
# get all the pictures in directory
images = []
ext = (".jpeg", ".jpg", ".png", "PNG")
for (dirpath, dirnames, filenames) in walk(input_dir):
    for filename in filenames:
        if filename.endswith(ext):
            images.append(os.path.join(dirpath, filename))

print("Working...")
for image in images:
    file_name = image.split('/')[-1]
    dst = out_dir + '/range-' + file_name.split('_')[0] + ".png"
    copyfile(image, dst)
    # os.system(
    #     "ffmpeg -r 1/5 -i /home/dong/PycharmProjects/DeepAutoTest/images_renamed/*.png -vcodec mpeg4 -y video.mp4")
    os.system("ffmpeg -r 1 -pattern_type glob -i '/home/dong/PycharmProjects/DeepAutoTest/images_renamed/range-*.png' -c:v libx264 video.mp4")
