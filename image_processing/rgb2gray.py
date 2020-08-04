from PIL import Image
import cv2
image = 'Apple_iPhone-6S Plus_Jul_4_2020_11_08_28.png'
width = 720
height = 1280

img = cv2.imread(image)
h, w = img.shape[:2]
ratio = w / h
h1 = height
w1 = round(h1 * ratio)
scaled_img = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_AREA)
gray_image = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('grayscale.jpg', gray_image)
