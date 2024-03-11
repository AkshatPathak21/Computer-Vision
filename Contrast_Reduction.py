import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread("C:\Python Program\Computer Vision\Jett.jpg")
rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

name = "Jett"
plt.title(name)
plt.imshow(rgb_img)
cv2.imwrite("Jett_BGR.jpeg",rgb_img)
plt.show()

#contrast reduction
crimg = cv2.bitwise_not(rgb_img)
name = "Contrast Reduction"
plt.imshow(crimg)
cv2.imwrite("Jett_CR.jpeg",crimg)
plt.show()

# Conversion to grayscale
gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
name = "Gray img"
plt.title(name)
plt.imshow(gray_img)
cv2.imwrite("Gray_jett.jpeg", gray_img)
plt.show()

# corrected gray img
crrgrayimg = cv2.cvtColor(gray_img,cv2.COLOR_BGR2RGB)
plt.title("Corrected Gray Image")
plt.imshow(crrgrayimg)
plt.show()