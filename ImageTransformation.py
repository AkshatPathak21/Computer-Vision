import cv2
import matplotlib.pyplot as plt
import numpy as np
bgrimg = cv2.imread("C:\Python Program\Computer Vision\Jett.jpg")
img = cv2.cvtColor(bgrimg,cv2.COLOR_BGR2RGB)

invimg = 255 - img
plt.title("Inverse Image")
plt.imshow(invimg)
plt.show()

crimg = cv2.bitwise_not(invimg)
plt.title("Contrast Inverse Image")
plt.imshow(crimg)
plt.show()

print(img[0][0])