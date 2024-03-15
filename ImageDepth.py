import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('C:\Python Program\Computer Vision\Left.jpg', 0)
imgR = cv2.imread('C:\Python Program\Computer Vision\Right.png', 0)

imgL.resize(255,255)
imgR.resize(255,255)
stereo = cv2.StereoBM_create(numDisparities = 16, blockSize = 15)

disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
plt.show()
disparity.shape