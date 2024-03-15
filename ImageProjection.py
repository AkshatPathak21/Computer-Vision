import cv2
import numpy as np

im = cv2.imread('C:\Python Program\Computer Vision\Jett.jpg', cv2.IMREAD_GRAYSCALE)
im = 255-im
proj = np.sum(im, 1)

m = np.max(proj)
w = 500

result = np.zeros((proj.shape[0], 500))

for row in range(im.shape[0]):
    cv2.line(result, (0, row), (int(proj[row]*w/m), row), (255, 255, 255), 1)

cv2.imshow('Original', im)
cv2.imshow('Horizontal projection', result)
cv2.waitKey(0)
cv2.destroyAllWindows()