import cv2
from matplotlib import pyplot as plt

img = cv2.imread("C:/Python Program/Computer Vision//banana.jpeg")

edges = cv2.Canny(img,110,180)

cv2.imshow("orignal image",img)
cv2.imshow("Edges",edges)

cv2.waitKey(0)
cv2.destroyAllWindows()