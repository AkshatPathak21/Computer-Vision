import cv2
import numpy as np

# Read the image
image = cv2.imread('path/to/your/image.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary_threshold = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

adaptive_threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

_, otsu_threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

edges = cv2.Canny(gray_image, 100, 200)

gray_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

markers = np.zeros_like(gray_image, dtype=np.int32)
markers[gray_image < 100] = 1
markers[gray_image > 150] = 2
cv2.watershed(gray_3ch, markers)
segmented_image = np.zeros_like(gray_3ch)
segmented_image[markers == 1] = [255, 0, 0]  
segmented_image[markers == 2] = [0, 0, 255] 

contours, _ = cv2.findContours(binary_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

meanshift = cv2.pyrMeanShiftFiltering(image, 20, 40)

rect = (50, 50, 450, 290)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)
mask = np.zeros(image.shape[:2], np.uint8)
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
grabcut_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
grabcut_result = image * grabcut_mask[:, :, np.newaxis]

cv2.imshow('Original Image', image)
cv2.imshow('Binary Thresholding', binary_threshold)
cv2.imshow('Adaptive Thresholding', adaptive_threshold)
cv2.imshow('Otsu\'s Thresholding', otsu_threshold)
cv2.imshow('Canny Edge Detection', edges)
cv2.imshow('Watershed Segmentation', segmented_image)
cv2.imshow('Contour Detection', contour_image)
cv2.imshow('Mean-Shift Segmentation', meanshift)
cv2.imshow('GrabCut Segmentation', grabcut_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
