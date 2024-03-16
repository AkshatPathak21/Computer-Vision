import cv2
import numpy as np

def reconstruct_3d_model(stereo_image_pair, focal_length, baseline):
    left_image, right_image = stereo_image_pair

    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    disparity = stereo.compute(left_gray, right_gray)

    depth_map = (focal_length * baseline) / disparity

    return depth_map

left_image = cv2.imread('left_image.jpg')
right_image = cv2.imread('right_image.jpg')

K = np.array([[638, 0, 725],
              [0, 649, 384],
              [0, 0, 1]])
baseline = 0.1  

depth_map = reconstruct_3d_model((left_image, right_image), focal_length=K[0, 0], baseline=baseline)

cv2.imshow('Depth Map', depth_map / depth_map.max()) 
cv2.waitKey(0)
cv2.destroyAllWindows()
