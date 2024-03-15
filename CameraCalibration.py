import numpy as np
import cv2

def calibrate_camera(image_paths, pattern_size, square_size):
    # Prepare object points based on the pattern size and square size
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    # Lists to store object points and image points from all calibration images
    obj_points = []  # 3D points in the world coordinates
    img_points = []  # 2D points in the image plane

    for image_path in image_paths:
        # Load the calibration image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            obj_points.append(objp)
            img_points.append(corners)

    # Perform camera calibration
    _, camera_matrix, distortion_coeffs, _, _ = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    return camera_matrix, distortion_coeffs

# Example usage:
# Provide a list of calibration images, pattern size, and square size
calibration_images = ['C:\Python Program\Computer Vision\img1.jpg',
                      'C:\Python Program\Computer Vision\img2.jpg',
                      'C:\Python Program\Computer Vision\img3.jpg']
pattern_size = (9, 6)  # Number of inner corners (rows, columns) in the calibration pattern
square_size = 25.0  # Size of each square in the calibration pattern (in millimeters)

# Perform camera calibration
camera_matrix, distortion_coeffs = calibrate_camera(calibration_images, pattern_size, square_size)

print("Camera Matrix:")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(distortion_coeffs)
