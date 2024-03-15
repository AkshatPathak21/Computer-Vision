import numpy as np
import cv2

def project_points_onto_image(points_3d, camera_matrix, distortion_coeffs, rvec=None, tvec=None):
    """
    Project 3D points onto a 2D image plane using the camera model.

    Parameters:
        points_3d (numpy.ndarray): Array of 3D points (shape: Nx3) in the world coordinates.
        camera_matrix (numpy.ndarray): The camera matrix.
        distortion_coeffs (numpy.ndarray): The distortion coefficients.
        rvec (numpy.ndarray, optional): Rotation vector (3x1) representing the camera orientation.
        tvec (numpy.ndarray, optional): Translation vector (3x1) representing the camera position.

    Returns:
        points_2d (numpy.ndarray): Array of 2D points (shape: Nx2) in the image plane.
    """
    if rvec is None or tvec is None:
        # If rvec and tvec are not provided, assume the camera is at the origin looking along the z-axis
        rvec = np.zeros(3)
        tvec = np.zeros(3)

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Perform projection
    points_2d, _ = cv2.projectPoints(points_3d, R, tvec, camera_matrix, distortion_coeffs)

    # Convert points to numpy array and reshape to Nx2
    points_2d = np.squeeze(points_2d)
    if len(points_2d.shape) == 1:
        points_2d = np.expand_dims(points_2d, axis=0)

    return points_2d

# Example usage:
# Suppose you have a 3D point (X, Y, Z) in the world coordinates
point_3d = np.array([[1.0, 2.0, 3.0]])

fx = 1000.0
fy = 1000.0
cx = 320.0
cy = 240.0
k1 = 0.1
k2 = -0.2
p1 = 0.01
p2 = -0.01
k3 = 0.0
# Load the camera matrix and distortion coefficients obtained from calibration
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]])
distortion_coeffs = np.array([k1, k2, p1, p2, k3])

# Optional: Provide the camera's rotation vector (rvec) and translation vector (tvec)
rvec = np.array([0.1, -0.2, 0.3])  # Example rotation vector
tvec = np.array([1.0, -1.0, 5.0])  # Example translation vector

# Perform projection
points_2d = project_points_onto_image(point_3d, camera_matrix, distortion_coeffs, rvec, tvec)

print("Projected 2D point:", points_2d)
