import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def estimate_defocus_blur(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_map = cv2.Laplacian(gray_image, cv2.CV_64F)
    blur_score = np.var(blur_map)
    return blur_score


def estimate_depth_from_defocus(blur_scores):
    pass


def generate_depth_map(images):
    depth_map = np.zeros_like(images[0], dtype=np.float32)
    for image in images:
        blur_score = estimate_defocus_blur(image)
        depth = estimate_depth_from_defocus(blur_score)
        depth_map += depth

    depth_map /= len(images)
    return depth_map


def reconstruct_3d_model(depth_map, focal_length, sensor_size):
    depth_values = depth_map * (sensor_size / focal_length)

    rows, cols = depth_map.shape[:2]
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    x -= cols / 2
    y -= rows / 2
    x *= depth_values
    y *= depth_values
    points_3d = np.stack((x, y, depth_values), axis=-1)

    return points_3d


def visualize_3d_model(points_3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]

    ax.scatter(X, Y, Z, c='b', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mean_x = X.mean()
    mean_y = Y.mean()
    mean_z = Z.mean()
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)

    plt.show()

