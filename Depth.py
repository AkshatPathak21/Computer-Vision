import numpy as np
import cv2

def generate_depth_map(image_shape, constant_depth):
    """
    Generate a synthetic depth map with constant depth.

    Parameters:
        image_shape (tuple): Shape of the output depth map (rows, columns).
        constant_depth (float): Constant depth value for all pixels in the depth map.

    Returns:
        depth_map (numpy.ndarray): Synthetic depth map.
    """
    depth_map = np.ones(image_shape, dtype=np.float32) * constant_depth
    return depth_map

# Example usage:
# Image size (rows, columns)
image_shape = (1920,1080)

# Constant depth value in arbitrary units
constant_depth = 5.0

# Generate the synthetic depth map
depth_map = generate_depth_map(image_shape, constant_depth)

# Display the depth map as a grayscale image
depth_map_normalized = (depth_map / constant_depth * 255).astype(np.uint8)
cv2.imshow("Depth Map", depth_map_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
