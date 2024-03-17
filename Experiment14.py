import cv2
import open3d as o3d

image_path = "input_image.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

point_cloud = o3d.geometry.PointCloud()
points = []
for keypoint in keypoints:
    depth = 1.0 
    point = o3d.geometry.PointXYZ(keypoint.pt[0], keypoint.pt[1], depth)
    points.append(point)
point_cloud.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([point_cloud])
