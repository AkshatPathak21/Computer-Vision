import cv2
import numpy as np

images = ['left_image.jpg', 'right_image.jpg'] 
K = np.array([[638, 0, 725],
              [0, 649, 384],
              [0, 0, 1]])

def reconstruct_3d_model(images, K):
    feature_detector = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    points_3d = []

    for i in range(len(images) - 1):
        keypoints1, descriptors1 = feature_detector.detectAndCompute(images[i], None)
        keypoints2, descriptors2 = feature_detector.detectAndCompute(images[i + 1], None)
        matches = matcher.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        points1 = np.array([keypoints1[match.queryIdx].pt for match in matches])
        points2 = np.array([keypoints2[match.trainIdx].pt for match in matches])
        E, _ = cv2.findEssentialMat(points1, points2, cameraMatrix=K)
        _, R, t, _ = cv2.recoverPose(E, points1, points2, cameraMatrix=K)

        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((R, t))
        points_4d_hom = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
        points_3d_hom = points_4d_hom / points_4d_hom[3] 
        points_3d.append(points_3d_hom[:3].T)

    points_3d = np.vstack(points_3d)

    return points_3d

points_3d = reconstruct_3d_model(images, K)

