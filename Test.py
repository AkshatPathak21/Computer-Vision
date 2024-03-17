import random
import numpy as np

def ransac(data, num_iter,threshold):
    best_line = None
    best_inliers = []
    for _ in range(num_iter):
        sample = random.sample(data,2)

        x1,y1 = sample[0]
        x2,y2 = sample[1]

        slope = y2-y1/x2-x1
        intercept = y1 - slope*x1

        inliers = []

        for point in data:
            x,y = point
            dist = abs(y-(slope*x+intercept))
            if dist<threshold:
                inliers.append(point)
        if len(inliers)>len(best_inliers):
            best_inliers=inliers
            best_line = (slope,intercept)

    return best_line,best_inliers