import random
import numpy as np

def fit_line_ransac(data, num_iterations, threshold):
    best_line = None
    best_inliers = []
    
    for _ in range(num_iterations):
        # Randomly select two points from the data
        sample = random.sample(data, 2)
        
        # Fit a line to the selected points
        x1, y1 = sample[0]
        x2, y2 = sample[1]
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        # Count inliers
        inliers = []
        for point in data:
            x, y = point
            distance = abs(y - (slope * x + intercept))
            if distance < threshold:
                inliers.append(point)
        
        # Update best model if this iteration has more inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_line = (slope, intercept)
    
    return best_line, best_inliers

# Example usage
data = [(1, 2), (2, 3), (3, 5), (4, 4), (5, 6), (6, 8), (7, 7), (8, 10)]
num_iterations = 100
threshold = 1.0

best_line, best_inliers = fit_line_ransac(data, num_iterations, threshold)
print("Best Line:", best_line)
print("Inliers:", best_inliers)