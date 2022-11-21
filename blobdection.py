import cv2
import numpy as np

# Read image
img = cv2.imread(r'C:\Users\chris\projects\Belegarbeit-Handschtift-KI\images\20-11-2022-16-53-09.png', cv2.IMREAD_GRAYSCALE)
# Set up the blob detector.
detector = cv2.SimpleBlobDetector_create()

# Detect blobs from the image.
keypoints = detector.detect(img)
pts = [key_point.pt for key_point in keypoints]
print(pts)

blank = np.zeros((1, 1))
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS - This method draws detected blobs as red circles and ensures that the size of the circle corresponds to the size of the blob.
blobs = cv2.drawKeypoints(img, keypoints, blank, (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

# Show keypoints
cv2.imwrite("out/test.png", blobs)
