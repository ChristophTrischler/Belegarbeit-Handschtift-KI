import cv2
import numpy as np
import matplotlib.pyplot as plt
from sys import argv


"""read img src, thres min, blurs ize from commandline arguments"""
img = cv2.imread(argv[1])
t = int(argv[2])
b = int(argv[3])

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.bitwise_not(img)
plt.imshow(img)
plt.title("src")
plt.show()
tresh = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)[1]
plt.imshow(tresh)
plt.title("none")
plt.show()
blur = cv2.blur(img, (int(b/2+1), int(b/2+1)))
blur = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY)[1]
plt.imshow(blur)
plt.title("blur")
plt.show()
gausin = cv2.GaussianBlur(img, (b, b), 0)
gausin = cv2.threshold(gausin, t, 255, cv2.THRESH_BINARY)[1]
plt.imshow(gausin)
plt.title("gausin")
plt.show()
median = cv2.medianBlur(img, ksize=int(b/2+1))
median = cv2.threshold(median, t, 255, cv2.THRESH_BINARY)[1]
plt.imshow(median)
plt.title("median")
plt.show()