from sys import argv
import cv2
import numpy as np

label = argv[1]
img = cv2.imread(argv[2])
csvpath = 

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.bitwise_not(img)
img = cv2.medianBlur(img, ksize=15)
img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)


with open(csvpath, "a+"):

