from sys import argv
import cv2
import numpy as np
import matplotlib.pyplot as plt

label = argv[1]
img = cv2.imread(argv[2])
csvpath = argv[3] if 3 < len(argv) else "test.csv"

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.bitwise_not(img)
img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

plt.imshow(img)
plt.show()


img = list(img.reshape(784))
img.insert(0, label)

s = ",".join([str(c) for c in img]) + "\n"

with open(csvpath, "a+") as csvF:
    csvF.write(s)

loadCsvDataset("test.csv")
