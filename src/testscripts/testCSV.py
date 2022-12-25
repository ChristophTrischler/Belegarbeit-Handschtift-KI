import numpy as np
import matplotlib.pyplot as plt
from sys import argv
import cv2

datasetPath = argv[1] if 1 < len(argv) else "test.csv"


for row in open(datasetPath):
    # in csv row 0 for labels rest image data
    row = row.split(",")

    label = int(row[0])

    image = np.array(row[1:], dtype="uint8")
    image = image.reshape((28, 28))  # reshape from 1d array to 2d array
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)

    # can be used show letters from the Dataset
    plt.imshow(image)
    plt.show()