import matplotlib.pyplot as plt

import image
import model
from sys import argv
import cv2
import tensorflow
import numpy as np


m = model.loadModel()


def main():
    img = cv2.imread(argv[1])
    img, numImgs = image.createNumImages(img)
    for x in numImgs:
        print(x.shape)
        plt.imshow(x)
    plt.show()

    model.testNumImg(numImgs, m)


if __name__ == "__main__":
    main()
