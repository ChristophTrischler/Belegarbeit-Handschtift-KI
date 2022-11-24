import numpy as np

from image import *
from model import *
import cv2
from sys import argv

model = loadModel()


def main():
    img = cv2.imread(argv[1])
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    img = np.array(img, dtype=np.uint8)
    img = np.array([img])
    print(testImgs(img, model))


if __name__ == "__main__":
    main()