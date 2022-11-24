import cv2
import numpy as np
import imutils
from imutils import contours
from model import testImg, loadModel, Labels


def getImgs(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
    img = cv2.bitwise_not(img)
    cv2.imwrite("test/img.png", img)
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    rects = [cv2.boundingRect(c) for c in cnts]
    rects.sort(key=lambda y:y[0])
    imgs = [img[y:y+h, x:x+w] for x, y, w, h in rects]
    imgs = [np.array(i, np.uint8) for i in imgs]
    imgs = [cv2.blur(i, (5, 5)) for i in imgs]
    imgs = [cv2.threshold(i, 50, 255, cv2.THRESH_BINARY)[1] for i in imgs]
    imgs = [cv2.resize(i, (32, 32), interpolation=cv2.INTER_AREA) for i in imgs]
    imgs = np.array(imgs, np.float32)
    imgs = np.expand_dims(imgs, axis=-1)
    imgs /= 255
    return imgs


def main():
    m = loadModel()
    img = cv2.imread(r"examples/Sack.png")
    print([i for i in range(36)])
    print(Labels)
    for x, i in enumerate(getImgs(img)):

        res = testImg(i, m)
        print(Labels[res])
        print(res)



if __name__ == "__main__":
    main()