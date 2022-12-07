import cv2
import keras.utils
import numpy as np
from matplotlib import pyplot as plt
from imutils import grab_contours
from sys import argv
import math

rows = 23
rowsHeigth = 200


def makeSquare(r):
    x, y, w, h = r
    if w < h:
        w = h
    else:
        h = w
    return x, y, w, h


def getBlobs(img):
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    rects = [cv2.boundingRect(c) for c in cnts]
    # filter by the area of the rects from the blobs
    rects = [makeSquare((x, y, w, h)) for x, y, w, h in rects if w * h > 400]  # img of an 'I' w = 2 h = 200 -> w = 200 h = 200
    rects.sort(key=lambda e: e[0])  # e->{x, y} => sort by x position
    imgs = [np.array(img[y:y + h, x:x + w], np.uint8) for x, y, w, h in rects]
    imgs = [cv2.resize(i, (32, 32), interpolation=cv2.INTER_AREA) for i in imgs]

    imgs = np.array(imgs, np.float32)
    imgs = np.expand_dims(imgs, axis=-1)
    imgs /= 255

    return imgs


def readTest(img=cv2.imread("examples/Test8.jpeg", 1)):
    height, width, _ = img.shape

    # FIND THE RED POINTS
    # create convert to hsv for better color specification
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # create mask with only red pixels
    mask = cv2.inRange(hsv, np.array([150, 120, 100]), np.array([200, 200, 200]))
    # test img output
    mask_img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite("out/mask_img.png", mask_img)

    # list of all red pixels in mask
    red_pxs = cv2.findNonZero(mask)
    red_pxs = [x[0] for x in red_pxs]

    # indexes of points closet to the corners
    up_left = np.argmin([np.sqrt(x ** 2 + y ** 2) for x, y in red_pxs])
    up_right = np.argmin([np.sqrt((width - x) ** 2 + y ** 2) for x, y in red_pxs])
    down_left = np.argmin([np.sqrt(x ** 2 + (height - y) ** 2) for x, y in red_pxs])
    down_right = np.argmin([np.sqrt((width - x) ** 2 + (height - y) ** 2) for x, y in red_pxs])
    # array of the corner points in the img
    pts_src = np.array([red_pxs[up_left], red_pxs[up_right], red_pxs[down_left], red_pxs[down_right]], np.float32)

    # array of points in the new img
    test_height = rows * rowsHeigth
    print(test_height)
    pts_dst = np.array([[0, 0], [3200, 0], [0, test_height], [3200, test_height]], np.float32)
    # creating of a rotation matrix
    m = cv2.getPerspectiveTransform(pts_src, pts_dst)
    # rotating the img in to the new format
    target = cv2.warpPerspective(img, m, (3200, test_height))
    # convert img to grey and invert img
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    target = cv2.bitwise_not(target)
    # filter the img by the brightness
    target = cv2.medianBlur(target, ksize=15)
    target = cv2.threshold(target, 120, 255, cv2.THRESH_BINARY)[1]

    cv2.imwrite("out/target.png", target)

    imgs = [np.array(target[y * rowsHeigth + 5:(y + 1) * rowsHeigth - 5, 1100: -100], np.uint8) for y in range(rows)]

    for i, x in enumerate(imgs):
        plt.subplot(5, 5, i+1)
        plt.imshow(x)
        cv2.imwrite(f"out/nums/let{i}.png", x)
    plt.show()

    imgs = [getBlobs(image) for image in imgs]

    return target, imgs


def main():
    m = loadModel()
    img = cv2.imread(argv[1])
    plt.imshow(img)
    plt.show()
    t, imgs = readTest(img)
    for i, y in enumerate(imgs):
        n = 1
        l: int = math.ceil(math.sqrt(len(y)))
        for x in y:
            plt.subplot(l, l, n)
            n += 1
            plt.imshow(x)
            #  cv2.imwrite(f"out/nums/img_{i}.png", x)
        plt.title(f"{i}")
        plt.show()
        print(testImgs(y, m))


if __name__ == "__main__":
    from model import loadModel, Labels, testImgs
    main()
