import cv2
import keras.utils
import numpy as np
from matplotlib import pyplot as plt


def createNumImages(img=cv2.imread("examples/Test8.jpeg", 1)):
    height, width, _ = img.shape

    # FIND THE RED POINTS
    # create convert to hsv for better color specification
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # create mask with only red pixels
    mask = cv2.inRange(hsv, np.array([150, 120, 100]), np.array([197, 199, 187]))
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
    pts_dst = np.array([[0, 0], [74, 0], [0, 368], [74, 368]], np.float32)
    pts_dst *= 10
    print(pts_dst)
    # creating of a rotation matrix
    m = cv2.getPerspectiveTransform(pts_src, pts_dst)
    # rotating the img in to the new format
    target = cv2.warpPerspective(img, m, (740, 3680))
    # convert img to grey and invert img
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    target = cv2.bitwise_not(target)
    # filter the img by the brightness
    brightness_mask = cv2.threshold(target, 150, 255, cv2.THRESH_BINARY)[1]

    target = cv2.bitwise_and(target, target, mask=brightness_mask)

    plt.imshow(target)
    plt.show()

    cv2.imwrite("out/target.png", target)

    num_imgs = [target[x*360+100:(x+1)*360+20, 400:-60] for x in range(10)]
    num_imgs = [cv2.blur(i, (44, 44)) for i in num_imgs]
    num_imgs = [cv2.threshold(i, 50, 255, cv2.THRESH_BINARY)[1] for i in num_imgs]
    num_imgs = [cv2.resize(i, (32, 32), interpolation=cv2.INTER_AREA) for i in num_imgs]

    for i, x in enumerate(num_imgs):
        cv2.imwrite(f"out/nums/img_{i}.png", cv2.cvtColor(np.invert(x), cv2.COLOR_GRAY2BGR))

    num_imgs = np.array(num_imgs, np.float32)
    num_imgs = np.expand_dims(num_imgs, axis=-1)
    num_imgs /= 255

    return target, num_imgs


def main():
    with open("output.txt", "w", encoding="UTF-8") as f:
        imgs = createNumImages()
        for i, x in enumerate(imgs):
            for row in x:
                f.write(str(row).replace("\n", "") + "\n")
            cv2.imwrite(f"nums/img_{i}.png", x)


if __name__ == "__main__":
    main()
