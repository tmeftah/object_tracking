# detect color and textur diff --> working

import cv2
import numpy as np
import matplotlib.pyplot as plt
from slice_image import slice_image
from PIL import Image

from colorT import ColorThief
import math


ct = ColorThief()


# define the function to compute MSE between two images
def mse(img1, img2):
    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse = err / (float(h * w))
    return mse, diff


def color_difference(color1, color2):
    r1, g1, b1 = color1
    r2, g2, b2 = color2

    # Calculate the squared difference for each color channel
    diff_r = (r1 - r2) ** 2
    diff_g = (g1 - g2) ** 2
    diff_b = (b1 - b2) ** 2

    # Calculate the total Euclidean distance
    color_diff = math.sqrt(diff_r + diff_g + diff_b)

    # Calculate the percentage difference
    max_diff = math.sqrt(255**2 + 255**2 + 255**2)
    percentage_diff = (color_diff / max_diff) * 100

    return percentage_diff


slice_nb = 6

img1list = slice_image(cv2.imread("0.png"), slice_nb)
img2list = slice_image(cv2.imread("3.png"), slice_nb)

errors_list = []

for i, im_1 in enumerate(img1list):
    # 1) Check if 2 images are equals
    # convert the images to grayscale
    img1 = cv2.cvtColor(img1list[i], cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2list[i], cv2.COLOR_BGR2GRAY)

    error, diff = mse(img1, img2)

    ct.image = Image.fromarray(img1list[i])
    domin1 = ct.get_color(quality=7)

    ct.image = Image.fromarray(img2list[i])
    domin2 = ct.get_color(quality=7)

    colorDifference = round(color_difference(domin1, domin2))

    errors_list.append(
        (img1, img2, diff, error, colorDifference, img1list[i], img2list[i])
    )

    # print("Error:", error)
    # print("Color difference :", colorDifference)


max_x = max(errors_list, key=lambda item: item[3])
max_color = max(errors_list, key=lambda item: item[4])

print(20 * "*")
print("max_x Error:", round(max_x[3]))
print("max_color Error:", round(max_color[4]))


######################
cv2.putText(
    max_x[6],
    str(round(max_x[3], 0)),
    (10, 20),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0, 0, 0),
    2,
)

cv2.imshow("difference", max_x[2])
cv2.imshow("org", max_x[5])
cv2.imshow("sec", max_x[6])


######################

cv2.putText(
    max_color[6],
    str(round(max_color[4], 0)),
    (10, 20),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0, 0, 0),
    2,
)


cv2.imshow("org_color", max_color[5])
cv2.imshow("sec_color", max_color[6])

plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
