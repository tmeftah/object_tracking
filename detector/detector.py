# detect color and textur diff --> working

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from slice_image import slice_image
from PIL import Image
from collections import Counter
import random

import colorgram

from colorT import ColorThief
import math

start_time = time.time()
frame_count = 0

new_width = int(640)
new_height = int(480)


cam = cv2.VideoCapture("6.mp4")
success, image = cam.read()

# Resize the frame
# image = cv2.resize(image, (new_width, new_height))

fps = 0

slice_nb = 5
# image =cv2.imread("0.png")
# (y1:y2,x1:x2)
ROI = (60, -60, 0, -1)
img1list = slice_image(image[ROI[0] : ROI[1], ROI[2] : ROI[3]], slice_nb)


cv2.namedWindow("test")


ct = ColorThief()


# define the function to compute MSE between two images
def mse(img1, img2):
    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse = err / (float(h * w))
    return mse, diff


def compute_mse(image1, image2):
    # Convert the images to NumPy arrays
    image1_arr = np.array(image1)
    image2_arr = np.array(image2)

    # Calculate the squared difference between the images
    squared_diff = np.square(image1_arr - image2_arr)

    # Compute the mean of the squared difference
    mse = np.mean(squared_diff)

    return mse


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


while True:
    ret, frame = cam.read()
    if not ret:
        cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cam.read()
    # Increment frame count
    frame_count += 1

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # x1, y1 = random.randrange(0, 600), random.randrange(0, 300)
    # x2, y2 = random.randrange(10, 600), random.randrange(0, 300)
    # color = (
    #     random.randrange(0, 255),
    #     random.randrange(0, 255),
    #     random.randrange(0, 255),
    # )
    # cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=-1)

    # # Calculate FPS
    # fps = frame_count / elapsed_time

    # Calculate FPS every 5 seconds
    if elapsed_time > 2:
        # Calculate FPS
        fps = frame_count / elapsed_time

        # Print the FPS
        print(f"FPS:{round(fps, 1)}")

        # Reset variables
        frame_count = 0
        start_time = time.time()
    frame = cv2.resize(frame, (new_width, new_height))
    frame = frame[ROI[0] : ROI[1], ROI[2] : ROI[3]]
    if not ret:
        print("failed to grab frame")
        break

    errors_list = []
    img2list = slice_image(frame, slice_nb)
    for i, im_1 in enumerate(img1list):
        # 1) Check if 2 images are equals
        # convert the images to grayscale
        img1 = cv2.cvtColor(img1list[i], cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2list[i], cv2.COLOR_BGR2GRAY)

        error, diff = mse(img1, img2)
        ms = compute_mse(img1, img2)

        # ct.image = Image.fromarray(img1list[i])
        # domin1 = ct.get_color(quality=50)
        ##domin1 = colorgram.extract(Image.fromarray(img1list[i]), 1)[0].rgb

        # Get the pixel colors
        domin1 = Image.fromarray(img1list[i]).getdata()
        # Count the occurrence of each color
        domin1 = Counter(domin1)
        # Get the most common color (dominant color)
        domin1 = domin1.most_common(1)[0][0]

        # ct.image = Image.fromarray(img2list[i])
        # domin2 = ct.get_color(quality=50)
        ## domin2 = colorgram.extract(Image.fromarray(img2list[i]), 1)[0].rgb

        # Get the pixel colors
        domin2 = Image.fromarray(img2list[i]).getdata()
        # Count the occurrence of each color
        domin2 = Counter(domin2)
        # Get the most common color (dominant color)
        domin2 = domin2.most_common(1)[0][0]

        colorDifference = color_difference(domin1, domin2)

        errors_list.append(
            (img1, img2, diff, error, colorDifference, img1list[i], img2list[i], ms)
        )

        # print("Error:", error)
        # print("Color difference :", colorDifference)
    max_x = 0
    max_color = 0
    max_x = max(errors_list, key=lambda item: item[3])
    max_color = max(errors_list, key=lambda item: item[4])

    print(60 * "*")
    print("max_x Error:", round(max_x[3]), round(max_x[-1]))
    print("max_color Error:", round(max_color[4]))

    cv2.putText(
        frame,
        f"FPS:{round(fps, 1)}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
    )
    cv2.imshow("test", frame)
    if round(max_x[-1]) > 5:
        ######################
        cv2.putText(
            max_x[6],
            str(round(max_x[-1])),
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )

        # cv2.imshow("difference", max_x[2])
        cv2.imshow("org", max_x[5])
        cv2.imshow("sec", max_x[6])

    ######################

    if round(max_color[4]) > 7:
        cv2.putText(
            max_color[6],
            str(round(max_color[4])),
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )

        cv2.imshow("org_color", max_color[5])
        cv2.imshow("sec_color", max_color[6])

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break


cam.release()

cv2.destroyAllWindows()
