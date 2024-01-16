import cv2
import numpy as np

# Create an object to read from camera
cap = cv2.VideoCapture(0)
# we check if the camera is opened previously or not
if cap.isOpened() == False:
    print("Error reading video file")

while True:
    ret, rawImage = cap.read()

    cv2.imshow("Original Image", rawImage)

    hsv = cv2.cvtColor(rawImage, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV Image", hsv)

    hue, saturation, value = cv2.split(hsv)
    cv2.imshow("Saturation Image", saturation)

    retval, thresholded = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    cv2.imshow("Thresholded Image", thresholded)

    medianFiltered = cv2.medianBlur(thresholded, 5)
    cv2.imshow("Median Filtered Image", medianFiltered)
    cv2.waitKey(0)
    cnts, hierarchy = cv2.findContours(
        medianFiltered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for c in cnts:
        # compute the center of the contour

        M = cv2.moments(c)
        print(M)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    first = cv2.drawContours(rawImage, [c], -1, (0, 255, 0), 2)
    second = cv2.circle(rawImage, (cX, cY), 1, (255, 255, 255), -1)

    cv2.imshow("Objects Detected", rawImage)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
