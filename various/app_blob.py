#!/usr/bin/env python3
import cv2
import numpy as np
import sys
from utils import measure_time


video = cv2.VideoCapture(0)


@measure_time
def blobs(image, show_area=False):
    # Read image

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create blob detector parameters
    params = cv2.SimpleBlobDetector_Params()

    # Set parameters
    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by area
    params.filterByArea = True
    params.minArea = 100

    # Filter by circularity
    params.filterByCircularity = True
    params.minCircularity = 0.6

    # Filter by convexity
    params.filterByConvexity = False
    params.minConvexity = 0.1

    # Filter by inertia : how elongated or elongated a blob is
    params.filterByInertia = True
    params.minInertiaRatio = 0.2

    # Create blob detector
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(gray)

    if show_area:
        # Draw detected blobs on the image
        image = cv2.drawKeypoints(
            image,
            keypoints,
            np.array([]),
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        # Draw the detected blobs and display the area
        for point in keypoints:
            x = int(point.pt[0])
            y = int(point.pt[1])
            area = int(
                point.size
            )  #  number of pixels enclosed by the detected blob is equal to area
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(
                image,
                f"Area: {area}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        # Display the image with the detected blobs and their areas
        cv2.imshow("Blobs", image)
    else:
        # Draw detected blobs as red circles
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(
            gray,
            keypoints,
            blank,
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        # Show the detected blobs
        cv2.imshow("Blobs", blobs)


if __name__ == "__main__":
    show_area = True
    while True:
        # Read a frame from the video
        ret, frame = video.read()

        # Check if the frame was successfully read
        if not ret:
            break

        # Display the frame
        blobs(frame, show_area)

        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            video.release()
            cv2.destroyAllWindows()
            break
