import cv2
import numpy as np
import time
from centroidtracker import CentroidTracker


# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)


start_time = time.time()
frame_count = 0

cap = cv2.VideoCapture(0)

prev_frame = None

lower_color = np.array([136, 87, 111])
upper_color = np.array([180, 255, 255])
min_area = 300


# lower boundary RED color range values; Hue (0 - 10)
lower1 = np.array([0, 100, 20])
upper1 = np.array([10, 255, 255])

# upper boundary RED color range values; Hue (160 - 180)
lower2 = np.array([160, 100, 20])
upper2 = np.array([180, 255, 255])


# Define line coordinates
line_start = (0, 250)  # Adjust these values based on your conveyor belt setup
line_end = (640, 250)

gap = 70
line_start2 = (
    0,
    line_start[1] + gap,
)  # Adjust these values based on your conveyor belt setup
line_end2 = (640, line_end[1] + gap)

# Set up variables for counting
object_count = []
object_crossed = []

fps = 0
while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    # Increment frame count
    frame_count += 1

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

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

    # Convert the frame to HSV color space

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_mask = cv2.inRange(hsv_frame, lower1, upper1)
    upper_mask = cv2.inRange(hsv_frame, lower2, upper2)

    full_mask = lower_mask + upper_mask

    full_mask = cv2.GaussianBlur(full_mask, (7, 7), 0)
    full_mask = cv2.threshold(full_mask, 0, 255, cv2.THRESH_OTSU)[1]

    result = cv2.bitwise_and(frame, frame, mask=full_mask)

    # Create a mask using the color thresholds
    # mask = cv2.inRange(hsv_frame, lower_color, upper_color)

    # Apply morphological operations to remove noise
    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)

    # Draw the line on the frame
    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)
    cv2.line(frame, line_start2, line_end2, (0, 255, 255), 2)

    # cv2.line(frame, (10, 1), (10, 420), (200, 200, 255), 2)

    # Find contours of the objects in the mask
    contours, _ = cv2.findContours(
        full_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Iterate through the contours and find the centroid of the moving object
    rects = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            rects.append(contour)
            # Calculate the centroid of the contour
            M = cv2.moments(contour)
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
            # Draw a circle at the centroid
            # cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)

    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = ct.update(rects)

    # loop over the tracked objects

    for objectID, centroid in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(
            frame,
            text,
            (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # Check if the centroid crosses the line
        if centroid[1] > line_start[1] and centroid[1] < line_end2[1]:
            if not objectID in object_crossed:
                # object_crossed = True
                # object_count += 1
                object_crossed.append(objectID)
        else:
            if objectID in object_crossed:
                object_count.append(objectID)
                del object_crossed[object_crossed.index(objectID)]

    for objectID in object_crossed:
        if objectID not in [objectID for objectID, centroid in objects.items()]:
            del object_crossed[object_crossed.index(objectID)]

    # Print FPS on the frame
    cv2.rectangle(frame, (0, 0), (640, 30), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"FPS:{round(fps, 1)} Objects:{len(rects)}  Count:{len(object_count)} crossed:{len(object_crossed)}",
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        1,
    )
    # Display the frame
    cv2.imshow("Red Color Detection", frame)
    # cv2.imshow("hsv_frame", hsv_frame)
    # cv2.imshow("mask", mask)
    # cv2.imshow("result", result)
    cv2.imshow("full_mask", full_mask)
    key = cv2.waitKey(1)
    # Exit the loop if 'q' is pressed
    if key & 0xFF == ord("q"):
        break

    if key == ord("a"):
        print(object_count)
        object_count = []


# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
