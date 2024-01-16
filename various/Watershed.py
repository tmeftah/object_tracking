import time
import cv2
import numpy as np


# Create an object to read from camera
cap = cv2.VideoCapture(0)
# we check if the camera is opened previously or not
if cap.isOpened() == False:
    print("Error reading video file")


while True:
    ret, frame = cap.read()

    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0
        markers = cv2.watershed(frame, markers)
        frame[markers == -1] = [255, 0, 0]

        cv2.imshow("frame", frame)
        cv2.imshow("thresh", thresh)
        cv2.imshow("sure_fg", sure_fg)

        # Press Q to stop the process
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# When everything is done, release the video capture and videi write objects
cap.release()
cv2.destroyAllWindows()
