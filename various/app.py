import time
import cv2
import numpy as np

# The duration in seconds for the video captured
capture_duration = 1000

# Create an object to read from camera
cap = cv2.VideoCapture(2)
# we check if the camera is opened previously or not
if cap.isOpened() == False:
    exit()

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("U", "Y", "V", "Y"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 / 2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 / 2)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn the autofocus off
print("Width=", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height=", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# VideoWriter object will create a frame of the above defined output is stored in 'output.avi' file.
result = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))

start_time = time.time()

while int(time.time() - start_time) < capture_duration:
    ret, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    frame = cv2.normalize(
        frame, None, alpha=10, beta=200, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 30)
    contours, _ = cv2.findContours(
        edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_image = np.zeros_like(frame)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    if ret == True:
        result.write(frame)
        cv2.imshow("OpenCVCam", frame)
        cv2.imshow("contour_image", contour_image)
        cv2.imshow("gray", gray)
        cv2.imshow("blur", blur)

        # Press Q to stop the process
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# When everything is done, release the video capture and videi write objects
cap.release()
result.release()
cv2.destroyAllWindows()
