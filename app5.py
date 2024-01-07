"""
we use opencv and python. Simple object tracking with OpenCV. write us a script to get bounding box coordinates and compute centroids



"""

import cv2

# Load the video or camera feed
video_path = "path_to_your_video_file.mp4"  # Set to 0 for webcam input
cap = cv2.VideoCapture(2)

# Create a tracker
tracker = cv2.TrackerCSRT_create()

# Initialize variables
bounding_box = None
centroids = []

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame for faster processing (optional)
    frame = cv2.resize(frame, (800, 600))

    # Select the region of interest (ROI) using a bounding box
    if bounding_box is None:
        bounding_box = cv2.selectROI(
            "Object Tracking", frame, fromCenter=False, showCrosshair=True
        )
        tracker.init(frame, bounding_box)

    # Update the tracker and get the new bounding box coordinates
    success, bounding_box = tracker.update(frame)

    if success:
        # Compute the centroid based on the bounding box coordinates
        x, y, w, h = map(int, bounding_box)
        centroid_x = int(x + (w / 2))
        centroid_y = int(y + (h / 2))

        # Draw the bounding box and centroid on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)

        # Store the centroid coordinates
        centroids.append((centroid_x, centroid_y))

    # Display the frame with bounding box and centroid
    cv2.imshow("Object Tracking", frame)

    # Exit the loop by pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video or camera feed and close all windows
cap.release()
cv2.destroyAllWindows()

# Print the computed centroids
print("Centroids:")
for centroid in centroids:
    print(centroid)
