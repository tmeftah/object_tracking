import cv2
import numpy as np

# Configure video capture from camera
cap = cv2.VideoCapture(
    2
)  # Change the argument to the appropriate camera index if needed

# Define line coordinates
line_start = (0, 300)  # Adjust these values based on your conveyor belt setup
line_end = (640, 300)

# Set up variables for counting
object_count = 0
object_crossed = False

while True:
    # Read frame from the video feed
    ret, frame = cap.read()
    if not ret:
        break

    # Pre-process the frame if needed (e.g., resize, crop, or smooth)

    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper color thresholds (adjust these values for your specific color)
    lower_color = np.array([136, 87, 111])
    upper_color = np.array([180, 255, 255])

    # Create a mask based on the color thresholds
    color_mask = cv2.inRange(hsv_frame, lower_color, upper_color)

    # Apply morphological operations to remove noise and enhance the mask if needed

    # Find contours of objects in the mask
    contours, _ = cv2.findContours(
        color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Iterate through contours and track objects
    for contour in contours:
        if cv2.contourArea(contour) > 600:
            # Compute centroid of each contour
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Draw a circle at the centroid
            cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

            # Check if the centroid crosses the line
            if cY > line_start[1] and cY < line_end[1] + 10:
                if not object_crossed:
                    object_crossed = True
                    object_count += 1
            else:
                object_crossed = False

    # Draw the line on the frame
    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)

    # Display the frame and object count
    cv2.putText(
        frame,
        f"Object Count: {object_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    cv2.imshow("Conveyor Belt Object Count", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
