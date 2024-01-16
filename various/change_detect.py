import cv2
import os
import numpy as np

# File paths for the reference and given images
reference_image_path = "1.png"
given_image_path = "4.png"

# Check if both image files exist
if not os.path.exists(reference_image_path) or not os.path.exists(given_image_path):
    print("One or both image files do not exist.")
else:
    # Load the reference and given images
    reference_image = cv2.imread(reference_image_path)
    given_image = cv2.imread(given_image_path)

    # Convert images to grayscale for feature matching
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    given_gray = cv2.cvtColor(given_image, cv2.COLOR_BGR2GRAY)

    # Detect and match keypoints between the two images
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(reference_image, None)
    kp2, des2 = orb.detectAndCompute(given_image, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    print(matches)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    print(len(src_pts))
    print(len(dst_pts))

    # Estimate an affine transformation (ignores perspective distortion)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    if M is not None:
        # Apply the estimated transformation to the reference image
        aligned_reference = cv2.warpAffine(
            reference_image, M, (given_image.shape[1], given_image.shape[0])
        )

        # Compute the absolute difference between the aligned reference and given images
        diff_frame = cv2.absdiff(given_image, aligned_reference)

        diff_gray = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to highlight the differences
        _, thresholded = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to further enhance the differences
        thresholded = cv2.erode(thresholded, None, iterations=2)
        thresholded = cv2.dilate(thresholded, None, iterations=2)

        # Find contours of the differences
        contours, _ = cv2.findContours(
            thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw rectangles around the detected differences
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(given_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    given_image,
                    "Change Detected",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                print("Significant change detected.")

        # Display the given image with detected changes
        cv2.imshow("Change Detected", given_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No transformation found; images are too dissimilar.")
