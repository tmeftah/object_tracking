import numpy as np
import cv2


image0 = cv2.imread("./0.png")
image1 = cv2.imread("./1.png")

# Cropping an image
image0 = image0[70:220, 0::]
image1 = image1[70:220, 0::]

sub = cv2.subtract(image0, image1)


# --- take the absolute difference of the images ---
res = cv2.absdiff(image0, image1)

# --- convert the result to integer type ---
res = res.astype(np.uint8)

# --- find percentage difference based on number of pixels that are not zero ---
s = np.sum(res)/(255*res.size)
print(s)

percentage = (np.count_nonzero(res) * 100) / res.size

print(percentage)

cv2.imshow("A New Image", image0)
cv2.imshow("Noisy Image (Salt and Pepper)", image1)
cv2.imshow("sub Image", sub)
cv2.imshow("res Image", res)


cv2.waitKey(0)
cv2.destroyAllWindows()
