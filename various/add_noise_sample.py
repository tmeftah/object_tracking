import numpy as np
import cv2


def add_salt_and_pepper_noise(image, noise_ratio=0.02):
    noisy_image = image.copy()
    h, w, c = noisy_image.shape
    noisy_pixels = int(h * w * noise_ratio)

    for _ in range(noisy_pixels):
        row, col = np.random.randint(0, h), np.random.randint(0, w)
        if np.random.rand() < 0.5:
            noisy_image[row, col] = [0, 0, 0]
        else:
            noisy_image[row, col] = [255, 255, 255]

    return noisy_image


height, width = 420, 420
b, g, r = 0x3E, 0x88, 0xE5  # orange
image = np.zeros((height, width, 3), np.uint8)
image[:, :, 0] = b
image[:, :, 1] = g
image[:, :, 2] = r


noisy_image = add_salt_and_pepper_noise(image, noise_ratio=0.02)
# Start coordinate, here (5, 5)
# represents the top left corner of rectangle
start_point = (5, 5)

# Ending coordinate, here (220, 220)
# represents the bottom right corner of rectangle
end_point = (220, 220)

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

# Using cv2.rectangle() method
# Draw a rectangle with blue line borders of thickness of 2 px
noisy_image = cv2.rectangle(noisy_image, start_point, end_point, color, thickness)

sub = cv2.subtract(image, noisy_image)


# --- take the absolute difference of the images ---
res = cv2.absdiff(image, noisy_image)

# --- convert the result to integer type ---
res = res.astype(np.uint8)

# --- find percentage difference based on number of pixels that are not zero ---
percentage = (np.count_nonzero(res) * 100) / res.size

print(percentage)

cv2.imshow("A New Image", image)
cv2.imshow("Noisy Image (Salt and Pepper)", noisy_image)
cv2.imshow("sub Image", sub)
cv2.imshow("res Image", res)

cv2.waitKey(0)
cv2.destroyAllWindows()
