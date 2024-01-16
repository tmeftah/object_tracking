import cv2


def slice_image(image, size=2):
    # Get image dimensions
    height, width, _ = image.shape

    # Calculate the size of each smaller image
    slice_height = height // size
    slice_width = width // size

    # Slice the image into 8 smaller images
    slices = []
    for i in range(size):
        for j in range(size):
            start_y = i * slice_height
            end_y = start_y + slice_height
            start_x = j * slice_width
            end_x = start_x + slice_width

            # Crop the image
            slice_img = image[start_y:end_y, start_x:end_x]
            slices.append(slice_img)

    return slices


if __name__ == "__main__":
    # Load the image
    image_path = "1.png"
    image = cv2.imread(image_path)

    # Slice the image
    sliced_images = slice_image(image)

    # Display the sliced images
    for i, img in enumerate(sliced_images):
        cv2.imshow(f"Sliced Image {i+1}", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
