import cv2
import numpy as np

def remove_largest_black_particle(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image for black particles
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area (representing the largest black particle)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the largest black particle
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Remove the largest black particle by drawing it in white on the binary image
    cv2.rectangle(binary, (x, y), (x + w, y + h), (255, 255, 255), cv2.FILLED)

    # Draw rectangles around the remaining black particles on the original image
    black_particles_image = image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(black_particles_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(black_particles_image, "Hole", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    return black_particles_image

def remove_non_blue_particles(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for blue color detection in HSV
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise AND operation to extract blue regions
    blue_regions = cv2.bitwise_and(image, image, mask=mask)

    return blue_regions

# Read the image
image = cv2.imread("./defect images/hole/hole_7.jpeg")

# Resize the image to a smaller size
height, width = image.shape[:2]
new_width = 500
new_height = int((new_width / width) * height)
resized_image = cv2.resize(image, (new_width, new_height))

# Remove non-blue particles
blue_particles = remove_non_blue_particles(resized_image)

# Remove the largest black particle and draw rectangles around remaining black particles
image_with_black_particles = remove_largest_black_particle(blue_particles)

# Display the result
cv2.imshow("Image with Black Particles", image_with_black_particles)
cv2.waitKey(0)
cv2.destroyAllWindows()
