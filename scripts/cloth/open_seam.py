import cv2
import numpy as np
from PIL import Image

def detect_open_seam(image):
    original_image = Image.open(image)
    original_image = np.array(original_image)
    output = original_image.copy() # Prevent original image from being edited
    detected_defect = 0

    # Setting kernels for morphological operation
    big_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) # Used for removing unremoved noises

    # Convert to grayscale
    grayscale_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Remove Noise (Median Filter)
    median_filtered_img = cv2.medianBlur(grayscale_img, 5)

    # Adaptive Thresholding (Otsu Method)
    T, threshold_img = cv2.threshold(median_filtered_img, 0, 255, cv2.THRESH_OTSU)
    
    # Morphological Operation (Opening)
    # Getting stitches
    opening_img = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN, big_kernel)

    # Subtracting opened image with original image to find any difference between the images
    stitch_img = cv2.subtract(threshold_img, opening_img)

    # Removing small dots that are not removed by opening (Using small kernel to prevent the detected stitch from eroding too much)
    stitch_img = cv2.morphologyEx(stitch_img, cv2.MORPH_OPEN, small_kernel)
    
    # Finding contours
    contours, hierarchy = cv2.findContours(stitch_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Drawing contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Ignore any detected regions that are too small
        if w * h > 50:
            detected_defect += 1
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw the rectangle

    if (detected_defect == 0):
        print("No defect detected")
    else:
        # Display defect
        cv2.putText(output, "Defect Detected", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return output


    return []

