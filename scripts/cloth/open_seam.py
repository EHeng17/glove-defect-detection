import cv2
import numpy as np
from PIL import Image

def detect_open_seam(image):
    original_image = Image.open(image)
    original_image = np.array(original_image)

    img = cv2.resize(original_image, (500, 500))

    output = img.copy() # Prevent original image from being edited
    detected_defect = 0

    # Glove Color Boundaries
    lower_blue = np.array([70, 0, 0])
    upper_blue = np.array([255, 255, 255])

    # Color Segmentation
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    mask_result = cv2.bitwise_and(output, output, mask=mask)

    # Convert mask to grayscale
    grayscale_mask = cv2.cvtColor(mask_result, cv2.COLOR_BGR2GRAY)

    # Remove Noise (Median Filter)
    median_filtered_img = cv2.medianBlur(grayscale_mask, 5)

    # Thresholding
    _, threshold = cv2.threshold(median_filtered_img, 0, 255, cv2.THRESH_BINARY)  

    # Dilation (Fill in the holes in the glove if any)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate = cv2.morphologyEx(threshold, cv2.MORPH_DILATE, dilate_kernel)

    # Opening (To get seams)
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    opening = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, opening_kernel)

    # Subtracting opened image with original image to find any difference between the images
    seams = cv2.subtract(dilate, opening)

    # Removing small dots that are not removed by opening (Using small kernel to prevent the detected stitch from eroding too much)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) # Used for removing unremoved noises
    opening_seams = cv2.morphologyEx(seams, cv2.MORPH_OPEN, kernel_small)

    # Finding contours
    contours, hierarchy = cv2.findContours(opening_seams, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Drawing contours
    for contour in contours:
        # Get area of contour
        area = cv2.contourArea(contour)

        # Ignore any detected regions that are too small
        if area >= 50:
            detected_defect += 1

            # Draw rectangle around defect
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw the rectangle

    if (detected_defect == 0):
        cv2.putText(img, "No Defect Detected", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return img
    else:
        # Display defect
        cv2.putText(output, "Defect Detected", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return output

