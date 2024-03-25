import cv2
import numpy as np
from PIL import Image

def detect_tearing_poly(image):
    original_image = Image.open(image)
    original_image = np.array(original_image)

    img = cv2.resize(original_image, (500, 500))

    output = img.copy() # Prevent original image from being edited
    tear = False

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Glove Color Boundaries
    # Define lower and upper bounds for blue color
    lower_blue = np.array([0, 120, 100])
    upper_blue = np.array([40, 255, 255])

    # Define lower and upper bounds for white color
    lower_white = np.array([0, 0, 100])
    upper_white = np.array([180, 20, 255])

    # Create masks for blue and white colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Combine masks using bitwise OR operation
    combined_mask = cv2.bitwise_or(mask_blue, mask_white)

    # Apply the combined mask to the original image
    mask_result = cv2.bitwise_and(img, img, mask=combined_mask)

    # Convert mask to grayscale
    grayscale_mask = cv2.cvtColor(mask_result, cv2.COLOR_BGR2GRAY)
    
    # Remove Noise (Median Filter)
    median_filtered_img = cv2.medianBlur(grayscale_mask, 5)

    # Thresholding
    # Used threshold as it is easier to detect edges more clearly
    # Without thresholding, the edge detection might capture patterns of the glove as well
    _, threshold = cv2.threshold(median_filtered_img, 0, 255, cv2.THRESH_BINARY)  

    # Dilation (Link all the dots in the glove)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilate = cv2.morphologyEx(threshold, cv2.MORPH_DILATE, dilate_kernel)
    
    # Finding contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
    # Iterate over contours
    for contour in contours:
        # Calculate the area of each contour
        area = cv2.contourArea(contour)

        # Check if the contour represents a smaller defect
        if area > 250 and area < 5000:  # threshold_area is your defined threshold for small defects
            # Draw the contour on the output image
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
            tear = True

    if tear:
        cv2.putText(output, "Defect Detected", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return output
    else:
        cv2.putText(img, "No Defect Detected", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return img
    