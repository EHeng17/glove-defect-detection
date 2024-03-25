import cv2
import numpy as np
from PIL import Image

def detect_missing_dot(image):
    original_image = Image.open(image)
    original_image = np.array(original_image)

    img = cv2.resize(original_image, (500, 500))
    output = img.copy() # Prevent original image from being edited
    total_missing_dot = 0

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for white color
    lower_white = np.array([0, 0, 100]) 
    upper_white = np.array([180, 20, 255])

    # Create masks for white colors
    mask_white = cv2.inRange(hsv, lower_white, upper_white) 

    # Bitwise-AND mask and original image
    mask_result = cv2.bitwise_and(img, img, mask=mask_white)

    # Convert mask to grayscale
    grayscale_mask = cv2.cvtColor(mask_result, cv2.COLOR_BGR2GRAY)

    # Remove Noise (Median Filter)
    median_filtered_img = cv2.medianBlur(grayscale_mask, 3)

    _, threshold = cv2.threshold(median_filtered_img, 0, 255, cv2.THRESH_BINARY)  

    # Erosion (Link all the missing dots in the glove)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    erode = cv2.morphologyEx(threshold, cv2.MORPH_ERODE, erode_kernel)

    # Finding contours
    contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter Contour based on area
    for contour in contours:
        # Get area of contour
        area = cv2.contourArea(contour)

        # Ignore any detected regions that are too small and too big
        if area >= 200 and area <= 5000:
            # assuming 50px is the average size of a dot
            missing_dot = int(area/50)
            total_missing_dot += missing_dot

            # Draw rectangle around defect
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw the rectangle
            
            # Write the number of missing dots inside the rectangle
            text = f"{missing_dot}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x + int((w - text_size[0]) / 2)  # Center the text horizontally
            text_y = y + int((h + text_size[1]) / 2)  # Center the text vertically
            cv2.putText(output, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Write the total missing dots if more than 1
    if total_missing_dot == 0:
        cv2.putText(img, "No Defect Detected", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return img
    else:
        cv2.putText(output, f"Defect Detected; Missing {total_missing_dot} dots", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return output