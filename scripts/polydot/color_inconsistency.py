import cv2
import numpy as np
from PIL import Image

def detect_color_inconsistency(image):
    original_image = Image.open(image)
    original_image = np.array(original_image)

    img = cv2.resize(original_image, (500, 500))
    output = img.copy() # Prevent original image from being edited
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for light blue color
    lower_light_blue = np.array([0, 120, 100])
    upper_light_blue = np.array([20, 255, 255])
    
    # Define lower and upper bounds for dark blue color
    lower_dark_blue = np.array([21, 120, 100])
    upper_dark_blue = np.array([40, 255, 255]) 

    # Create masks for blue colors
    mask_light_blue = cv2.inRange(hsv, lower_light_blue, upper_light_blue) 
    mask_dark_blue = cv2.inRange(hsv, lower_dark_blue, upper_dark_blue) 

    # Bitwise-AND light blue mask and original image
    mask_result_light = cv2.bitwise_and(img, img, mask=mask_light_blue)
    light_blue_count = cv2.countNonZero(mask_light_blue)

    # Bitwise-AND dark blue mask and original image
    mask_result_dark = cv2.bitwise_and(img, img, mask=mask_dark_blue)
    dark_blue_coount = cv2.countNonZero(mask_dark_blue)

    # No defect detected if there is only one color type in the dots
    if(light_blue_count == 0 or dark_blue_coount == 0):
        cv2.putText(img, "No Defect Detected", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return img
    
    # Highlight the dots with the least color on the output
    if(light_blue_count > dark_blue_coount):
        mask_result = mask_result_dark
    else:  
        mask_result = mask_result_light
    
    # Convert mask to grayscale
    grayscale_mask = cv2.cvtColor(mask_result, cv2.COLOR_BGR2GRAY)

    # Remove Noise (Median Filter)
    median_filtered_img = cv2.medianBlur(grayscale_mask, 3)

    _, threshold = cv2.threshold(median_filtered_img, 0, 255, cv2.THRESH_BINARY)  

    # Dilation (Link all the dots in the glove)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilate = cv2.morphologyEx(threshold, cv2.MORPH_DILATE, dilate_kernel)

    # Finding contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter Contour based on area
    for contour in contours:
        # Get area of contour
        area = cv2.contourArea(contour)
        
        # Ignore any detected regions that are too small and too big
        if area >= 50 and area <= 20000:
            # Draw rectangle around inconsistent color
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw the rectangle
            
    # Display defect
    cv2.putText(output, f"Defect Detected", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return output