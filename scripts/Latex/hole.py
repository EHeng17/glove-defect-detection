import cv2
import numpy as np
from PIL import Image

def latex_detect_hole(image):

    original_image = Image.open(image)
    original_image = np.array(original_image)
    original_image = cv2.resize(original_image, (500, 500))
    
    image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    
    output = original_image.copy()

    #preprocessing
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50]) 
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_regions = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(blue_regions, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)

    # Detect Defects and Draw Rectangles
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(binary, (x, y), (x + w, y + h), (255, 255, 255), cv2.FILLED)

    # Plotting Defect
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(output, "Hole", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return output
