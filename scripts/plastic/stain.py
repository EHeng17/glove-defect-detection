import cv2
import numpy as np
import math
from PIL import Image

def resize_image(image, width):
    inter = cv2.INTER_AREA
    (h, w) = image.shape[:2]

    # calculate the ratio of the width and construct the
    # dimensions
    r = width / float(w)
    dim = (width, int(h * r))

    # resize the image
    resized_img = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized_img

def process(img_c, c_value = None):
    img_gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

    # The image is blurred using Median Blurring to remove noise
    img_gray_blur = cv2.medianBlur(img_gray, 15)

    mean_value = np.mean(img_gray)
    
    _, th3 = cv2.threshold(img_gray_blur, mean_value, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    glove_contours = max(contours, key = cv2.contourArea)
    cv2.fillPoly(th3, pts=[glove_contours], color=(255, 255, 255))

    return th3, glove_contours

def identify_color(hsv_value):
    # Define the color ranges in HSV
    color_ranges = {
        'red': [(0, 50, 20), (10, 255, 255)],
        'blue': [(100, 50, 50), (130, 255, 255)],
        'yellow': [(25, 50, 20), (35, 255, 255)],
        'purple': [(125, 50, 20), (150, 255, 255)],
        'black': [(0, 0, 0), (180, 255, 50)],
        # ... add other colors as needed
    }

    # Determine the color
    for color_name, (lower_bound, upper_bound) in color_ranges.items():
        lower_bound = np.array(lower_bound, dtype="uint8")
        upper_bound = np.array(upper_bound, dtype="uint8")
        
        # Check if the HSV value is within the range
        # if cv2.inRange(np.array([[hsv_value]], dtype="uint8"), lower_bound, upper_bound):
        #     return color_name
        if hsv_value >= lower_bound[0] and hsv_value <= upper_bound[0]:
            return color_name

    return ""  # If the color does not match any range

def detect_stain(image):

    img = Image.open(image)
    img= np.array(img)

    # Copying image
    img_c = img.copy()

    img_c = resize_image(img_c, width=650)

    # Define HSV range for purple
    lower_purple_stain = np.array([110, 50, 50])
    upper_purple_stain = np.array([130, 255, 150])

    # Define HSV range for black
    lower_black_stain = np.array([0, 0, 0])
    upper_black_stain = np.array([180, 255, 50])

    stain_number = 0

    glove_mask, glove_contours = process(img_c)

    # Segmenting out the gloves
    segmented_glove = cv2.bitwise_and(img_c, img_c, mask=glove_mask)
    segmented_glove_hsv = cv2.cvtColor(segmented_glove, cv2.COLOR_BGR2HSV)

    purple_stain_mask = cv2.inRange(segmented_glove_hsv, lower_purple_stain, upper_purple_stain)
    black_stain_mask = cv2.inRange(segmented_glove_hsv, lower_black_stain, upper_black_stain)

    # Combine both stain mask
    stain_mask = cv2.bitwise_or(purple_stain_mask, black_stain_mask)

    # Find the intersection of the glove and stain mask
    stains_on_glove = cv2.bitwise_and(stain_mask, glove_mask)

    # Erode the mask to remove noise
    kernel = np.ones((2, 2), np.uint8)
    stains_on_glove = cv2.erode(stains_on_glove, kernel, iterations=1)

    contours, _ = cv2.findContours(stains_on_glove, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:
            continue
        
        stain_number += 1
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(segmented_glove, (x, y), (x + w, y + h), (0, 255, 0), 2)

        stain_region = segmented_glove[y:y+h, x:x+w]
        mean_color = cv2.mean(stain_region, mask=stains_on_glove[y:y+h, x:x+w])
        
        # # Convert mean color from BGR to HSV
        mean_color_hsv = cv2.cvtColor(np.uint8([[mean_color[:3]]]), cv2.COLOR_BGR2HSV)[0][0]
        mean_hue = mean_color_hsv[0]
        color_name = identify_color(mean_hue)
        
        cv2.putText(segmented_glove, f"Stain {stain_number} - {color_name}", (x-20, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if stain_number == 0:
        cv2.putText(segmented_glove, "No stains found", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return segmented_glove