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


def detect_tear(image):
    img = Image.open(image)
    img= np.array(img)

    # Copying image
    img_c = img.copy()

    img_c = resize_image(img_c, width=650)

    
    tear_num = 0

    # Define HSV range for black
    lower_black_tear = np.array([0, 0, 0])
    upper_black_tear = np.array([180, 255, 50])

    glove_mask, glove_contours = process(img_c)

    # Segmenting out the gloves
    segmented_glove = cv2.bitwise_and(img_c, img_c, mask=glove_mask)
    segmented_glove_gray = cv2.cvtColor(segmented_glove, cv2.COLOR_BGR2GRAY)
    segmented_glove_hsv = cv2.cvtColor(segmented_glove, cv2.COLOR_BGR2HSV)

    black_tear_mask = cv2.inRange(segmented_glove_hsv, lower_black_tear, upper_black_tear)

    # Find the intersection of the glove and stain mask
    tear_on_glove = cv2.bitwise_and(black_tear_mask, glove_mask)

    contours, _ = cv2.findContours(tear_on_glove, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            continue

        tear_num += 1
        tear_category = ""
        # Identify the tear based on the area and categorize into small, medium, and large
        if area > 1000:
            tear_category = "Large Tear"
        elif area > 500:
            tear_category = "Medium Tear"
        elif area > 100:
            tear_category = "Small Tear"
        else:
            tear_category = "Tiny Tear"

        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(segmented_glove, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv2.putText(segmented_glove, f"Tear {tear_num} - {tear_category}", (x - 40, y - 25 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return segmented_glove