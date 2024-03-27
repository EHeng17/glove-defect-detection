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

    contours, _ = cv2.findContours(th3, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # Use for Method 1, otherwise empty
    glove_contour_list = []

    # Method 1 - For size defects
    for contour in contours: 
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 11:
            # Fill contours to create a mask for the gloves
            glove_contour_list.append(contour)
            cv2.fillPoly(th3, pts=[contour], color=(255, 255, 255))

    # Method 2 - for stain and tear defects
    glove_contours = max(contours, key = cv2.contourArea)
    
    return th3, glove_contour_list, glove_contours

def get_centroid(contour):
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return tuple((cx, cy))

# Vertial Line
def draw_line_from_center_vertical(contour, centroid, color, segmented_glove):
    highest_point = 0
    lowest_point = 0

    # Calculate the highest and lowest point within the glove_contour
    for c in contour:
        x, y = c[0]
        # Check that they are in the same x-coordinate
        if x != centroid[0]:
            continue
        if y < centroid[1]:
            if highest_point == 0:
                highest_point = y
            if y < highest_point:
                highest_point = y
        elif y > centroid[1]:
            if lowest_point == 0:
                lowest_point = y
            if y > lowest_point:
                lowest_point = y

    dist = math.dist((centroid[0], highest_point), (centroid[0], lowest_point))
    cv2.line(segmented_glove, (centroid[0], highest_point), (centroid[0], lowest_point), color, 2)
    cv2.circle(segmented_glove, centroid, 15, (0, 255, 0), -1)
    cv2.putText(segmented_glove, f"Height {str(dist)}", (centroid[0] - 50, centroid[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    return dist, segmented_glove

# Horizontal Line
def draw_line_from_center_horizontal(contour, centroid, color, segmented_glove):
    left_point = 0
    right_point = 0

    # Calculate the highest and lowest point within the glove_contour
    for c in contour:
        x, y = c[0]
        # Check that they are in the same y-coordinate
        if y != centroid[1]:
            continue
        if x < centroid[0]:
            if left_point == 0:
                left_point = x
            if x < left_point:
                left_point = x
        elif x > centroid[0]:
            if right_point == 0:
                right_point = x
            if x > right_point:
                right_point = x

    dist = math.dist((left_point, centroid[1]), (right_point, centroid[1]))
    cv2.line(segmented_glove, (left_point, centroid[1]), (right_point, centroid[1]), color, 2)
    cv2.circle(segmented_glove, centroid, 15, (0, 255, 0), -1)
    cv2.putText(segmented_glove, f"Length {str(dist)}", (centroid[0] + 25, centroid[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    return dist, segmented_glove


def detect_inconsistent_size(image):
    
    img = Image.open(image)
    img= np.array(img)

    # Copying image
    img_c = img.copy()

    img_c = resize_image(img_c, width=650)

    glove_mask, glove_contour_list, glove_contours = process(img_c)

    # Segmenting out both gloves
    segmented_glove = cv2.bitwise_and(img_c, img_c, mask=glove_mask)

    # Splitting the glove contours into left and right glove
    right_glove_contour = glove_contour_list[0]
    left_glove_contour = glove_contour_list[1]

    cv2.drawContours(segmented_glove, right_glove_contour, -1, (255, 0, 0), 3)
    cv2.drawContours(segmented_glove, left_glove_contour, -1, (0, 255, 0), 3)

    right_glove_centroid = get_centroid(right_glove_contour)
    left_glove_centroid = get_centroid(left_glove_contour)

    right_glove_height, segmented_glove = draw_line_from_center_vertical(right_glove_contour, right_glove_centroid, [255, 0, 0], segmented_glove=segmented_glove)
    left_glove_height, segmented_glove= draw_line_from_center_vertical(left_glove_contour, left_glove_centroid, [0, 255, 0], segmented_glove=segmented_glove)

    right_glove_length, segmented_glove = draw_line_from_center_horizontal(right_glove_contour, right_glove_centroid, [255, 0, 0], segmented_glove=segmented_glove)
    left_glove_length, segmented_glove = draw_line_from_center_horizontal(left_glove_contour, left_glove_centroid, [0, 255, 0], segmented_glove=segmented_glove)

    if right_glove_height == left_glove_height:
        cv2.putText(segmented_glove, "Consistent Size", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(segmented_glove, "Inconsistent Size", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return segmented_glove