import cv2
import numpy as np
from PIL import Image

def detect_missing_finger(image):
    original_image = Image.open(image)
    original_image = np.array(original_image)

    img = cv2.resize(original_image, (500, 500))
    output = img.copy() # Prevent original image from being edited
    finger_count = 0

    # Glove Color Boundaries
    lower_blue = np.array([12, 0, 0])
    upper_blue = np.array([200, 255, 255])

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

    # Find Contour
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (Hand)
    max_contour = max(contours, key=cv2.contourArea)

    # Convex Hull
    convexHull = cv2.convexHull(max_contour, returnPoints=False)

    # Convexity Defects
    defects = cv2.convexityDefects(max_contour, convexHull)

    if defects is not None:
        for i in range(defects.shape[0]):

            # Convexity defect will return:
            # 1. Start of convexity defect
            # 2. End of convexity defect
            # 3. Middle point from the convexity defect (Far Index)
            # 4. Distance between the middle point and the convex hull
            
            start_index, end_index, far_index, distance = defects[i, 0]
            start = tuple(max_contour[start_index][0])
            end = tuple(max_contour[end_index][0])
            far = tuple(max_contour[far_index][0])

            # Calculate the distance between the farthest point and the convex hull
            # Euclidean distance
            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

            # Calculate angle using cosine rule
            angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c)) * 180 / np.pi

            # If angle between fingers is less than 90 degrees, then theres a finger
            if angle < 90:
                finger_count += 1

    if finger_count > 1:
        finger_count += 1

    if finger_count == 0 or finger_count < 5:
        text = "Missing Finger Detected: Detected only {}".format(finger_count)
        cv2.putText(output, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return output
    else:
        cv2.putText(img, "No Defect Detected", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return img