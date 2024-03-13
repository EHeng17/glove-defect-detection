import cv2
import numpy as np
from PIL import Image

def detect_tearing(image):
    original_image = Image.open(image)
    original_image = np.array(original_image)

    img = cv2.resize(original_image, (500, 500))

    output = img.copy() # Prevent original image from being edited
    tear = False

    # Glove Color Boundaries
    lower_blue = np.array([0, 0, 0])
    upper_blue = np.array([255, 138, 212])

    # Color Segmentation
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    mask_result = cv2.bitwise_and(output, output, mask=mask)

    # Convert mask to grayscale
    grayscale_mask = cv2.cvtColor(mask_result, cv2.COLOR_BGR2GRAY)

    # Remove Noise (Median Filter)
    median_filtered_img = cv2.medianBlur(grayscale_mask, 5)

    # Thresholding
    # Used threshold as it is easier to detect edges more clearly
    # Without thresholding, the edge detection might capture patterns of the glove as well
    _, threshold = cv2.threshold(median_filtered_img, 0, 255, cv2.THRESH_BINARY)  

    # Edge Detection
    # Used 0 as low threshold and 255 as high threshold because its binary image
    edges = cv2.Canny(threshold, 0, 255)

    # Finding contours
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter Contour based on perimeter and shape
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)

        # If small perimeter, ignore as it could be unremoved noises
        if perimeter > 200 and perimeter < 300:

            # Contour approximation
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approximations = cv2.approxPolyDP(contour, epsilon, True)

            # Shapes can be used to determine the tear. Those with lesser approximations are usually lines or patterns from gloves
            if len(approximations) >= 7:
                # Detect Defect
                tear = True

                # Draw Rectangle around tear
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if tear:
        cv2.putText(output, "Defect Detected", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if (tear):
        # Display defect
        cv2.putText(output, "Defect Detected", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return output

    else:
        # No Defects
        cv2.putText(img, "No Defect Detected", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return img