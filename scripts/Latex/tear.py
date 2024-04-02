import cv2
import numpy as np
from PIL import Image

def latex_detect_tear(image):
    # Read the image
    original_image = Image.open(image)
    original_image = np.array(original_image)

    original_image = cv2.resize(original_image, (500, 500), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    
    output = original_image.copy()
    
    # Preprocessing
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([85, 111, 122])
    upper = np.array([103, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    blurred_frame = cv2.medianBlur(mask, 9)
    kernel = np.ones((3, 3), np.uint8)
    eroded_frame = cv2.erode(blurred_frame, kernel)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the frame
    cv2.drawContours(eroded_frame, contours, -1, (0, 255, 0), 2)

    # Filter out internal contours (defects within gloves)
    internal_cnt = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] >= 0]

    # Iterate over the internal contours
    if len(contours) > 0:
        blue_area = max(contours, key=cv2.contourArea)
        (xg, yg, wg, hg) = cv2.boundingRect(blue_area)

        if len(internal_cnt) > 0:
            for i in internal_cnt:
                area = cv2.contourArea(i)
                if area > 40:
                    xd, yd, wd, hd = cv2.boundingRect(i)
                    cv2.rectangle(output, (xd, yd), (xd + wd, yd + hd), (0, 0, 255), 1)

                    if area > 400:
                        output = cv2.putText(output, 'Tearing', (xd, yd - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                             0.5, (0, 0, 255), 1, cv2.LINE_AA)

    return output
