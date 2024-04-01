import cv2
import numpy as np

def detect_chemical_contamination(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=cv2.contourArea)[-1]
    mask = np.zeros(gray.shape, dtype="uint8")
    masked_red = cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)
    final_image = cv2.bitwise_and(image, image, mask=masked_red)

    # Chemical Contamination detection
    gray_img = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_image = cv2.bitwise_not(binary_image)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(final_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(final_image, "Chemical Contamination", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return final_image
