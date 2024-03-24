import cv2
import numpy as np
import matplotlib.pyplot as plt

class StainDetector:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)

    def preprocess(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(contours, key=cv2.contourArea)[-1]
        mask = np.zeros(gray.shape, dtype="uint8")
        masked_red = cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)
        self.final_image = cv2.bitwise_and(self.image, self.image, mask=masked_red)

    def detect_stains(self):
        gray_img = cv2.cvtColor(self.final_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_image = cv2.bitwise_not(binary_image)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(self.final_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def display_result(self):
        plt.imshow(cv2.cvtColor(self.final_image, cv2.COLOR_BGR2RGB))
        plt.show()

# Usage
image_path = './defect images/Stain/image (4).jpg'
stain_detector = StainDetector(image_path)
stain_detector.preprocess()
stain_detector.detect_stains()
stain_detector.display_result()
