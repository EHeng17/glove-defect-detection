import cv2
import numpy as np

class TearingGloves:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.fixed_size = (500, 500)
        self.frame = cv2.resize(self.img, self.fixed_size, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

    def detect(self):
        hsv_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        # Mask for detecting glove
        lower = np.array([85, 111, 122])
        upper = np.array([103, 255, 255])
        mask = cv2.inRange(hsv_frame, lower, upper)

        # Apply median filtering
        blurred_frame = cv2.medianBlur(mask, 9)

        # Define the structuring element (kernel) for erosion
        kernel = np.ones((3, 3), np.uint8)

        # Perform erosion
        eroded_frame = cv2.erode(blurred_frame, kernel)

        # Find Contours
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(eroded_frame, contours, -1, (0, 255, 0), 2)

        # Detect the defect within the glove
        internal_cnt = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] >= 0]

        if len(contours) > 0:
            blue_area = max(contours, key=cv2.contourArea)
            (xg, yg, wg, hg) = cv2.boundingRect(blue_area)

            # Draw rectangle for glove
            cv2.rectangle(self.frame, (xg, yg), (xg + wg, yg + hg), (255, 0, 0), 1)

            # Label the glove
            self.frame = cv2.putText(self.frame, 'Glove', (xg, yg - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                     0.5, (255, 0, 0), 1, cv2.LINE_AA)

            # Find defect
            if len(internal_cnt) > 0:
                for i in internal_cnt:
                    # Check defect size
                    area = cv2.contourArea(i)
                    if area > 40:
                        xd, yd, wd, hd = cv2.boundingRect(i)
                        # Draw rectangle for defect
                        cv2.rectangle(self.frame, (xd, yd), (xd + wd, yd + hd), (0, 0, 255), 1)

                        # Label the defect
                        if area > 400:
                            # Defect Type: Tearing
                            self.frame = cv2.putText(self.frame, 'Tearing', (xd, yd - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                                     0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow('Result', self.frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    # Path to the input image
    image_path = "./defect images/Tear/tear_4.jpg"
    detector = TearingGloves(image_path)
    detector.detect()

if __name__ == "__main__":
    main()
