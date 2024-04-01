import cv2
import numpy as np

def detect_tearing(img_path):

    #preprocessing
    img = cv2.imread(img_path)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([85, 111, 122])
    upper = np.array([103, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    blurred_frame = cv2.medianBlur(mask, 9)
    kernel = np.ones((3, 3), np.uint8)
    eroded_frame = cv2.erode(blurred_frame, kernel)

    # Find Defects
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(eroded_frame, contours, -1, (0, 255, 0), 2)

    # Find Defects that is within gloves
    internal_cnt = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] >= 0]

    if len(contours) > 0:
        blue_area = max(contours, key=cv2.contourArea)
        (xg, yg, wg, hg) = cv2.boundingRect(blue_area)
        cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (255, 0, 0), 1)

        # Label the glove
        frame = cv2.putText(frame, 'Glove', (xg, yg - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Find defect
        if len(internal_cnt) > 0:
            for i in internal_cnt:
                area = cv2.contourArea(i)
                if area > 40:
                    xd, yd, wd, hd = cv2.boundingRect(i)
                    cv2.rectangle(frame, (xd, yd), (xd + wd, yd + hd), (0, 0, 255), 1)
                    if area > 400:
                        frame = cv2.putText(frame, 'Tearing', (xd, yd - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                             0.5, (0, 0, 255), 1, cv2.LINE_AA)

    return frame
