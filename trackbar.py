"""
usage: threshold_custom = tb.SimpleTrackbar(img, "ImgThresh")
"""

import cv2
import numpy as np

def empty_function():
    pass

def SimpleTrackbar(img, win_name):
    trackbar_name = win_name + "Trackbar"

    cv2.namedWindow(win_name)
    cv2.createTrackbar(trackbar_name, win_name, 0, 255, empty_function)

    while True:
        trackbar_pos = cv2.getTrackbarPos(trackbar_name, win_name)
        _, img_th = cv2.threshold(img, trackbar_pos, 255, cv2.THRESH_BINARY)
        cv2.imshow(win_name, img_th)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            break

    cv2.destroyAllWindows()
    return trackbar_pos