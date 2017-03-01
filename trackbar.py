"""
usage: threshold_custom = tb.SimpleTrackbar(img, "ImgThresh")
"""

import cv2
import numpy as np

def empty_function(*arg):
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

def CannyTrackbar(img, win_name):
    trackbar_name = win_name + "Trackbar"

    cv2.namedWindow(win_name)
    cv2.resizeWindow(win_name, 500,100)
    cv2.createTrackbar("first", win_name, 0, 255, empty_function)
    cv2.createTrackbar("second", win_name, 0, 255, empty_function)
    cv2.createTrackbar("third", win_name, 0, 255, empty_function)

    while True:
        trackbar_pos1 = cv2.getTrackbarPos("first", win_name)
        trackbar_pos2 = cv2.getTrackbarPos("second", win_name)
        trackbar_pos3 = cv2.getTrackbarPos("third", win_name)
        img_blurred = cv2.GaussianBlur(img.copy(), (7,7), 2)
        canny = cv2.Canny(img_blurred, trackbar_pos1, trackbar_pos2)
        cv2.imshow(win_name, canny)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            break

    cv2.destroyAllWindows()
    return canny