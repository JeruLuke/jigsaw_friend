import argparse
import cv2
import auxcv as aux

refPt = []

def drag_crop(event, x, y, flags, param):
    global refPt

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))

def crop_image(image):
    global refPt

    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", drag_crop)

    while True:
        if len(refPt) < 2:
            cv2.imshow("image", image)
            key = cv2.waitKey(1) & 0xFF
        else:
            cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
            cv2.imshow("image", image)
            key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            image = clone.copy()
            refPt = []

        elif key == ord("c"):
            break

    if len(refPt) == 2:
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

    return roi
