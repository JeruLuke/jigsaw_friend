import cv2
import numpy as np
from matplotlib import pyplot as plt
import auxcv as aux
import piecematchersift as pms
import dragandcrop
import trackbar as tb

CANVAS_PATH = './data/image.jpg'
PIECE_PATH = './data/piece.jpg'

pm = pms.PieceMatcher(PIECE_PATH, CANVAS_PATH)

cropped_piece = pm.crop_using_contour()
#cropped_piece = dragandcrop.crop_image(pm.piece)

piece_masked_background = pm.get_piece_masked_background(cropped_piece)

piece_masked_background = aux.image_to_gray(piece_masked_background.copy())
piece_masked_background = cv2.GaussianBlur(piece_masked_background, (7,7), 0)
piece_masked_background = cv2.GaussianBlur(piece_masked_background, (7,7), 2)

#compare_histograms_normal_and_masked(dragandcrop.crop_image(pm.board), cropped_piece)

#edges1 = tb.CannyTrackbar(piece_masked_background, "cannyTrackbar")
edges1 = cv2.Canny(piece_masked_background,0,68)
#edges2 = tb.CannyTrackbar(pm.board_gray, "cannyTrackbar")
edges2 = cv2.Canny(pm.board_gray, 165, 100)

_, contours, heirarchy = cv2.findContours(edges1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

aux.show_image("piece", cropped_piece, False)
aux.show_image("edges1", edges1, False)
aux.show_image("edges2", edges2, True)

cnts = sorted(contours, key = cv2.contourArea, reverse = True)
cv2.drawContours(edges1, cnts[1:], 0, (0,255,0), 3)
aux.show_image("contours", edges1, True)

edges1 = cv2.resize(edges1.copy(), None, edges1, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
aux.show_image("edges1smaller", edges1, True, True)

cv2.destroyAllWindows()
