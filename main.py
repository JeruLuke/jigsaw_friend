import cv2
import numpy as np
from matplotlib import pyplot as plt
import auxcv as aux
import piecematchersift as pms
import drag_and_crop as dac

CANVAS_PATH = './data/image.jpg'
PIECE_PATH = './data/piece.jpg'

pm = pms.PieceMatcher(PIECE_PATH, CANVAS_PATH)

cropped_piece = pm.crop_using_contour()
#cropped_piece = dac.crop_image(pm.piece)
#aux.show_image("cropped", cropped_piece)
cropped_piece = aux.image_to_gray(cropped_piece)
mask = pm.get_mask_sobel(cropped_piece)
aux.show_image("mask", mask, True, True)

cv2.destroyAllWindows()
