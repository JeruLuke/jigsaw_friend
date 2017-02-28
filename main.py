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
cropped_piece_gray = aux.image_to_gray(cropped_piece.copy())
mask = pm.get_mask_sobel(cropped_piece_gray)
#aux.show_image("mask", mask, True, True)
piece_masked_background = cv2.bitwise_and(cropped_piece, cropped_piece, mask = mask)
#aux.show_image("piece", piece_masked_background, True)

cropped_board = dac.crop_image(pm.board)
#aux.show_image("board", cropped_board, True)

piece_masked_background = aux.image_to_gray(piece_masked_background.copy())
cropped_board = aux.image_to_gray(cropped_board.copy())
piece_masked_background = cv2.GaussianBlur(piece_masked_background, (5,5), 0)

fig = plt.figure()
ax1 = fig.add_subplot(211)
plt.hist(cropped_board.ravel(), 256, [0,256])
ax2 = fig.add_subplot(212)
plt.hist(piece_masked_background[np.nonzero(piece_masked_background)].ravel(), 256, [0,256])
#plt.show()

#aux.show_image("piece", piece_masked_background, False)
#aux.show_image("board", cropped_board, True)

edges = cv2.Canny(piece_masked_background,30,200)
aux.show_image("edges", edges, True)

cv2.destroyAllWindows()
