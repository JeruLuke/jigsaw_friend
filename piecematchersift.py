import cv2
import numpy as np
import auxcv as aux
import trackbar as tb

class PieceMatcher(object):
    def __init__(self, piece_path, board_path):
        self.piece = self.make_smaller(cv2.imread(piece_path))
        self.board = self.make_smaller(cv2.imread(board_path))
        self.piece_gray = cv2.cvtColor(self.piece, cv2.COLOR_BGR2GRAY)
        self.board_gray = cv2.cvtColor(self.board, cv2.COLOR_BGR2GRAY)

    def preprocess(self, threshold_low, threshold_high):
        """ Preprocess image for contour identification """
        piece_processed = self.piece_gray.copy()
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
        piece_processed = clahe.apply(piece_processed)
        _, piece_processed = cv2.threshold(piece_processed, threshold_low, threshold_high, 0)
        return piece_processed

    # Pre-process the piece
    def contour_for_crop(self, threshold_low=100, threshold_high=200):
        """Identify the contour around the piece"""
        piece = self.preprocess(threshold_low, threshold_high)
        _, contours, _ = cv2.findContours(piece, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
        big_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                big_contours.append(cnt)

        return big_contours[1]

    def get_bounding_rect(self, cnt):
        """Return the bounding rectangle given a contour"""
        x, y, w, h = cv2.boundingRect(cnt)
        return x, y, w, h

    def crop_using_contour(self):
        cnt = self.contour_for_crop()
        x, y, w, h = self.get_bounding_rect(cnt)
        offset = 10
        x -= offset
        y -= offset
        w += offset*2
        h += offset*2
        return self.piece[y:y+h, x:x+w]

    def find_contour(self, image, threshold_low=100, threshold_high=200):
        piece = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4,4))
        #piece = clahe.apply(piece)
        #threshold_custom = tb.SimpleTrackbar(piece, "PieceThresh")
        #print("threshold_custom", threshold_custom)
        aux.show_hist(piece)
        threshold_custom = 140
        ret, thresh = cv2.threshold(piece, threshold_custom, 255, 0)
        _, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(piece, contours, -1, (0,255,0), 3)
        aux.show_image("crpcnt", piece)

    def get_sobel(self, img, size = -1):
        sobelx64f = cv2.Sobel(img,cv2.CV_64F,2,0,size)
        abs_sobel64f = np.absolute(sobelx64f)
        return np.uint8(abs_sobel64f)

    def get_mask_sobel(self, img):
        sobel = self.get_sobel(img)
        sobel = aux.close_image(sobel, (7,7))
        sobel = aux.open_image(sobel, (9,9))
        #threshold_custom = tb.SimpleTrackbar(sobel, "SobelThresh")
        th, sobel = cv2.threshold(sobel, 49, 255, cv2.THRESH_BINARY_INV)
        sobel = aux.open_image(sobel, (9,9))
        sobel = aux.smoother_edges(sobel, (15,15), (5,5), 1.7, -0.3)
        #threshold_custom = tb.SimpleTrackbar(sobel, "SobelThresh")
        th, sobel = cv2.threshold(sobel, 40, 255, cv2.THRESH_BINARY_INV)
        return sobel

    @staticmethod
    def make_smaller(img):
        return cv2.resize(img, None, fx=0.2, fy=0.2)
