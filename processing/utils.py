import numpy as np
import cv2 as cv
import os
import time
import matplotlib.pyplot as plt
from skimage.filters import try_all_threshold, threshold_li, threshold_minimum, threshold_yen
from skimage import measure

def empty_callback(value):
    pass
cv.namedWindow('image')
cv.createTrackbar('thr1', 'image', 9, 255, empty_callback)
cv.createTrackbar('thr2', 'image', 200, 255, empty_callback)
cv.createTrackbar('kernel', 'image', 9, 255, empty_callback)
def perform_processing(image: np.ndarray) -> str:

    start = time.time()
    # resizing image to smaller
    image = cv.resize(image, (0, 0), fx=0.3, fy=0.3)
    img_gray = image.copy()
    img_gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)
    row, col = img_gray.shape[:]


    while True:
        thr1 = cv.getTrackbarPos('thr1', 'image')
        thr2 = cv.getTrackbarPos('thr2', 'image')
        thresh = cv.getTrackbarPos('kernel', 'image')

        # filtering image
        #img_gray = cv.GaussianBlur(img_gray, (9, 9), 0)
        img_gray = cv.medianBlur(img_gray, 11)
        img_gray2 = cv.bilateralFilter(img_gray, 9, 49, 49)

        hsv_frame = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        print(hsv_frame)
        edge = cv.Canny(img_gray2, 10, 200)
        cv.imshow('image_2', edge)
        # thresholding and opening
        #th3 = cv.adaptiveThreshold(edge, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
         #                         cv.THRESH_BINARY, 99, 46)
        #_, th3 = cv.threshold(img_gray2, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        kernel = (21, 21)
        #open = cv.morphologyEx(edge, cv.MORPH_OPEN, kernel)

        # # Finding Countours

        cont, hier = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        #cv.drawContours(image, cont, -1, (0, 200, 0), 2)
        perfect = []
        min_cont = []       #pierwszy narożnik x z lewej
        max_cont = []       #ostatni narożnik x z prawej
        letter_num = []     #numer Bboxa
        area = list()
        min_y, max_y, min_yr, max_yr = 0, 0, 0, 0
        for contour in cont:                                                                    #metoda na same litery i bounding boxy
            x, y, w, h = cv.boundingRect(contour)
            if h > w and w*h > 1500 and cv.contourArea(contour) > 600:
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 250), 6)
                perfect.append(contour)
            #cv.imshow('image_2', img_gray2)

                min_x_cont = np.min(contour[:, :, 0])
                #max_x_cont = np.max(contour[:, :, 0])
                max_x_cont = x+w
                min_cont.append(min_x_cont)
                max_cont.append(max_x_cont)
                min_cont = sorted(min_cont)
                max_cont = sorted(max_cont, reverse=True)
                print(f'Min cont {min_cont}, Max cont {max_cont}')
                if x == min_cont[0]:
                    min_y = y
                    max_y = y+h
                if x+w == max_cont[0]:
                    min_yr = y
                    max_yr = y+h
            # width = max_x_cont - min_x_cont
            # height = max_y_cont - min_y_cont
            # if 100000> area[-1] > 10000 and col -50 > (max_x_cont-min_x_cont) > col/3 and\        #metoda wykrywająca całą tablicę
            #                                     row-100 > (max_y_cont-min_y_cont) > row/10 and\
            #                                     (width/height) < 6:
            #     cv.drawContours(image, contour, -1, (0, 200, 0), 3)
            #     perfect.append(contour)
            #print(contour)
            #print(f'Min: {min_y_cont}, Max: {max_x_cont}')
        area = sorted(area, reverse=True)
        print(area[:10])

        rect = np.zeros((4, 2), dtype=np.float32)

        min_x_tab = min_cont[0]-5
        max_x_tab = max_cont[0]+5
        min_y_tab = min_y-5
        max_y_tab = max_y+5
        min_yr_tab = min_yr-5
        max_yr_tab = max_yr+5
        print(min_yr_tab, max_yr_tab)
    # Obliczenie skrajnych punktów dla transformacji perspektywicznej
        rect[0] = np.float32([min_x_tab, min_y_tab])
        rect[1] = np.float32([max_x_tab, min_yr_tab])
        rect[2] = np.float32([max_x_tab, max_yr_tab])
        rect[3] = np.float32([min_x_tab, max_y_tab])
        width = max_x_tab - min_x_tab
        height = max_y_tab - min_y_tab
        print(f'Width {width}, Height {height}')
        dst_rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        M = cv.getPerspectiveTransform(rect, dst_rect)
        warped_image = cv.warpPerspective(image, M, (int(width), int(height)))
        #image = cv.resize(image, (0, 0), fx=0.2, fy=0.2)

        cv.imshow('Wyprostowana perspektywa', warped_image)
        cv.imshow('image', image)
        stop = time.time()
        print('Czas przetwarzania: ', stop - start)
        key = cv.waitKey(0)
        if key == 27:
            break
            cv.destroyAllWindows()

            exit()
        print(f'image.shape: {image.shape}')
        # TODO: add image processing here

    return 'PO15'