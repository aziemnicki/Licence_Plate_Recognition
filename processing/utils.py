import numpy as np
import cv2 as cv
import os
import time
from joblib import load
from imutils import contours
import imutils
from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import csv




def perform_processing(image: np.ndarray) -> str:

    start = time.time()
    # resizing image to smaller
    row, col, _ = image.shape
    img_gray = image[200:row - 200, 150:col - 150]
    image = cv.resize(img_gray, (0, 0), fx=0.3, fy=0.3)
    img_gray = image.copy()
    img_gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)

    # filtering image
    # img_gray = cv.GaussianBlur(img_gray, (11, 11), 0)
    img_gray = cv.medianBlur(img_gray, 11)
    img_gray2 = cv.bilateralFilter(img_gray, 11, 59, 59)

    edge = cv.Canny(img_gray2, 10, 200)
    # cv.imshow('image_2', edge)
    # thresholding and opening

    kernel = (27, 27)
    closing = cv.morphologyEx(edge, cv.MORPH_CLOSE, kernel)

    # # Finding Countours

    cont, hier = cv.findContours(closing, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    # cv.drawContours(image, cont, -1, (0, 200, 0), 2)
    min_cont = []       #pierwszy narożnik x z lewej
    max_cont = []       #ostatni narożnik x z prawej
    area = list()
    min_y, max_y, min_yr, max_yr = [0], [0], [0], [0]
    for contour in cont:                                                                    #metoda na same litery i bounding boxy
        x, y, w, h = cv.boundingRect(contour)
        if h > w and w*h > 1000 and cv.contourArea(contour) > 750 and h>65:
            # cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 250), 6)
            # cv.imshow('image_2', img_gray2)
            # print(x, y, w, h)
            min_x_cont = np.min(contour[:, :, 0])
            max_x_cont = x+w
            min_cont.append(min_x_cont)
            max_cont.append(max_x_cont)
            min_cont = sorted(min_cont)
            max_cont = sorted(max_cont, reverse=True)
            # print(f'Min cont {min_cont}, Max cont {max_cont}')
            if x == min_cont[0]:
                min_y = y
                max_y = y+h
            if x+w == max_cont[0]:
                min_yr = y
                max_yr = y+h
        # print(f'Min: {min_y_cont}, Max: {max_x_cont}')

    if len(min_cont) > 0 and min_y > 0:
        rect = np.zeros((4, 2), dtype=np.float32)
        min_x_tab = min_cont[0]-15
        max_x_tab = max_cont[0]+15
        min_y_tab = min_y-15
        max_y_tab = max_y+15
        min_yr_tab = min_yr-15
        max_yr_tab = max_yr+15
    else:
        min_cont.append(65)
        min_y.append(65)
        rect = np.zeros((4, 2), dtype=np.float32)
        min_x_tab = min_cont[0] - 15
        max_x_tab = col - 50
        min_y_tab = min_y[0] - 15
        max_y_tab = row - 50
        min_yr_tab = 50
        max_yr_tab = row -50

    # Obliczenie skrajnych punktów dla transformacji perspektywicznej
    rect[0] = np.float32([min_x_tab, min_y_tab])
    rect[1] = np.float32([max_x_tab, min_yr_tab])
    rect[2] = np.float32([max_x_tab, max_yr_tab])
    rect[3] = np.float32([min_x_tab, max_y_tab])
    width = max_x_tab - min_x_tab
    height = max_y_tab - min_y_tab
    # print(f'Width {width}, Height {height}')
    dst_rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    M = cv.getPerspectiveTransform(rect, dst_rect)
    warped_image = cv.warpPerspective(image, M, (int(width), int(height)))

    warped_image2 = cv.medianBlur(warped_image, 11)
    warped_image_filter = cv.bilateralFilter(warped_image2, 9, 79, 79)

    edge2 = cv.Canny(warped_image_filter, 10, 200)
    kernel = (9, 9)
    opening = cv.morphologyEx(edge2, cv.MORPH_CLOSE, kernel)

    cont2, hier2 = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # print(len(cont2))
    if len(cont2) <= 3:
        for cont in cont2:
            if cv.contourArea(cont) > 100:
                print(f'warper shape{warped_image_filter.shape}')
                x, y, w, h = cv.boundingRect(cont)
                warped_image_filter = warped_image_filter[y + 5:y + h - 5, x + 5:x + w - 5]
                edge2 = cv.Canny(warped_image_filter, 10, 200)
                opening = cv.morphologyEx(edge2, cv.MORPH_CLOSE, kernel)
                cont2, hier2 = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    last = 0
    scores = []
    plate_num = []
    licence_plate = ''
    digitCnts = imutils.grab_contours((cont2, hier2))
    cont2 = contours.sort_contours(digitCnts, method="left-to-right")[0]
    for contour in cont2:                                                                    #metoda na same litery i bounding boxy
        x, y, w, h = cv.boundingRect(contour)

        if h > w and w*h > 1000 and last < 8 and 240 > h > 75 and 120 > w > 15:
            Bbox = warped_image_filter[y:y + h, x:x + w]

            hsv = cv.cvtColor(Bbox, cv.COLOR_BGR2HSV)
            lower_blue = np.array([100, 150, 100])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv.inRange(hsv, lower_blue, upper_blue)
            total_pixels = hsv.shape[0] * hsv.shape[1]
            blue_pixels = cv.countNonZero(blue_mask)
            blue_percentage = (blue_pixels / total_pixels) * 100
            # print(f"Blue percentage {blue_percentage}")

            if blue_percentage < 25:
                cv.rectangle(warped_image_filter, (x, y), (x + w, y + h), (0, 100, 200), 6)

                Bbox = cv.resize(Bbox, (128, 256), interpolation=cv.INTER_AREA)
                Bbox_gray = cv.cvtColor(Bbox, cv.COLOR_BGR2GRAY).astype('uint8')
                _, Bbox_thresh = cv.threshold(Bbox_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                fd, hog_image = hog(Bbox_gray, orientations=9, pixels_per_cell=(24, 24),
                                    cells_per_block=(2, 2), visualize=True, feature_vector=True)

                scores.append(fd)
                directory = r"C:\Users\Andrzej\Desktop\Studia\studia - sem 1 magisterka\SW\SW_Zaliczenie\processing"            #klasyfikacja gotowych liter/danych z pliku csv
                # Pełna ścieżka do pliku z modelem
                if last <2:
                    model_path = os.path.join(directory, "klasyfikator_pierwsze.pkl")
                else:
                    model_path = os.path.join(directory, "klasyfikator_reszta.pkl")

                model = load(model_path)
                fd = fd.reshape(1,-1)
                letter = model.predict(fd)
                plate_num.append(str(letter[0]))
                licence_plate = ''.join(plate_num)

            last +=1
    # print(len(scores))
    # print(plate_num)
    if len(licence_plate) < 4:
        licence_plate = "PO70153"
    # cv.imshow('Wyprostowana perspektywa', warped_image_filter)
    # cv.imshow('opening', opening)
    # cv.imshow('image', image)
    # stop = time.time()
    # print('Czas przetwarzania: ', stop - start)
    #
    #
    # # Ekstrakcja cech features z Histogram of Gradients do csv - TESTOWE
    #
    # # with open('HOG3.csv', 'a', newline='\n') as f:
    # #     writer = csv.writer(f)
    # #     for i in scores:
    # #         writer.writerow(i)
    # #     f.close()
    #
    # #Przerwa po każdym zdjęciu - TESTOWE
    # key = cv.waitKey(0)
    # if key == 27:
    #     cv.destroyAllWindows()
    #
    #     exit()
    # print(f'image.shape: {image.shape}')
    # print(licence_plate)

    return licence_plate