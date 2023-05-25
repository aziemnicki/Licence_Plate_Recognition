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




def perform_processing(image: np.ndarray) -> str:

    start = time.time()
    # resizing image to smaller
    image = cv.resize(image, (0, 0), fx=0.3, fy=0.3)
    img_gray = image.copy()
    img_gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)

    # filtering image
    #img_gray = cv.GaussianBlur(img_gray, (11, 11), 0)
    img_gray = cv.medianBlur(img_gray, 11)
    img_gray2 = cv.bilateralFilter(img_gray, 11, 59, 59)

    edge = cv.Canny(img_gray2, 10, 200)
    #cv.imshow('image_2', edge)
    # thresholding and opening
    #th3 = cv.adaptiveThreshold(edge, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
     #                         cv.THRESH_BINARY, 99, 46)
    #_, th3 = cv.threshold(img_gray2, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    kernel = (27, 27)
    closing = cv.morphologyEx(edge, cv.MORPH_CLOSE, kernel)

    # # Finding Countours

    cont, hier = cv.findContours(closing, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    #cv.drawContours(image, cont, -1, (0, 200, 0), 2)
    min_cont = []       #pierwszy narożnik x z lewej
    max_cont = []       #ostatni narożnik x z prawej
    area = list()
    min_y, max_y, min_yr, max_yr = [0], [0], [0], [0]
    for contour in cont:                                                                    #metoda na same litery i bounding boxy
        x, y, w, h = cv.boundingRect(contour)
        if h > w and w*h > 1000 and cv.contourArea(contour) > 750 and h>65:
            #cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 250), 6)
            #cv.imshow('image_2', img_gray2)
            #print(x, y, w, h)
            min_x_cont = np.min(contour[:, :, 0])
            max_x_cont = x+w
            min_cont.append(min_x_cont)
            max_cont.append(max_x_cont)
            min_cont = sorted(min_cont)
            max_cont = sorted(max_cont, reverse=True)
            #print(f'Min cont {min_cont}, Max cont {max_cont}')
            if x == min_cont[0]:
                min_y = y
                max_y = y+h
            if x+w == max_cont[0]:
                min_yr = y
                max_yr = y+h
        #print(f'Min: {min_y_cont}, Max: {max_x_cont}')

    rect = np.zeros((4, 2), dtype=np.float32)
    min_x_tab = min_cont[0]-15
    max_x_tab = max_cont[0]+15
    min_y_tab = min_y-15
    max_y_tab = max_y+15
    min_yr_tab = min_yr-15
    max_yr_tab = max_yr+15

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

#Znajdowanie match template liter
    warped_image2 = cv.medianBlur(warped_image, 11)
    warped_image_filter = cv.bilateralFilter(warped_image2, 9, 119, 119)

    edge2 = cv.Canny(warped_image_filter, 10, 200)
    kernel = (11, 11)
    opening = cv.morphologyEx(edge2, cv.MORPH_CLOSE, kernel)

    cont2, hier2 = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    print(len(cont2))
    if len(cont2) <= 3:
        for cont in cont2:
            if cv.contourArea(cont) > 100:
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
        if h > w and w*h > 1000 and last < 8 and h > 75 and w > 20:
            cv.rectangle(warped_image_filter, (x, y), (x + w, y + h), (200, 0, 0), 6)
            # for dir_name, subdir, filesnames in os.walk(letter_dir):  # przechodzi przez wszystkie pliki w katalogu
            #     # print(dir_name, subdir, filesname)
            #     for filename in filesnames:  # sprawdza każdy plik
            #         full_name = os.path.join(dir_name, filename)  # Łaczy ścieżką z nazwą pliku
            #         template = cv.imread(str(full_name), cv.IMREAD_GRAYSCALE).astype('uint8')
            #         #template = cv.resize(template, (w, h), interpolation=cv.INTER_AREA)
            #         _, template= cv.threshold(template, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
            #         cv.imshow('template2', template)
            #         method = eval('cv.TM_CCOEFF')
            #         #Bbox = warped_image[y-10:y+h+10, x-10:x+w+10]
            #         Bbox = warped_image[y:y+h, x:x+w]
            #         Bbox = cv.resize(Bbox, (128, 256), interpolation=cv.INTER_AREA)
            #         Bbox_gray = cv.cvtColor(Bbox, cv.COLOR_BGR2GRAY).astype('uint8')
            #         _, Bbox_thresh= cv.threshold(Bbox_gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
            #
                    # res = cv.matchTemplate(Bbox_thresh, template, method)
                    # cv.imshow('template3', res)
                    # # min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                    # # if min_val < minValue:
                    # #     minValue = min_val
                    # #     letter = filename[0]
                    # (_, score, _, _) = cv.minMaxLoc(res)
                    # if score > last:
                    #     letter = filename[0]
            Bbox = warped_image_filter[y:y + h, x:x + w]
            Bbox = cv.resize(Bbox, (128, 256), interpolation=cv.INTER_AREA)
            cv.imshow('resized', Bbox)
            Bbox_gray = cv.cvtColor(Bbox, cv.COLOR_BGR2GRAY).astype('uint8')
            _, Bbox_thresh = cv.threshold(Bbox_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            fd, hog_image = hog(Bbox_gray, orientations=9, pixels_per_cell=(24, 24),
                                cells_per_block=(2, 2), visualize=True, feature_vector=True)

            directory = r"C:\Users\Andrzej\Desktop\Studia\studia - sem 1 magisterka\SW\SW_Zaliczenie\processing"
            # Pełna ścieżka do pliku z modelem
            model_path = os.path.join(directory, "klasyfikator.pkl")
            model = load(model_path)
            fd = fd.reshape(1,-1)
            letter = model.predict(fd)
            plate_num.append(str(letter[0]))
            licence_plate = ''.join(plate_num)
            scores.append(fd)
            last +=1
    #print(len(scores))
    print(plate_num)

    cv.imshow('Wyprostowana perspektywa', warped_image_filter)
    cv.imshow('opening', opening)
    cv.imshow('image', image)
    stop = time.time()
    print('Czas przetwarzania: ', stop - start)

    # Ekstrakcja cech features z Histogram of Gradients do csv

    # with open('HOG2.csv', '', newline='\n') as f:
    #     writer = csv.writer(f)
    #     for i in scores:
    #         writer.writerow(i)
    #     f.close()

    # Przerwa po każdym zdjęciu

    # key = cv.waitKey(0)
    # if key == 27:
    #     cv.destroyAllWindows()
    #
    #     exit()
    print(f'image.shape: {image.shape}')
    print(licence_plate)

    return licence_plate