import numpy as np
import cv2 as cv
import os
import time


def perform_processing(image: np.ndarray) -> str:
    stat = time.time()
    # resizing image to smaller
    image = cv.resize(image, (0, 0), fx=0.3, fy=0.3)
    img_gray = image.copy()
    img_gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)
    row, col = img_gray.shape[:]

    # filtering image
    img_gray = cv.GaussianBlur(img_gray, (11, 11), 0)
    #img_gray = cv.medianBlur(img_gray, 21)
    img_gray = cv.bilateralFilter(img_gray, 15, 17, 17)

    hsv_frame = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    #print(hsv_frame)

    # thresholding and opening
    #ret, dst = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)
    #th3, thresh3 = cv.threshold(img_gray, 100, 255, cv.THRESH_TOZERO_INV)
    th3 = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 45, 2)
    kernel = (9, 9)
    open = cv.morphologyEx(th3, cv.MORPH_OPEN, kernel)

    # Finding Countours
    edge = cv.Canny(open, 30, 200)
    cont, hier = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    area = list()
    for contour in cont:
        area.append(cv.contourArea(contour))
        if area[-1] > 20:
            cv.drawContours(image, contour, -1, (0, 200, 0), 5)

    area = sorted(area, reverse=True)
    print(area[:10])

    if len(cont) >= 4:
        # Znalezienie konturów o największym obszarze
        largest_contours = sorted(cont, key=cv.contourArea, reverse=True)[:4]

        # Przygotowanie punktów w odpowiednim formacie dla transformacji perspektywicznej
        points = np.vstack([contour.reshape(-1, 2) for contour in largest_contours])
        rect = np.zeros((4, 2), dtype=np.float32)

        # Obliczenie skrajnych punktów dla transformacji perspektywicznej
        sum_points = points.sum(axis=1)
        rect[0] = points[np.argmin(sum_points)]
        rect[2] = points[np.argmax(sum_points)]

        diff_points = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff_points)]
        rect[3] = points[np.argmax(diff_points)]

        # Określenie docelowych rozmiarów prostokąta wynikowego
        width = max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3]))
        height = max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2]))
        dst_rect = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        M = cv.getPerspectiveTransform(rect, dst_rect)
        warped_image = cv.warpPerspective(image, M, (int(width), int(height)))

    cv.imshow('Wyprostowana perspektywa', warped_image)

    cv.imshow('image', image)
    key = cv.waitKey(0)
    if key == 27:
        cv.destroyAllWindows()
        exit()
    print(f'image.shape: {image.shape}')
    # TODO: add image processing here
    stop = time.time()

    return 'PO15'