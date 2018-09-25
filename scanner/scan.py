import sys
sys.path.insert(0, '~/Documents/programming/python/ml/pyImageSearchBlog/perspective_transform')

import cv2
import imutils

def scan_image(image, bw=True):
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    # convert the image to grayscale, blur it, and find edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 75, 200)

    print('[INFO] 1: Edge Detection')
    # cv2.imshow('Image', image)
    # cv2.imshow('Edged', edged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]
    screenCnts = []

    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            screenCnts.append(approx)

    print('STEP 2: Find the contours')
    cv2.drawContours(image, screenCnts, -1, (0, 255, 0), 2)
    cv2.imshow('Outline', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()