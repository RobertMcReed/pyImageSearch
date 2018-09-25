import numpy as np
import cv2


def distance(a, b):
    """Pass in a 2 lists of [x,y] pairs to return their distance"""
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# you must maintain a consistent ordering when using this function
def order_points(pts):
    # initialize a list of coordinates that will be ordered TL, TR, BR, BL
    # np.zeros creates a new array with the shape specified filled with 0's
    rect = np.zeros((4, 2), dtype="float32")

    # the top left point will have the smallest sum, and the bottom right the largest
    # axis=1 indicates to sum each pair[[x,y], [x,y], [x,y], [x,y]]
    s = pts.sum(axis=1)
    # the smallest sum
    rect[0] = pts[np.argmin(s)]
    # the largest sum
    rect[2] = pts[np.argmax(s)]

    # compute the difference between the points
    # the top-right will have the smallest difference, and the bottom left the largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points via order pts and unpack them
    rect = order_points(pts)
    tl, tr, br, bl = rect

    # compute the width of the new image,
    # the max distance between the BR and BL x-coords or the TR and TL x-coords once straightened
    widthA = distance(br, bl)
    widthB = distance(tr, tl)
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image,
    # the max distance between the TR and BR or TL and BL
    heightA = distance(tr, br)
    heightB = distance(tl, bl)
    maxHeight = max(int(heightA), int(heightB))

    # construct the set of destination points to obtain a "birds eye view"
    # pts are again in TL, TR, BR, BL

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
