from transform import four_point_transform
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image')
ap.add_argument('-c', '--coords')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
pts = np.array(eval(args['coords']), dtype="float32")
print(pts)
warped = four_point_transform(image, pts)
cv2.circle(image, (pts[0][0], pts[0][1]), 5, (0, 0, 255), 2)
cv2.circle(image, (pts[1][0], pts[1][1]), 5, (0, 0, 255), 2)
cv2.circle(image, (pts[2][0], pts[2][1]), 5, (0, 0, 255), 2)
cv2.circle(image, (pts[3][0], pts[3][1]), 5, (0, 0, 255), 2)

cv2.imshow('Original', image)
cv2.imshow('Warped', warped)

cv2.waitKey(0)

# python persepective_transform/transform_example.py -i perspective_transform/test.png -c '[(140, 465), (690, 230), (900, 500), (360, 850)]'