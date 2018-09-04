import argparse
import cv2
from neural_style_transfer import transfer

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True)
ap.add_argument('-i', '--image', required=True)
args = vars(ap.parse_args())

image = cv2.imread(args['image'])

output = transfer(image, args['model'], True)

cv2.imshow('Original', image)
cv2.imshow('Modified', output)
cv2.waitKey(0)
