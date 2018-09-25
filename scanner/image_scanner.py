import argparse
import cv2
from scanner.scan import scan_image

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True)
image = vars(ap.parse_args())['image']
image = cv2.imread(image)

scan_image(image)
