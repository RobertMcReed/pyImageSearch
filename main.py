import cv2
from scanner.scan import scan_image

image = 'scanner/set.jpg'
image = cv2.imread(image)

scan_image(image)
