from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
from read_text import ocr


east = 'frozen_east_text_detection.pb'
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True)
ap.add_argument('-c', '--min_confidence', type=float, default=0.5)
ap.add_argument('-w', '--width', type=int, default=320)
ap.add_argument('-e', '--height', type=int, default=320)
ap.add_argument('-o', '--offset', type=int, default=5)
ap.add_argument('-b', '--blur', type=bool, default=False)
ap.add_argument('-f', '--font_size', type=float, default=2)
ap.add_argument('-r', '--resize', type=int, default=1000)
args = vars(ap.parse_args())

# resize the input image if flagged
image = cv2.imread(args['image'])
if args['resize']:
    newW = args['resize']
    Ho, Wo = image.shape[:2]
    newH = int(Ho * newW / Wo)

    image = cv2.resize(image, (newW, newH))

original_image = image.copy()

# identify the ratio of change for each dimension
Ho, Wo = image.shape[:2]
newH, newW = args['height'], args['width']
ratioW = Wo / float(newW)
ratioH = Ho / float(newH)

# resize the image and get the new dimensions
image = cv2.resize(image, (newW, newH))
H, W = image.shape[:2]

# define the output layer names for the EAST detector model
# The first is the output probabilities
# the second is used to derive the boudning box coordinates of the text

layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
]

# load the pre-trained EAST text detector
print("[INFO] Loading the EAST text detector...")
net = cv2.dnn.readNet(east)

# construct a blob from the image and perform a forward pass of the model to obtain the output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
scores, geometry = net.forward(layerNames)
end = time.time()
print("[INFO] Text detection took {:.6f} seconds".format(end - start))

# determine the number of rows and columns from the scores,
# then initialize the set of bounding box rectangles and confidence scores
numRows, numCols = scores.shape[2:4]
rects = []
confidences = []

for y in range(0, numRows):
    # extract the scores (probabilities), followed by the geometrical data used to derive the potential bounding
    # box coordinates that surround the text
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]

    # loop over each column
    for x in range(0, numCols):
        # if the score has a low probability, ignore it
        if scoresData[x] < args["min_confidence"]:
            continue

        # compute the offset factor as our resulting feature maps will be 4x smaller than the input image
        offsetX, offsetY = x * 4.0, y * 4.0

        # extract the rotation angle for the prediction and then compute the sine and cosine
        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)

        # use the geometry data to derive the width and height of the bounding box
        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]

        # compute the starting and ending (x,y)-coords for the bounding box
        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)

        # add the bounding box coordinates and probability score to our respective lists
        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])

boxes = non_max_suppression(np.array(rects), probs=confidences)
# loop over the bounding boxes

labeled_image = original_image.copy()

for (startX, startY, endX, endY) in boxes:
    # scale the bounding box coordinates based on the respective ratios
    startX = int(startX * ratioW)
    startY = int(startY * ratioH)
    endX = int(endX * ratioW)
    endY = int(endY * ratioH)

    # create a numpy array representing the found text area
    box_offset = args['offset']
    box_l = startX - box_offset if startX - box_offset > 0 else 0
    box_r = endX + box_offset if endX + box_offset < Wo else Wo
    box_u = startY - box_offset if startY - box_offset > 0 else 0
    box_d = endY + box_offset if endY + box_offset < Ho else Ho

    text_slice = original_image[box_u:box_d, box_l:box_r]
    text = ocr(text_slice, args['blur'])

    # draw the bounding box on the image
    cv2.rectangle(labeled_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # put the read text on the image
    labelY = startY - 20 if startY - 20 > 0 else endY + 20
    font_size = args['font_size']
    cv2.putText(labeled_image, text, (startX, labelY), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 255, 0), 2)

cv2.imshow("OCR - {}".format(args['image']), labeled_image)
cv2.waitKey(0)