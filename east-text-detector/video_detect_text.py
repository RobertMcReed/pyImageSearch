from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2

def decode_predictions(scores, geometry, confidence=0.5):
    # grab the number of rows and columns from the scores volume
    # initialize our set of bounding box rectangles and corresponding confidence scores
    numRows, numCols = scores.shape[2:4]
    rects = []
    conf = []

    for y in range(0, numRows):
        # extract the scores (probs), geometrical data to derive the bounding box, and coordinates that surround the text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            # if the score is too low ignore it
            if scoresData[x] < confidence:
                continue

            # compute the offset factor as our feature maps will be 4x smaller than the input image
            offsetX, offsetY = x * 4.0, y * 4.0

            # extract the rotation angle for the prediction and compute the sin and cos
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute the starting and ending coords for the bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coords and probabilities to the lists
            rects.append((startX, startY, endX, endY))
            conf.append(scoresData[x])

    # return a tuple of the boxes and corresponding confidences
    return rects, conf

east = 'frozen_east_text_detection.pb'
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video')
ap.add_argument('-c', '--min_confidence', type=float, default=0.5)
ap.add_argument('-w', '--width', type=int, default=320)
ap.add_argument('-e', '--height', type=int, default=320)
args = vars(ap.parse_args())
video = args.get('video', False)

# initialize the frame dimensions and ratios
W, H = None, None
newW, newH = args['width'], args['height']
rW, rH = None, None

# define the output layer names for the EAST detector
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
]

# load the EAST text detector
print('[INFO] Loading text detector...')
net = cv2.dnn.readNet(east)

# if a video path wasn't supplied, grab the live stream
if not video:
    print('[INFO] Starting video stream...')
    vs = VideoStream(src=0).start()
    time.sleep(1)

else:
    vs = cv2.VideoCapture(args['video'])

# start the FPS throughput estimator
fps = FPS().start()

while True:
    # grab the current frame and handle webcam vs video file
    frame = vs.read()
    frame = frame[1] if video else frame

    # check if we have reached the end of the stream
    if frame is None:
        break

    # resize the frame, maintaining the aspect ratio
    frame = imutils.resize(frame, width=1000)
    orig = frame.copy()

    # if the frame dimensions are None we need to compute the ratio of old frame dimensions to new dimensions
    if W is None or H is None:
        H, W = frame.shape[:2]
        rW = W / float(newW)
        rH = H / float(newH)

    # resize the frame, ignoring aspect ratio
    frame = cv2.resize(frame, (newW, newH))

    # construct a blob from the frame and perform a foward pass of the model to get the output layers
    blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    scores, geometry = net.forward(layerNames)

    # decode the predictions and apply the non-maxima suppression to suppress weak or overlapping bounding boxes
    rects, confs = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confs)

    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # update the counter
    fps.update()

    cv2.imshow("Text Detector", orig)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

fps.stop()
print('[INFO] Elapsed time: {:.2f}'.format(fps.elapsed()))
print('[INFO] Approximate FPS: {:.2f}'.format(fps.fps()))

if not video:
    vs.stop()
else:
    vs.release()

cv2.destroyAllWindows()
