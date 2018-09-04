from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video')
ap.add_argument('-b', '--buffer', type=int, default=64)
args = vars(ap.parse_args())


def tup_plus(tuple, addend):
    return tuple[0] + addend, tuple[1] + addend, tuple[2] + addend


def tweak_tup(tup, index, addend):
    one = tup[0] if index != 0 else tup[0] + addend
    two = tup[1] if index != 1 else tup[1] + addend
    three = tup[2] if index != 2 else tup[2] + addend
    tweaked_tup = (one, two, three)
    print('[INFO] HSV: {}'.format(tweaked_tup))

    return tweaked_tup


# initialize the list of tracked points
pts = deque(maxlen=args['buffer'])

video = args.get('video', False)

# if no video is supplied, grab a reference to the webcam
if not video:
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args['video'])

time.sleep(1)

# badge defaults
avg = (98, 114, 137)
diff = 50

# adjusted badge
avg = (96, 262, 125)
diff = 100

while True:
    # define the lower and upper boundaries of the color to be tracked in the HSV color space
    lower = tup_plus(avg, -diff)
    upper = tup_plus(avg, diff)

    frame = vs.read()
    frame = frame[1] if video else frame

    # end if video is provided and we have reached the end
    if frame is None:
        break

    # resize the frame, blur it, and convert it to HSV
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color range specified
    # perform dilations and erosions to remove any small blobs in the mask
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current center of the found object
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    center = None

    # if no contours are found don't continue
    if len(cnts) > 0:
        # find the largest contour in the mask and use it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 75:
            # draw the circle and centroid on the frame and update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # update the points queue
    pts.appendleft(center)

    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # compute the thickness of the line and draw the connecting lines
        thickness = int(np.sqrt(args['buffer'] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.imshow('Tracker', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('a'):
        avg = tweak_tup(avg, 0, 2)
    elif key == ord('s'):
        avg = tweak_tup(avg, 1, 2)
    elif key == ord('d'):
        avg = tweak_tup(avg, 2, 2)
    elif key == ord('z'):
        avg = tweak_tup(avg, 0, -2)
    elif key == ord('x'):
        avg = tweak_tup(avg, 1, -2)
    elif key == ord('c'):
        avg = tweak_tup(avg, 2, -2)
    elif key == ord('e'):
        diff += 2
        print('[INFO] Diff: {}'.format(diff))
    elif key == ord('w'):
        diff -= 2
        print('[INFO] Diff: {}'.format(diff))


if not video:
    vs.stop()

else:
    vs.release()

cv2.destroyAllWindows()