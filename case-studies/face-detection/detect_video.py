from face_detector import FaceDetector
import imutils
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--face', required=True, help='path to face cascade')
ap.add_argument('-v', '--video', help='path to optional video file')
args = vars(ap.parse_args())

fd = FaceDetector(args['face'])

camera = cv2.VideoCapture(0) if not args.get('video', False) else cv2.VideoCapture(args['video'])

while True:
    grabbed, frame = camera.read()

    if args.get('video') and not grabbed:
        break

    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRects = fd.detect(gray, minSize=(10, 10))
    frameClone = frame.copy()

    for (x, y, w, h) in faceRects:
        cv2.rectangle(frameClone, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow('Faces', frameClone)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
