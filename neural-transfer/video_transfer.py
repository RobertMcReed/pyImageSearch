from imutils.video import VideoStream
from imutils import paths
import itertools
import argparse
import imutils
import time
import cv2
from neural_style_transfer import transfer

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--models', required=True)
models = vars(ap.parse_args())['models']

# grab the path to neural style transfers in the models directory
modelPaths = paths.list_files(models, validExts=('.t7',))
modelPaths = sorted(list(modelPaths))

# generate unique IDs for each model path and combine the lists
models = list(zip(range(0, len(modelPaths)), (modelPaths)))

# use the cycle function to loop over all model paths, then restart when the end is reached
modelIter = itertools.cycle(models)
modelID, modelPath = next(modelIter)

print('[INFO] starting video stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)
print('[INFO] {}. {}'.format(modelID + 1, modelPath))
net = None

while True:
    lastPath = modelPath
    frame = vs.read()
    orig = frame.copy()
    output, net = transfer(frame, modelPath, net=net)

    cv2.imshow('Output', output)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('n'):
        modelID, modelPath = next(modelIter)
        if modelPath != lastPath:
            net = None
        print('[INFO] {}. {}'.format(modelID + 1, modelPath))
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
