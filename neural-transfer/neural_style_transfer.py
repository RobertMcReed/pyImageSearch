import imutils
import time
import cv2


def print_if(bool, *args):
    if bool:
        print(*args)


def transfer(image, model, verbose=False, net=False):
    print_if(verbose, '[INFO] loading style transfer model...')
    net = net if net else cv2.dnn.readNetFromTorch(model)

    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    start = time.time()
    output = net.forward()
    end = time.time()

    # reshape the output tensor, add back in the mean subtraction, and then swap the channel ordering
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output /= 255.0
    output = output.transpose(1, 2, 0)

    print_if(verbose, '[INFO] neural style transfer took {:.4f} seconds'.format(end - start))

    return output, net
