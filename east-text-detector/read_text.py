from PIL import Image
import pytesseract
import os
import cv2

def ocr(image, blur=False):
    # convert the image slice to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    if blur:
        gray = cv2.medianBlur(gray, 3)

    # create a temporary file to read in PIL form
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)

    # use tesseract to read the text in the image slice
    text = pytesseract.image_to_string(Image.open(filename))

    # remove the temporary image
    os.remove(filename)

    return text
