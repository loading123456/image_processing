from os import path
from sys import argv
import pytesseract
import cv2

def image_processing(input: str):
    print("Scanning: ", input)

    if not path.isfile(input):
        print("File doesn't exist.")

        return;

    filename = path.basename(input)
    data = {}
    img = cv2.imread(input)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    im2 = img.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped = im2[y:y + h, x:x + w]
        text = pytesseract.image_to_string(cropped)
        data[''.join(map(str, [x, y, x+w, y+h]))] = text

    cv2.imwrite('output/' + filename, im2)
    print('Done!');

image_processing(argv[1]);
