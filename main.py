import pytesseract
from pytesseract import Output
import cv2
import socket

input = []
isBusy = False

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddress = ('localhost', 2021)


def listening():
    global input
    
    while True:
        data = sock.recvfrom(4096)
        if data:
            input.append(data.decode()) 
            if not isBusy:
                imageProcessing()




def imageProcessing():
    global input
    global isBusy

    isBusy = True

    print("Scanning: ",input[0])

    if len(input) > 0:
        data = {}
        link = input[0]
        img = cv2.imread('input/' + link)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        
        dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_NONE)
        im2 = img.copy()
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = im2[y:y + h, x:x + w]
            text = pytesseract.image_to_string(cropped)
            data[''.join(map(str, [x, y, x+w, y+h]))] = text
        
        cv2.imwrite('output/' + link, im2)
        sock.sendto(str(data).encode(), serverAddress)

    input.pop(0)

    if len(input) > 0:
        imageProcessing()
    else:
        isBusy = False

    print("     -->Finished: ", link)



listening()
