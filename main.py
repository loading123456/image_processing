import cv2
import numpy as np
import pytesseract
import time
from PIL import Image, ImageFont, ImageDraw
from pytesseract import Output
from googletrans import Translator
from concurrent import futures

runTime = 0

def isInclude(p, c):
    if (
        p[0] <= c[0] 
        and p[1] <= c[1] 
        and p[0] + p[2] >= c[0] + c[2] 
        and p[1] + p[3] >= c[1] + c[3]
    ):
        return True

    return False

def convertToRect(a):
    r = []
    for i in a:
        r.append(cv2.boundingRect(i))
 
    return r

def filterBox(a, b):
    boxes = convertToRect(a) + convertToRect(b)

    result = boxes.copy()
    
    for lastBox in boxes:
        for nextBox in boxes:
            if lastBox in result and lastBox != nextBox and isInclude(lastBox, nextBox):
                result.remove(lastBox)
   
    return result
  


def textDetector(img):
    st = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh0 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU 
                                            | cv2.THRESH_BINARY_INV)

    thresh1 = np.where(thresh0 == 0, 255, 0).astype('uint8')

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

    dilation0 = cv2.dilate(thresh0, rect_kernel, iterations = 1)
    dilation1 = cv2.dilate(thresh1, rect_kernel, iterations = 1)

    contours0, hierarchy = cv2.findContours(dilation0, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)
    contours1, hierarchy1 = cv2.findContours(dilation1, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)
                                                    
    print('Text detector execution time:', time.time() - st, 'seconds')

    return filterBox(contours0, contours1), thresh0, thresh1

def mergeWords(imageData, box):
    text = ''
    for i in range(len(imageData['text'])):
        x = imageData['left'][i]
        y = imageData['top'][i]
        w = imageData['width'][i]
        h = imageData['height'][i]
        if isInclude(box, (x, y, w, h)):
            text += ' ' + imageData['text'][i]
    return ' '.join(text.split())

# ===============================================================================

def textRecognition(boxes, thresh0, thresh1):
    st = time.time()
    
    im3 = thresh0 if np.count_nonzero(thresh0==255) > np.count_nonzero(thresh0==0) else thresh1
    imageData = pytesseract.image_to_data(im3, output_type=Output.DICT)
    
    data = {}

    for i in boxes:
        text = mergeWords(imageData, i)
        data[str(i)] = text
        

    print('Text recognition execution time:', time.time() - st, 'seconds')
    return data

def translateLine(key, value):
    return key, Translator().translate(value, dest='vi').text


def translateText(data):
    st = time.time()
    result = {}

    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        jobs = {executor.submit(translateLine, i, data[i]) for i in data if data[i] != '' }
        for job in futures.as_completed(jobs):
            r = job.result()
            if r[1] not in data[r[0]]:
                result[r[0]] = r[1]

    print('Translate execution time:', time.time() - st, 'seconds')
    return result

def endline(text, lines, font, width):
    if lines == 1:
        return text
    
    words = text.split()
    result = ''
    step = int(len(words)/lines)
    posStart = 0
    for i in range(lines-1):
        realStep = step

        line = ' '.join(words[posStart: posStart + realStep])
        lineWidth = font.getbbox(line)[2]

        if lineWidth > width:
            while lineWidth > width:
                realStep -= 1
                line = ' '.join(words[posStart: posStart + realStep])
                lineWidth = font.getbbox(line)[2]

        else:
            while lineWidth < width:
                realStep += 1
                line = ' '.join(words[posStart: posStart + realStep])
                lineWidth = font.getbbox(line)[2]
            
            if lineWidth > width:
                realStep -= 1
                line = ' '.join(words[posStart: posStart + realStep])

        result = '\n'.join([result, line])
        posStart += realStep

    result = result[1:]
    line = ' '.join(words[posStart:])
    return '\n'.join([result, line])

def getFont(box, text):
    x, y, w, h = box
    maxFontSize = int(h * 3/4)
    fontSize = maxFontSize
    lines = 1
    font = ImageFont.truetype(r'font/Arimo-VariableFont_wght.ttf',  fontSize)

    while   font.getbbox(text)[2] > w * lines:
        fontSize -= 2
        lines = int(maxFontSize / fontSize)
        font = ImageFont.truetype(r'font/Arimo-VariableFont_wght.ttf',  fontSize)

    tFont =  ImageFont.truetype(r'font/Arimo-VariableFont_wght.ttf',  fontSize + 1)
    
    if tFont.getbbox(text)[2] <= w * int(maxFontSize / (fontSize + 1)) :
        lines = int(maxFontSize / (fontSize + 1)) 
        font = tFont

    # if



    return font, endline(text, lines, font, w), x, y


def drawText(i , img1, text):
        x, y, w, h = eval(i)
        
        font , text= getFont(eval(i), text)
        img1.text((x, y), text=text, font=font, fill=(255,255,255,255))
        

def drawImage(transData, img_path):
    global runTime
    st = time.time()
    img = Image.open("images/" + img_path)
    img1 = ImageDraw.Draw(img)
    
    for i in transData:
        text = transData[i]
        x, y, w, h = eval(i)
        shape = [(x, y), (x+w, y+h)]
        img1.rectangle(shape, fill =(0, 0, 0))
        font , text, x, y= getFont(eval(i), text)
        rt = time.time()
        img1.text((x, y), text=text, font=font, fill=(255,255,255))
        runTime += time.time() - rt


        

    # with futures.ThreadPoolExecutor(max_workers=2) as executor:
    #     jobs = {executor.submit(drawText, i, img1, transData[i]) for i in transData}
    #     for job in futures.as_completed(jobs):
    #         r = job.result()


    print('Draw execution time:', time.time() - st, 'seconds')

    img.save('output/' + img_path, quality=99)


def translate(img_path):
    allTime = time.time()
    img = cv2.imread("images/" + img_path)

    boxes, thresh0, thresh1 = textDetector(img)
    unTransData = textRecognition(boxes, thresh0, thresh1)

    transData  = translateText(unTransData)
    drawImage(transData, img_path)
    print("All time: ", time.time()-allTime)
    print("======================================")


  

# translateImage('test1.png')
# translateImage('test2.png')
# translateImage('test3.png')
# translateImage('test4.png')
translate('2.png')