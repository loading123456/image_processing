from asyncio import events
from typing import Dict, List
from xmlrpc.client import Boolean
import cv2
import numpy as np
import pytesseract
import googletrans 
import time
from PIL import Image, ImageFont, ImageDraw
from pytesseract import Output
from concurrent import futures

def isInclude( p, c) -> Boolean:
    if (
        p[0] <= c[0] 
        and p[1] <= c[1] 
        and p[0] + p[2] >= c[0] + c[2] 
        and p[1] + p[3] >= c[1] + c[3]
    ):
        return True
    return False



class TextDetector:
    def detectText(self, imgPath ) -> List:
        img = cv2.imread(imgPath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        thresh0 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU 
                                                | cv2.THRESH_BINARY_INV)[1]
        thresh1 = np.where(thresh0 == 0, 255, 0).astype('uint8')

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

        dilation0 = cv2.dilate(thresh0, rect_kernel, iterations = 1)
        dilation1 = cv2.dilate(thresh1, rect_kernel, iterations = 1)
        
        contours0 = cv2.findContours(dilation0, cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_NONE)[0]
        contours1 = cv2.findContours(dilation1, cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_NONE)[0]
                                                        
        return [self.__filterBox(contours0, contours1), thresh0, thresh1]


    def __filterBox(self, a, b) -> List:
        boxes = self.__convertToRect(a) + self.__convertToRect(b)
        result = boxes.copy()
        
        for lastBox in boxes:
            for nextBox in boxes:
                if lastBox in result and lastBox != nextBox and isInclude(lastBox, nextBox):
                    result.remove(lastBox)
    
        return result

    def __convertToRect(self,a) -> List:
        r = []
        for i in a:
            r.append(cv2.boundingRect(i))
        return r

    # def __isInclude(self, p, c) -> Boolean:
    #     if (
    #         p[0] <= c[0] 
    #         and p[1] <= c[1] 
    #         and p[0] + p[2] >= c[0] + c[2] 
    #         and p[1] + p[3] >= c[1] + c[3]
    #     ):
    #         return True
    #     return False



class TextRecognizer:
    def recognizeText(self, boxes, thresh0, thresh1) -> Dict:
        im3 = thresh0 if np.count_nonzero(thresh0==255) > np.count_nonzero(thresh0==0) else thresh1
        imageData = pytesseract.image_to_data(im3, output_type=Output.DICT)
        
        untransData = {}

        for i in boxes:
            text = self.__mergeWords(imageData, i)
            untransData[str(i)] = text
        return untransData

    def __mergeWords(self, imageData, box) -> str:
        text = ''
        for i in range(len(imageData['text'])):
            x = imageData['left'][i]
            y = imageData['top'][i]
            w = imageData['width'][i]
            h = imageData['height'][i]
            if isInclude(box, (x, y, w, h)):
                text += ' ' + imageData['text'][i]
        return ' '.join(text.split())


class TextTranslator:
    def translateText(self, untransData) -> Dict:
        transData = {}

        with futures.ThreadPoolExecutor(max_workers=4) as executor:
            jobs = {executor.submit(self.__translateLine, i, untransData[i]) for i in untransData if untransData[i] != '' }
            for job in futures.as_completed(jobs):
                r = job.result()
                if r[1] not in untransData[r[0]]:
                    transData[r[0]] = r[1]
        return transData

    def __translateLine(self, key, value):
        return key, googletrans.Translator().translate(value, dest='vi').text


class TextDraw:
    def drawText(self, transData, imgPath, desPath) ->  None:
        img = cv2.imread(imgPath)
        img1 = img.copy()
        for i in transData:
            text = transData[i]
            x, y, w, h = eval(i)
            shape = [(x, y), (x+w, y+h)]
            cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 255, 0),4)
            # font , text = self.__getFont(eval(i), text)
            # img1.text((x, y), text=text, font=font, fill=(255,255,255))
        cv2.imwrite(desPath, img1)
        # img.save(desPath, quality=99)

    def __getFont(self, box, text):
        x, y, w, h = box
        maxFontSize = int(h * 3/4)
        fontSize = maxFontSize
        lines = 1
        font = ImageFont.truetype(r'font/Arimo-VariableFont_wght.ttf',  fontSize)

        while font.getbbox(text)[2] > w * lines:
            fontSize -= 2
            lines = int(maxFontSize / fontSize)
            font = ImageFont.truetype(r'font/Arimo-VariableFont_wght.ttf',  fontSize)

        tFont =  ImageFont.truetype(r'font/Arimo-VariableFont_wght.ttf',  fontSize + 1)
        
        if tFont.getbbox(text)[2] <= w * int(maxFontSize / (fontSize + 1)) :
            lines = int(maxFontSize / (fontSize + 1)) 
            font = tFont

        return font, self.__endline(text, lines, font, w)


    def __endline(self, text, lines, font, width) -> str:
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

class Translator:
    def translate(self, imgPath, desPath = None) -> None:
        if not desPath:
            desPath = imgPath

        st = time.time()
        boxes, thresh0, thresh1 = TextDetector().detectText(imgPath)
        print("Detect text: ", time.time() - st)
        
        st = time.time()
        untransData = TextRecognizer().recognizeText(boxes, thresh0, thresh1)
        print("Recognize Text: ", time.time() - st)

        img = cv2.imread(imgPath)
        for i in untransData:
            x, y, w, h = eval(i)
            # print(untransData[i])
            cv2.rectangle(img, (x, y), (x+w, y + h), (0, 255, 0), 2)
        
        cv2.imwrite(desPath, img)
        
        st = time.time()
        # transData = TextTranslator().translateText(untransData)
        
        print("Translate Text: ", time.time() - st)

        st = time.time()
        # TextDraw().drawText(transData, imgPath, desPath)
        print("Draw Text: ", time.time() - st)


# Translator().translate('images/1.png', 'output/a1.png')
# Translator().translate('images/2.png', 'output/2.png')
# Translator().translate('images/3.png', 'output/3.png')
# Translator().translate('images/4.png', 'output/4.png')
Translator().translate('images/5.png', 'output/5.png')
# Translator().translate('images/a5.png', 'output/a5.png')
# Translator().translate('images/a6.png', 'output/a6.png')

