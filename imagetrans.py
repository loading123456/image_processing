from tokenize import String
from turtle import Turtle
from typing import List
from xmlrpc.client import Boolean
import cv2
import numpy as np
import pytesseract
import googletrans 
import time
from PIL import Image, ImageFont, ImageDraw
from pytesseract import Output
from concurrent import futures

class TextDetector:
    def detectText(self, imgPath ) -> Turtle:
        img = cv2.imread(imgPath)
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        thresh0 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU 
                                                | cv2.THRESH_BINARY_INV)[1]
        thresh1 = np.where(self.thresh0 == 0, 255, 0).astype('uint8')

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

        dilation0 = cv2.dilate(thresh0, rect_kernel, iterations = 1)
        dilation1 = cv2.dilate(thresh1, rect_kernel, iterations = 1)
        
        contours0 = cv2.findContours(dilation0, cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_NONE)[0]
        contours1 = cv2.findContours(dilation1, cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_NONE)[0]
                                                        
        return (self.__filterBox(contours0, contours1), thresh0, thresh1)


    def __filterBox(self, a, b) -> List:
        boxes = self.__convertToRect(a) + self.__convertToRect(b)
        result = boxes.copy()
        
        for lastBox in boxes:
            for nextBox in boxes:
                if lastBox in result and lastBox != nextBox and self.__isInclude(lastBox, nextBox):
                    result.remove(lastBox)
    
        return result

    def __convertToRect(self,a) -> List:
        r = []
        for i in a:
            r.append(cv2.boundingRect(i))
        return r

    def __isInclude(self, p, c) -> Boolean:
        if (
            p[0] <= c[0] 
            and p[1] <= c[1] 
            and p[0] + p[2] >= c[0] + c[2] 
            and p[1] + p[3] >= c[1] + c[3]
        ):
            return True
        return False



class TextRecognizer:
    def recognizeText(self) -> None:
        im3 = self.thresh0 if np.count_nonzero(self.thresh0==255) > np.count_nonzero(self.thresh0==0) else self.thresh1
        imageData = pytesseract.image_to_data(im3, output_type=Output.DICT)
        
        self.untransData = {}

        for i in self.boxes:
            text = self.mergeWords(imageData, i)
            self.untransData[str(i)] = text
            

    def mergeWords(self, imageData, box) -> String:
        text = ''
        for i in range(len(imageData['text'])):
            x = imageData['left'][i]
            y = imageData['top'][i]
            w = imageData['width'][i]
            h = imageData['height'][i]
            if self.isInclude(box, (x, y, w, h)):
                text += ' ' + imageData['text'][i]
        return ' '.join(text.split())


class TextTranslator:
    def translateText(self) -> None:
        self.transData = {}

        with futures.ThreadPoolExecutor(max_workers=4) as executor:
            jobs = {executor.submit(self.translateLine, i, self.untransData[i]) for i in self.untransData if self.untransData[i] != '' }
            for job in futures.as_completed(jobs):
                r = job.result()
                if r[1] not in self.untransData[r[0]]:
                    self.transData[r[0]] = r[1]


    def translateLine(self, key, value):
        return key, googletrans.Translator().translate(value, dest='vi').text


class TextDraw:
    def drawText(self) ->  None:
        img = Image.open(self.imagePath)
        img1 = ImageDraw.Draw(img)

        for i in self.transData:
            text = self.transData[i]
            x, y, w, h = eval(i)
            shape = [(x, y), (x+w, y+h)]
            img1.rectangle(shape, fill =(0, 0, 0))
            font , text = self.getFont(eval(i), text)
            img1.text((x, y), text=text, font=font, fill=(255,255,255))
        img.save(self.desPath, quality=99)

    def getFont(self, box, text):
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

        return font, self.endline(text, lines, font, w)


    def endline(self, text, lines, font, width) -> String:
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
    def __init__(self) -> None:
        super().__init__()

    def translate(self, imagePath, desPath = None) -> None:
        
        if not desPath:
            self.desPath = imagePath
        else:
            self.desPath = desPath

        boxes, thresh0, thresh1 = TextDetector().detectText()

        self.detectText()
        self.recognizeText()
        self.translateText()
        self.drawText()


Translator().translate('images/1.png')