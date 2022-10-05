from xmlrpc.client import Boolean
import cv2
import numpy as np
import pytesseract
import googletrans 
import time
from pytesseract import Output
import re
from PIL import Image, ImageFont, ImageDraw


def getContrastColor(rbg):
    return [255-rbg[0], 255-rbg[1], 255-rbg[2]]

def getContrastImg(img):
    contrastImg = img.copy()

    for y in range(len(contrastImg)):
        for x in range(len(contrastImg[y])):
            contrastImg[y][x] = getContrastColor(contrastImg[y][x])
    return contrastImg            


class Word:
    def __init__(self, box, confi, text) -> None:
        self.text = text
        self.startPoint = [box[0], box[1]]
        self.endPoint = [box[0] + box[2], box[1] + box[3]]
        self.centerPoint = [box[0] + box[2] * 0.5, box[1] + box[3] * 0.5]
        self.width = box[2]
        self.height = box[3]
        self.confident = int(confi)
 
    def show(self):
        print("Start point : ",self.startPoint)
        print("End point   : ",self.endPoint)
        print("Center point: ",self.centerPoint)
        print("Width       : ",self.width)
        print("Height      : ",self.height)
        print("Confident   : ",self.confident)
        print("Text        : ",self.text)
        print("-----------------------------------------------------")


class Line:
    def __init__(self, word:Word) -> None:
        self.words = [word]
        self.size = 1
        self.centerPoint = word.centerPoint
        self.startPoint = word.startPoint
        self.endPoint = word.endPoint
        self.height = word.height
        self.X_NOUN = 0.7
        self.Y_NOUN = 0.3
        self.epsilonX = self.height * self.X_NOUN
        self.epsilonY = self.height * self.Y_NOUN


    def __getPosition(self, word:Word) -> int:
        if abs(self.centerPoint[1] - word.centerPoint[1]) <= self.epsilonY:
            position = 0

            for node in self.words:
                if node.centerPoint[0] < word.centerPoint[0]:
                    position += 1
                else:
                    break
            if position == 0:
                distance = self.words[0].startPoint[0] - word.endPoint[0]
                if distance <= self.epsilonX:
                    return 0
                return -1

            if position == self.size:
                distance = word.startPoint[0] - self.words[-1].endPoint[0]
                if distance <= self.epsilonX :
                    return position
                return -1
            lastDistance =  word.startPoint[0] - self.words[position - 1].endPoint[0]
            nextDistance = self.words[position].startPoint[0] - word.endPoint[0]

            if (lastDistance <= self.epsilonX 
                and nextDistance <= self.epsilonX
            ):
                return position
        return -1
    def __updateLine(self, word:Word):
        if self.startPoint[0] > word.startPoint[0]:
            self.startPoint[0] = word.startPoint[0]
        if self.startPoint[1] > word.startPoint[1]:
            self.startPoint[1] = word.startPoint[1]

        if self.endPoint[0] < word.endPoint[0]:
            self.endPoint[0] = word.endPoint[0]
        if self.endPoint[1] < word.endPoint[1]:
            self.endPoint[1] = word.endPoint[1]
        self.size += 1
        
        self.centerPoint[0] = (self.centerPoint[0] 
                                + (word.centerPoint[0] - self.centerPoint[0]) / self.size)
        self.centerPoint[1] =  (self.centerPoint[1] 
                                + (word.centerPoint[1] - self.centerPoint[1]) / self.size)

        self.epsilonX = (self.endPoint[1] - self.startPoint[1]) * self.X_NOUN
        self.epsilonY = (self.endPoint[1] - self.startPoint[1]) * self.Y_NOUN

    def insertWord(self, word:Word) -> Boolean:
        position = self.__getPosition(word)

        if position != -1:
            if position == self.size:
                self.words.append(word)
            else:
                self.words.insert(position, word)
            self.__updateLine(word)
            return True
        return False

    def getText(self):
        text = ''
        for word in self.words:
            text += ' ' + word.text
        if text[0] == ' ':
            text = text[1:]
        return re.sub(' +', ' ', text)


    def show(self):
        print("Size        : ",self.size)
        print("End point   : ",self.endPoint)
        print("Center point: ",self.centerPoint)
        print("Height      : ",self.height)
        print("Epsilon X   : ",self.epsilonX)
        print("Epsilon Y   : ",self.epsilonY)
        print("Text        : ",self.getText())
        print("-----------------------------------------------------")   



class UnArea:
    def __init__(self, word:Word) -> None:
        self.centerPoint = word.centerPoint
        self.startPoint = word.startPoint
        self.endPoint = word.endPoint
        self.height = word.height
        self.X_NOUN = 0.25
        self.Y_NOUN = 0.3
        self.epsilonX = self.height * self.X_NOUN
        self.epsilonY = self.height * self.Y_NOUN
        self.text = ''
        self.size = 1


    def __getPosition(self, word:Word) -> int:
        if abs(self.centerPoint[1] - word.centerPoint[1]) <= self.epsilonY:
            if abs(word.startPoint[0] - self.endPoint[0]) <= self.epsilonX:
                return 1
        return -1

    def __updateLine(self, word:Word):
        if self.startPoint[0] > word.startPoint[0]:
            self.startPoint[0] = word.startPoint[0]
        if self.startPoint[1] > word.startPoint[1]:
            self.startPoint[1] = word.startPoint[1]

        if self.endPoint[0] < word.endPoint[0]:
            self.endPoint[0] = word.endPoint[0]
        if self.endPoint[1] < word.endPoint[1]:
            self.endPoint[1] = word.endPoint[1]
        
        self.size += 1
        self.centerPoint[0] = (self.centerPoint[0] 
                                + (word.centerPoint[0] - self.centerPoint[0]) / self.size)
        self.centerPoint[1] =  (self.centerPoint[1] 
                                + (word.centerPoint[1] - self.centerPoint[1]) / self.size)

        self.epsilonX = (self.endPoint[1] - self.startPoint[1]) * self.X_NOUN
        self.epsilonY = (self.endPoint[1] - self.startPoint[1]) * self.Y_NOUN

    def insertWord(self, word:Word) -> Boolean:
        position = self.__getPosition(word)

        if position != -1:
            self.__updateLine(word)
            return True
        return False


    def updateText(self, img):
        x = int(self.startPoint[0] - self.epsilonX 
                if self.startPoint[0] - self.epsilonX >= 0 
                else self.startPoint[0])
        y = int(self.startPoint[1] - self.epsilonX 
                if self.startPoint[1] - self.epsilonX >= 0 
                else self.startPoint[1])
        w = int(self.endPoint[0] + self.epsilonX 
                if self.endPoint[0] + self.epsilonX <= img.shape[0]
                else self.endPoint[0]) 
        h = int(self.endPoint[1] + self.epsilonX 
                if self.endPoint[1] + self.epsilonX <= img.shape[1]
                else self.endPoint[1])

        contrastImg = getContrastImg(img[y:h, x:w])
        text = pytesseract.image_to_string(contrastImg, lang='eng')
        text = re.sub(r"[^a-zA-Z0-9]+", ' ', text)
        self.text = text


    def toWord(self):
        return Word([self.startPoint[0], self.startPoint[1], 
                        self.endPoint[0] - self.startPoint[0],
                        self.endPoint[1] - self.startPoint[1]],
                     50,
                     self.text)


class TextBox:
    def __init__(self, line:Line) -> None:
        self.lines = [line]
        self.lineSize = 1
        self.height = line.height
        self.epsilonHeight = self.height * 0.3
        self.lineSpace = self.height 
        self.textLines = []

    def __getPosition(self, line:Line):
        if (line.startPoint[1] - self.lines[-1].endPoint[1] <= self.lineSpace
            and line.startPoint[1] - self.lines[-1].endPoint[1] > 0
        ):
            if ((abs(self.lines[-1].startPoint[0] - line.startPoint[0]) <= line.epsilonX * 4
                   )
                or (abs(line.endPoint[0] - self.lines[-1].endPoint[0]) <= line.epsilonX * 4
                    )
            ):
                return True
        return False
 
    def insertLine(self, line:Line):
        position = self.__getPosition(line)
        if position:
            self.lines.append(line)
            self.lineSize += 1
            return True
        return False
    
    def show(self):
        text = ''
        l = 0
        for line in self.lines:
            text += line.getText() + ' _' + str(l) + ' '
            l += 1
        tText = googletrans.Translator().translate(text, dest='vi').text
        self.textLines = re.split('_\d', tText)
        for textLine in self.textLines:
            print(textLine,'\n=========================================')
        print('Line: ', len(self.lines))
        print("------------------------------------------------------------------")

    def draw(self, img):
        for i in range(self.lineSize):
            font = ImageFont.truetype(r'font/Arimo-VariableFont_wght.ttf', int(self.lines[i].height))
            img1 = ImageDraw.Draw(img)
            x, y = self.lines[i].startPoint
            x = int(x)
            y = int(y)
            w, h = self.lines[i].endPoint
            w = int(w)
            h = int(h)
            img1.rectangle([(x, y), (w, h)], fill=(255, 255, 255))
            img1.text((x, y),text=self.textLines[i] , font=font,  fill=(0, 0, 0))

def convertToWords(imgData, imgSize):
    words = []
    for i in range(len(imgData['text'])):
        x, y = int(imgData['left'][i]), int(imgData['top'][i])
        w, h = int(imgData['width'][i]), int(imgData['height'][i])
        if(imgData['text'][i] != '' 
            and not imgData['text'][i].isspace()
            and w < imgSize[0]
            and h < imgSize[1]
        ): 
            words.append(Word([x, y, w, h], imgData['conf'][i], imgData['text'][i]))
    
    words = sorted(words, key = lambda key: (key.startPoint[0]))
    return words

def main(imgPath):
    img = cv2.imread('images/' + imgPath)
    imgData = pytesseract.image_to_data(img, lang='eng', output_type=Output.DICT)
    words = convertToWords(imgData, img.shape)

    unAreas = []
    st = time.time()
    for word in words:
        if word.confident < 50:
            inserted = False
            for unArea in unAreas:
                if unArea.insertWord(word):
                    inserted = True
                    break
            if not inserted:
                unAreas.append(UnArea(word))
    for unArea in unAreas:
        unArea.updateText(img)
        words.append(unArea.toWord())


    words = sorted(words, key = lambda key: (key.startPoint[0]))


    lines = []

    for word in words:
        if word.confident >= 50:
            inserted = False
            for line in lines:
                if line.insertWord(word):
                    inserted = True
                    break
            if not inserted:
                lines.append(Line(word))
    lines = sorted(lines, key= lambda x: x.centerPoint[1])

    textBoxs = []


    for line in lines:
            inserted = False
            for textBox in textBoxs:
                if textBox.insertLine(line):
                    inserted = True
                    break 
            if not inserted:
                textBoxs.append(TextBox(line))

    for textBox in textBoxs:
        textBox.show()


    outputImg = Image.open('images/' + imgPath)

    for textBox in textBoxs:
        textBox.draw(outputImg)



    outputImg.save('output/finally_' + imgPath)
    # for  line in lines:
    #     cv2.rectangle(img, line.startPoint, line.endPoint, (255, 0, 255), 1)
        # cv2.putText(img, line.getText(), line.startPoint, 1, 1, (255, 0, 255))

    print("Excuse time: ", time.time() - st)

    cv2.imwrite('output/part_' + imgPath, img)

main('1.png')
main('2.png')
main('3.png')
main('4.png')
# main('5.png')
main('a5.png')
main('a6.png')

# text = googletrans.Translator().translate('Special pages _ Permanent link', dest='vi').text

# print(text)