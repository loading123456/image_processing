from statistics import quantiles
import cv2
import numpy as np


def threshold(imgPath, savePath):
    img = cv2.imread(imgPath)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binaryImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_OTSU 
                                            | cv2.THRESH_BINARY_INV)[1]
    cv2.imwrite(savePath, binaryImg)


threshold('images/1.png', 'images/threshold_1.png')
threshold('images/2.png', 'images/threshold_2.png')
threshold('images/3.png', 'images/threshold_3.png')
threshold('images/4.png', 'images/threshold_4.png')
threshold('images/5.png', 'images/threshold_5.png')
threshold('images/6.png', 'images/threshold_6.png')
threshold('images/7.png', 'images/threshold_7.png')
threshold('images/8.png', 'images/threshold_8.png')
threshold('images/9.png', 'images/threshold_9.png')