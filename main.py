# Import required packages
import cv2
import numpy as np
import pytesseract




def isInclude(p, c):
    if (p[0] <= c[0] 
        and p[1] <= c[1] 
        and p[2] >= c[2] 
        and p[3] >= c[3]
    ):
        return True
    return False

def convertToRect(a):
    r = []
    for i in a:
        r.append(cv2.boundingRect(i))
 
    return r

def filterBox(a, b):

    r0 = convertToRect(a)
    r1 = convertToRect(b)

    for i in r0:
        for j in r1:
            if isInclude(i, j):
                r0.remove(i)
                break
    for i in r1:
        for j in r0:
            if isInclude(i, j):
                r1.remove(i)
                break
    return r0 + r1




# Mention the installed location of Tesseract-OCR in your system
img_path = '1.png'
# Read image from which text needs to be extracted
img = cv2.imread("images/" + img_path)

# Preprocessing the image starts

# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imwrite('gray2.jpg', gray)

# Performing OTSU threshold
ret, thresh0 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

thresh1 = np.copy(thresh0)

for i in range(len(thresh0)):
    for j in range(len(thresh0[i])):
        if thresh0[i][j] == 0:
            thresh1[i][j] = 255
        else:
            thresh1[i][j] = 0
            


text = pytesseract.image_to_string(gray)
print(text)



rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

# print(thresh0)
# print(type(thresh0))
# print(type(thresh1))
# Applying dilation on the threshold image
dilation0 = cv2.dilate(thresh0, rect_kernel, iterations = 1)
dilation1 = cv2.dilate(thresh1, rect_kernel, iterations = 1)

# # Finding contours
contours0, hierarchy = cv2.findContours(dilation0, cv2.RETR_EXTERNAL,
												cv2.CHAIN_APPROX_NONE)
contours1, hierarchy1 = cv2.findContours(dilation1, cv2.RETR_EXTERNAL,
												cv2.CHAIN_APPROX_NONE)
# print(len(contours0))
# print(len(contours1))

    


# r_len = len(r)
# contours0_len = len(contours0)
# print(len(r[0]))
# # Creating a copy of image
im2 = img.copy()


# im2_size = im2.shape

# Looping through the identified contours
# Then rectangular part is cropped and passed on
# to pytesseract for extracting text from it
# Extracted text is then written into the text file

r = filterBox(contours0, contours1)

for i in r:
    x, y, w, h = i
  
    cropped = np.copy(img[y:y + h, x:x + w])

    text = pytesseract.image_to_string(cropped)

    rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print(len(text), ': ' + text)




cv2.imwrite('output/' + img_path,img)