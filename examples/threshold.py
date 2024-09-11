import cv2 as cv
import numpy as np


img = cv.imread('sample_images/cross1.jpg')

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#ret, img_out = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
#ret, img_out = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY_INV)
#ret, img_out = cv.threshold(img_gray, 127, 255, cv.THRESH_TRUNC)
#ret, img_out = cv.threshold(img_gray, 127, 255, cv.THRESH_TOZERO)
#ret, img_out = cv.threshold(img_gray, 127, 255, cv.THRESH_TOZERO_INV)

'''
img_out = cv.adaptiveThreshold(img_gray, 
                               255, 
                               cv.ADAPTIVE_THRESH_MEAN_C, 
                               cv.THRESH_BINARY,
                               11,
                               2)
'''

img_out = cv.adaptiveThreshold(img_gray, 
                               255, 
                               cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv.THRESH_BINARY,
                               11,
                               2)

key=ord('r')
while key != ord('s'):
    cv.imshow('original img', img)
    cv.imshow('threshold output', img_out)
    key=cv.waitKey()


