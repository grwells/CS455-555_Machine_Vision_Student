import cv2 as cv
import numpy as np

key = ord('r')

webcam = cv.VideoCapture(0)

img = cv.imread('sample_images/octo1.jpg')

while key != ord('s'):
    still = webcam.read()

    #gray = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #gray = cv.applyColorMap(still[1], cv.COLORMAP_RAINBOW)
    cv.imshow('Grayscale', gray)
    cv.imshow('Original', img)
    key = cv.waitKey()
    cv.imwrite('output_images/octo1_gray.jpg', gray)



