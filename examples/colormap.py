import cv2 as cv
import numpy as np

key = ord('r')

webcam = cv.VideoCapture(0)

#img = cv.imread('sample_images/octo1.jpg')

while key != ord('s'):
    still = webcam.read()

    #og_img = still[1].copy()

    #gray = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
    gray = cv.applyColorMap(still[1], cv.COLORMAP_RAINBOW)
    cv.imshow('Grayscale', gray)
    cv.imshow('Original', still[1])
    key = cv.waitKey()
