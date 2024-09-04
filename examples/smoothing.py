import cv2 as cv
import numpy as np

import mv_library

img = cv.imread('sample_images/octo1.jpg')

# Average Blur
#blurred = cv.blur(img, (15, 15), 0)

# Gaussian Blur
#blurred = cv.GaussianBlur(img, (15, 15), 0)

# Median Blur
#blurred = cv.medianBlur(img, 15)

# Bilateral Filtering
blurred = cv.bilateralFilter(img, 11, 61, 39)

key = ord('r')
while key != ord('s'):
    cv.imshow('Blurred', blurred)
    key = cv.waitKey()

key = ord('r')
while key != ord('s'):
    cv.imshow('Original', img)
    key = cv.waitKey()


