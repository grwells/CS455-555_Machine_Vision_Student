import cv2 as cv
import numpy as np

#img = cv.imread('sample_images/boat1.jpg')
img = cv.imread('sample_images/octo1.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray = cv.GaussianBlur(gray, (5,5), 0)

# Sobel Kernel
# ddepth = -1, keep original 
'''
output_img = cv.Sobel(gray, 
                      ddepth=-1,
                      dx=2,
                      dy=2,
                      ksize=5)
'''

# Scharr
# can't have dx & dy == 1
'''
output_img = cv.Scharr(gray,
                       -1,
                       0,
                       1)
'''

# Laplacian
output_img = cv.Laplacian(gray, cv.CV_64F)

key = ord('r')

while key != ord('s'):
    cv.imshow('original gray', gray)
    cv.imshow('output', output_img)
    key = cv.waitKey()

cv.destroyAllWindows()



