import cv2 as cv
import numpy as np
import mv_library
from matplotlib import pyplot as plt

img = cv.imread('sample_images/octo1.jpg')
#img = cv.imread('sample_images/boat1.jpg')

# Calcualting individual histogram
# Convert Image to Grayscale
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Pixel Luminosity (Grayscale)
#   from this: https://www.pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/
histogram = cv.calcHist(images=[img], 
                        channels=[0], 
                        mask=None, 
                        histSize=[256], 
                        ranges=[0,256])

plt.figure()
plt.title("Luminosity Histogram")
plt.xlabel("Bins")
plt.ylabel("Number of Pixels in each bin")
plt.plot(histogram)
plt.xlim([0,256])
plt.show()


# Equalize Histogram
equalized_img = cv.equalizeHist(img)
cv.imwrite("output_images/equalized_octo1.png", equalized_img)


key = ord("r")
while key != ord("s"):
    cv.imshow("Image", img)
    key = cv.waitKey()

cv.imwrite('output_images/gray_octo1.png', img)
