import cv2 as cv
import numpy as np


webcam = cv.VideoCapture(0)
img = cv.imread('sample_images/coins2.jpg')

def sobel():
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (5,5), 0)

    img_sobel = cv.Sobel(img_blur, -1, 1, 1, ksize=5)
    
    key = ord('r')

    while key != ord('s'):

        cv.imshow('original', img)
        cv.imshow('sobel', img_sobel)
        key = cv.waitKey()
    
def canny():
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (5,5), 0)

    img_canny = cv.Canny(img_blur, 50, 200)
    
    key = ord('r')

    while key != ord('s'):

        cv.imshow('original', img)
        cv.imshow('canny', img_canny)
        key = cv.waitKey()

def auto_canny():
    cv.namedWindow('controls')
    cv.createTrackbar('lower', 
                      'controls', 
                      0, 
                      255, 
                      lambda *args: None)

    cv.createTrackbar('upper', 
                      'controls', 
                      0, 
                      255, 
                      lambda *args: None)

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (5,5), 0)
    v = np.median(img_blur)
    sigma = 0.33
    
    key = ord('r')

    while key != ord('s'):


        #lower = int(cv.getTrackbarPos('lower', 'controls'))
        lower = int(max(0, (1.0 - sigma)*v))
        #upper = int(cv.getTrackbarPos('upper', 'controls'))
        upper = int(max(0, (1.0 + sigma)*v))

        img_canny = cv.Canny(img_blur, lower, upper)



        cv.imshow('original', img)
        cv.imshow('canny', img_canny)
        key = cv.waitKey(5)

def contours():
    cv.namedWindow('controls')
    cv.createTrackbar('lower', 
                      'controls', 
                      0, 
                      255, 
                      lambda *args: None)

    cv.createTrackbar('upper', 
                      'controls', 
                      0, 
                      255, 
                      lambda *args: None)


    key = ord('r')

    while key != ord('s'):

        still = webcam.read()
        og_img = still[1].copy()

        img_gray = cv.cvtColor(og_img, cv.COLOR_BGR2GRAY)
        img_blur = cv.GaussianBlur(img_gray, (5,5), 0)

        lower = int(cv.getTrackbarPos('lower', 'controls'))
        upper = int(cv.getTrackbarPos('upper', 'controls'))

        img_canny = cv.Canny(img_blur, lower, upper)

        contours, hierarchy = cv.findContours(img_canny, 
                                              cv.RETR_TREE,
                                              cv.CHAIN_APPROX_SIMPLE)

        
        cv.drawContours(og_img, contours, -1, (255, 0, 0), 3)
        img = og_img

        cv.imshow('out', img)
        key = cv.waitKey(5)


#auto_canny()
contours()
