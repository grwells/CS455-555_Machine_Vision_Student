import cv2 as cv
import numpy as np

webcam = cv.VideoCapture(0)


def connected():
    img = cv.imread('sample_images/license.jpg')
    og_img = img.copy()

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, img_thresh = cv.threshold(img_gray,
                                   0,
                                   255,
                                   cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    
    (numlabels, labels, stats, centroids) = cv.connectedComponentsWithStats(
                                                img_thresh,
                                                8, 
                                                cv.CV_32S
    )

    for i in range(0, numlabels):
        x =  stats[i, cv.CC_STAT_LEFT]
        y =  stats[i, cv.CC_STAT_TOP]
        w =  stats[i, cv.CC_STAT_WIDTH]
        h =  stats[i, cv.CC_STAT_HEIGHT]

        cv.rectangle(og_img, (x,y), (x+w, y+h), (0, 255, 0), 3)

    
    key = ord('r')
    while key != ord('s'):
        cv.imshow('image', og_img)
        key = cv.waitKey(5)


def grabcut():

    img = cv.imread('sample_images/robot.jpg')
    img = cv.resize(img, 
                    (int(img.shape[1]*0.20), int(img.shape[0]*0.20)), 
                    interpolation=cv.INTER_AREA)

    mask = np.zeros(img.shape[:2], np.uint8)

    backgndModel = np.zeros((1,65), np.float64)
    foregndModel = np.zeros((1,65), np.float64)


    #rect = (0,0, img.shape[0], img.shape[1])
    rect = (133, 185, 394, 489)

    cv.grabCut(img,
               mask, 
               rect, 
               backgndModel, 
               foregndModel,
               3,
               cv.GC_INIT_WITH_RECT)

    mask_out = np.where((mask==cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')

    img_out = img * mask_out[:,:, np.newaxis]

    key = ord('r')
    while key != ord('s'):
        cv.imshow('image', img_out)
        key = cv.waitKey(5)

grabcut()


