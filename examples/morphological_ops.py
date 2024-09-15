import cv2 as cv
import numpy as np

webcam = cv.VideoCapture(0)


def morph_ops():

    cv.namedWindow('controls')
    cv.createTrackbar('threshold', 
                      'controls',
                      0,
                      255,
                      lambda *args: None)

    cv.setTrackbarPos('threshold', 'controls', 127)

    cv.createTrackbar('blur', 
                      'controls',
                      3,
                      100,
                      lambda *args: None)

    cv.setTrackbarPos('blur', 'controls', 3)

    cv.createTrackbar('erosion', 
                      'controls',
                      0,
                      30,
                      lambda *args: None)

    cv.setTrackbarPos('erosion', 'controls', 15)

    cv.createTrackbar('dilation', 
                      'controls',
                      0,
                      30,
                      lambda *args: None)

    cv.setTrackbarPos('dilation', 'controls', 15)

    # morph toggles
    cv.createTrackbar('dilation toggle', 
                      'controls',
                      0,
                      1,
                      lambda *args: None)
    cv.setTrackbarPos('dilation toggle', 'controls', 0)

    cv.createTrackbar('erosion toggle', 
                      'controls',
                      0,
                      1,
                      lambda *args: None)

    cv.setTrackbarPos('erosion toggle', 'controls', 0)



    key = ord('r')

    while key != ord('s'):
        still = webcam.read()
        img_gray = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)

        # ops
        thresh = int(cv.getTrackbarPos('threshold', 'controls'))
        ret, img = cv.threshold(img_gray, 
                                thresh, 
                                255, 
                                cv.THRESH_BINARY)

        blur_level = int(cv.getTrackbarPos('blur', 'controls'))
        if blur_level % 2 == 0:
            blur_level += 1

        img = cv.GaussianBlur(img, (blur_level, blur_level), 0)

        # morph ops
        dilation = int(cv.getTrackbarPos('dilation', 'controls'))
        dilation_tog = int(cv.getTrackbarPos('dilation toggle', 'controls'))

        erosion = int(cv.getTrackbarPos('erosion', 'controls'))
        erosion_tog = int(cv.getTrackbarPos('erosion toggle', 'controls'))

        if erosion_tog:
            img = cv.erode(img,
                            np.ones((erosion, erosion),
                            dtype=int))

        if dilation_tog:
            img = cv.dilate(img, 
                            np.ones((dilation, dilation), 
                            dtype=int))


        cv.imshow('output img', img)
        key = cv.waitKey(5)


morph_ops()
