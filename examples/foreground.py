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


def background_subtractor():

    bg = cv.createBackgroundSubtractorMOG2()

    key = ord('r')
    while key != ord('s'):
        still = webcam.read()
        img_current_gray = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
        img = bg.apply(img_current_gray)
        
        cv.imshow('output', img)
        key = cv.waitKey(5)


def contour_masking():

    cv.namedWindow('controls')

    # threshold trackbar
    cv.createTrackbar('lower', 'controls', 0, 255, lambda *args: None)
    cv.createTrackbar('upper', 'controls', 0, 255, lambda *args: None)

    key = ord('r')
    while key != ord('s'):

        still = webcam.read()
        img = still[1].copy()
        img_gray = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)

        img_blur = cv.GaussianBlur(img_gray, (7,7), 0)

        lower = int(cv.getTrackbarPos('lower', 'controls'))
        upper = int(cv.getTrackbarPos('upper', 'controls'))

        img_canny = cv.Canny(img_blur, lower, upper)
        img_morphed = cv.morphologyEx(img_canny, cv.MORPH_CLOSE, np.ones((5,5)))

        contours, hierarchy = cv.findContours(img_morphed, 
                                              cv.RETR_EXTERNAL, 
                                              cv.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=cv.contourArea)
        mask = np.zeros_like(img_morphed)

        cv.drawContours(mask, 
                        [contours[-1]],
                        -1,
                        255,
                        cv.FILLED,
                        1)

        img[mask == 0] = 0
        cv.imshow('image', img)
        key = cv.waitKey(5)



def watershed():
    # see python opencv tutorial for alternate example: https://docs.opencv.org/4.x/d2/dbd/tutorial_distance_transform.html
    img = cv.imread('sample_images/coins3.jpg')

    og_img = img.copy()

    # Preprocessing
    # grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # gaussian blur
    img_blur = cv.GaussianBlur(img_gray, (7,7), 0)
    # threshold, otsu's
    ret, img_thresh = cv.threshold(img_blur, 
                            0,
                            255,
                            cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    # morph opening
    img_open = cv.morphologyEx(img_thresh,
                               cv.MORPH_OPEN,
                               np.ones((3,3), np.uint8),
                               iterations=2)

    # morph dilate
    bg = cv.dilate(img_open, 
                    np.ones((3,3), np.uint8),
                    iterations=3)

    # distance transform
    distance_transform = cv.distanceTransform(img_open, cv.DIST_L2, 5)
    # threshold dist. transform output
    ret, fg = cv.threshold(distance_transform, 0.1*distance_transform.max(), 255, 0)

    # calculate diff between foreground, background
    # outline shape of coins (topology for watershed)
    fg = np.uint8(fg)
    unknown = cv.subtract(bg, fg)

    # use connected components to find markers/labels
    #   gives us "regions" as connected components which are labeled
    #   could also use contours, see opencv watershed docs/tutorial
    ret, markers = cv.connectedComponents(fg)
    print('cc markers', markers)
    # make background 1, not 0
    markers = markers + 1
    # where bg - fg = 255 (unknown) mark as 0
    markers[unknown == 255] = 0
    # recalculate with watershed
    markers = cv.watershed(og_img, markers)
    og_img[markers == -1] = [0, 255, 255]

    key = ord('r')
    while key != ord('s'):

        cv.imshow('opening', img_open)
        cv.imshow('background', bg)
        cv.imshow('foreground', fg)
        cv.imshow('distance transform', distance_transform)
        cv.imshow('unknown', unknown)
        cv.imshow('final', og_img)
        key = cv.waitKey()

    cv.destroyAllWindows()


if __name__ == '__main__':
    #connected()
    #grabcut()
    #contour_masking()
    #background_subtractor()
    watershed()


    



