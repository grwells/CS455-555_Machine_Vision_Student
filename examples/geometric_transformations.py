#!/usr/bin/env python3
import numpy as np
import cv2 as cv
import mv_library 

#Help from here
#https://docs.opencv.org/4.5.2/d3/df2/tutorial_py_basic_ops.html
#https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
#https://learnopencv.com/image-resizing-with-opencv/
#https://docs.opencv.org/4.5.2/da/d6e/tutorial_py_geometric_transformations.html

#For later: 
#https://www.geeksforgeeks.org/how-to-display-multiple-images-in-one-window-using-opencv-python/

#img = cv.imread('./sample_images/octo1.jpg')
img = cv.imread('./sample_images/boat1.jpg')
rows, columns, channels = img.shape
mv_library.showImage(img)

# Translation 
img = cv.imread('./sample_images/boat1.jpg')
rows, columns, channels = img.shape
M = np.float32([[1, 0, 100],[0,1,50]])

img = cv.warpAffine(img, M, (columns, rows))
mv_library.showImage(img)

# Rotation
#getRotationMatrix2D takes the center, angle, and scale 
#https://docs.opencv.org/4.5.2/da/d54/group__imgproc__transform.html#gafbbc470ce83812914a70abfb604f4326

img = cv.imread('./sample_images/boat1.jpg')
rows, columns, channels = img.shape
#center of rotation, angle of rotation, scale 
M = cv.getRotationMatrix2D(((columns-1)/2.0, (rows-1)/2.0), 90, 1)
img = cv.warpAffine(img, M, (columns, rows))
mv_library.showImage(img)


# Shear
# a01 should be shx, a10 should be shy
img = cv.imread('./sample_images/boat1.jpg')
rows, columns, channels = img.shape
M = np.float32([
                [1, 0.5, 0],
                [0.5, 1, 0 ]
            ])
img = cv.warpAffine(img, M, (columns, rows))
mv_library.showImage(img)


# Resize an image
# Can mess with the interpolation

img = cv.imread('./sample_images/boat1.jpg')
rows, columns, channels = img.shape
new_img = cv.resize(img.copy(), (int(img.shape[1]*1.5), int(img.shape[0]*1.5)), interpolation=cv.INTER_NEAREST)
cv.imwrite('./sample_images/examples/inter_nearest.jpg', new_img)
new_img = cv.resize(img.copy(), (int(img.shape[1]*1.5), int(img.shape[0]*1.5)), interpolation=cv.INTER_LINEAR)
cv.imwrite('./sample_images/examples/inter_linear.jpg', new_img)
new_img = cv.resize(img.copy(), (int(img.shape[1]*1.5), int(img.shape[0]*1.5)), interpolation=cv.INTER_CUBIC)
cv.imwrite('./sample_images/examples/inter_cubic.jpg', new_img)
new_img = cv.resize(img.copy(), (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), interpolation=cv.INTER_AREA)
mv_library.showImage(img)
cv.imwrite('./sample_images/examples/inter_area_down.jpg', new_img)

#img = cv.imread('./sample_images/boat1.jpg')
#rows, columns, channels = img.shape
def scale_down(img, scale_value):
    row_dim = int(img.shape[0] * scale_value)
    col_dim = int(img.shape[1] * scale_value)
    img = cv.resize(img, (col_dim, row_dim), interpolation=cv.INTER_LINEAR)
    return img



#img = scale_down(img, 0.5)
#img = cv.resize(img.copy(), None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
#img = cv.resize(img.copy(), None, fx=0.1, fy=0.1, interpolation=cv.INTER_CUBIC)
#mv_library.showImage(img)

#General Affine Transformation

img = cv.imread('./sample_images/boat1.jpg')
rows, columns, channels = img.shape
first_pts = np.float32([[50,50],[200, 50],[50,200]])
next_pts = np.float32([[10,100],[200, 50],[100,250]])
M = cv.getAffineTransform(first_pts, next_pts)
img = cv.warpAffine(img, M, (columns, rows))
mv_library.showImage(img)


#Perspective Transform

img = cv.imread('./sample_images/boat1.jpg')
rows, columns, channels = img.shape
first_pts = np.float32([[50,50],[400, 50],[50,400], [400, 400]])
#next_pts = np.float32([[0,0],[600, 0],[0,600], [600, 600]])
next_pts = np.float32([[50, 0],[600, 0],[50, 50], [600, 600]])

M = cv.getPerspectiveTransform(first_pts, next_pts)
img = cv.warpPerspective(img, M, (columns, rows))
mv_library.showImage(img)


# Splitting output into channels

img = cv.imread('./sample_images/boat1.jpg')
rows, columns, channels = img.shape
#Splits into blue, green, red channels
b, g, r = cv.split(img)
mv_library.showImage(b, frames=1000)
mv_library.showImage(g, frames=1000)
mv_library.showImage(r, frames=1000)


#Merges the channels back in

img = cv.merge((b, g, r))
mv_library.showImage(img, frames=1000)


# Channels using numpy

img = cv.imread('./sample_images/boat1.jpg')
rows, columns, channels = img.shape
#Numpy way to split into channels (blue in this case)
#Numpy is faster
# b = img[:,:,0]
# mv_library.showImage(b, frames=1000)

# #Set all red pixels to 0
#Numpy way
img[:,:,2] = 0
mv_library.showImage(img)



# Brightness + Contrast 
#img = cv.imread('./sample_images/boat1.jpg')
#rows, columns, channels = img.shape
#Brightness and Contrast
#output pixel = (input pixel * Alpha) + Beta
#Alpha = contrast = gain
#Beta = brightness = bias

img = cv.imread('./sample_images/octo1.jpg')
alpha = 2.2
beta = 20.0

# for row_pixel in range(0, img.shape[0]):
#     for column_pixel in range(0, img.shape[1]):
#         for channel_pixel in range(0, img.shape[2]):
#             #Non-clipping. Pretty bad.
#             #img[row_pixel][column_pixel][channel_pixel] = img[row_pixel][column_pixel][channel_pixel]*alpha + beta
#             #np clip function. Better but takes quite a while. 
#             img[row_pixel][column_pixel][channel_pixel] = np.clip(alpha*img[row_pixel][column_pixel][channel_pixel] + beta, 0, 255)

# mv_library.showImage(img)
#Or
img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
mv_library.showImage(img)

img = cv.imread('./sample_images/boat1.jpg')
#Gamma Correction
gamma = 1.0
look_up = np.empty((1,256), np.uint8)
for i in range(256):
    look_up[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
img = cv.LUT(img, look_up)

mv_library.showImage(img)
