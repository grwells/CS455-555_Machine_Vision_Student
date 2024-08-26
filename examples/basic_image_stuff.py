import cv2 as cv

'''
img = cv.imread('./sample_images/octo1.jpg')

key = ord('r')
while key != ord('s'):

    cv.imshow("Octopus", img)
    key = cv.waitKey()

cv.destroyAllWindows()
'''
'''
webcam = cv.VideoCapture(0)
key = ord('r')
while key != ord('s'):
    still = webcam.read()
    print(still)
    cv.imshow("Webcam", still[1])
    key = cv.waitKey(10)

cv.destoryAllWindows()

'''

# Pixel Values
#
'''
img = cv.imread('./sample_images/trex.jpg')
print(img[40, 40, 0])

print(img.shape)
print(img.dtype)
'''
'''
# Image Borders
#
img = cv.imread('./sample_images/octo1.jpg')
img = img[250:500, 400:700]

border_size = 20

border_color = [255, 0, 0]
'''

'''
img = cv.copyMakeBorder(img, 
                        border_size, 
                        border_size, 
                        border_size, 
                        border_size, 
                        cv.BORDER_CONSTANT, 
                        value=border_color)
img = cv.copyMakeBorder(img, 
                        border_size, 
                        border_size, 
                        border_size, 
                        border_size, 
                        cv.BORDER_REPLICATE,
                        value=border_color)
img = cv.copyMakeBorder(img, 
                        border_size, 
                        border_size, 
                        border_size, 
                        border_size, 
                        cv.BORDER_REFLECT,
                        value=border_color)
img = cv.copyMakeBorder(img, 
                        border_size, 
                        border_size, 
                        border_size, 
                        border_size, 
                        cv.BORDER_WRAP,
                        value=border_color)

key = ord('r')
while key != ord('s'):
    cv.imshow('Bordered Octopus', img)
    key = cv.waitKey()

cv.destroyAllWindows()

path = './output_images/octo_gw.png'
cv.imwrite(path, img)
'''

img = cv.imread('./sample_images/octo1.jpg')
def write(img, text, 
          org=(50,50), 
          font=cv.FONT_HERSHEY_SIMPLEX, 
          fontScale=1, color=(255, 0, 0), 
          thickness=2,
          line_type=cv.LINE_AA):

    newImg
