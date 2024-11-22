import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt

webcam = cv.VideoCapture(0)


def pyramids():
    img = cv.imread('sample_images/octo1.jpg')

    layer = img.copy()

    for i in range(4):
        plt.subplot(2,2, i+1)
        layer = cv.pyrDown(layer)

        plt.imshow(layer)
        cv.imshow(str(i), layer)
        cv.waitKey(2000)

    cv.destroyAllWindows()


#pyramids()

def fourier():

    img = cv.imread('sample_images/octo1.jpg', 0)

    img_float32 = np.float32(img)

    dft = cv.dft(img_float32, flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],
                                                dft_shift[:,:,1]))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')

    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    plt.show()

fourier()
