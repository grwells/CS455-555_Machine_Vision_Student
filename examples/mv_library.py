#!/usr/bin/env python3
import cv2 as cv
#from deepface import DeepFace
import os
import numpy as np

#Help from here: https://newbedev.com/how-to-iterate-over-files-in-a-given-directory


#Pull a single still image from the webcam in use.
def capture_still(webcam):
    img = webcam.read() 
    still = img[1].copy()
    return still

#Uses deepface's library function to verify that the face in one image is the same as the face in another.
def recognize_face(img1_path, img2_path):
    #verification = DeepFace.verify(img1_path, img2_path, enforce_detection=False)
    try:
        verification = DeepFace.verify(img1_path, img2_path, enforce_detection=True)
        print(verification)
        if verification['verified']:
            return "Verified"
        else:
            return "Imposter"
    except:
        return "No Face Detected"


def verify_face_by_name(webcam, name):

    path = "/home/marz/Documents/vision/sample_images/"
    save_path = path + "test_face.jpg"
    print("Capturing new image")
    img = capture_still(webcam)
    cv.imwrite(save_path, img)
    
    ver_path = "/home/marz/Documents/vision/sample_images/users/" + name + ".jpg"
    status = recognize_face(img1_path=ver_path, img2_path=save_path)        
    if status == "Verified":
                write_string = status + ": " + name
                newImage = write(img.copy(), write_string)
    else:
        newImage = write(img.copy(), status)
    return newImage

def verify_face_search(webcam):
    path = "/home/marz/Documents/vision/sample_images/"
    save_path = path + "test_face.jpg"

    print("Capturing new image")
    img = capture_still(webcam)
    cv.imwrite(save_path, img)
    
    directory = "/home/marz/Documents/vision/sample_images/users/"
    for filename in os.listdir(directory):
            print(filename)
            print(save_path)
            search_file = directory + filename
            status = recognize_face(img1_path=search_file, img2_path=save_path)
            
            if status == "Verified":
                write_string = status + ": " + filename[:-4]
                newImage = write(img.copy(), write_string)
                return newImage, filename[:-4]

    newImage = write(img, status)
    return newImage, status

#Takes picture of a face, attempts to verify it, writes the verification status on the image and returns it. 
def verify_face_cam(webcam):

    path = "/home/marz/Documents/vision/sample_images/"
    save_path = path + "test_face.jpg"
    
    print("Capturing new image")
    img = capture_still(webcam)
    cv.imwrite(save_path, img)

    status = recognize_face("/home/marz/Documents/vision/sample_images/marz.jpg", img2_path = save_path)
    newImage = write(img, status)
    return newImage

#Scale down an image by a factor
def scale_down(img, scale_value):
    row_dim = int(img.shape[0] * scale_value)
    col_dim = int(img.shape[1] * scale_value)
    img = cv.resize(img, (col_dim, row_dim), interpolation=cv.INTER_LINEAR)
    return img

#Shows an image indefinitely (default) or for a specified time in a window with default name "Image"
def showImage(img, frames=-1.0, window="Image"):
    k = ord('r')
    if frames < 0:
        while(k != ord('s')):
            #Show that frame in the window   
            cv.imshow(window, img)
            #Waits muy briefly
            k = cv.waitKey(10) 
    else:
        cv.imshow(window, img)
        k = cv.waitKey(frames)

#Writes text to an image with the defaults listed for text parameters. 
def write(img, text, org=(50, 50), font=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2, line_type=cv.LINE_AA):
    newImg = cv.putText(img, text, org, font, fontScale, color, thickness, line_type)
    return newImg

#Inner function must take only one or no arguments and return an image to use this decorator!
#This is a continuous stream. 
#Should be used for a continuous function/image processing item
def stream_decorator(func, window="Image", param="None"):
    k = ord('r')
    while (k != ord('s')):
        if param == "None":
            img = func()
        else:
            img = func(param)
        print("Showing new image")
        cv.imshow(window, img)
        k = cv.waitKey(2)


def multiple_window_test(webcam):
    k = ord('r')
    while (k != ord('s')):
        img = capture_still(webcam)
        cv.imshow("Press s to take picture", img)
        k = cv.waitKey(2)
    cv.destroyAllWindows()
    resultImg, name_status = verify_face_search(webcam)
    if name_status != "Imposter" and name_status != "No Face Detected":
        ref_string = "/home/marz/Documents/vision/sample_images/users/" + name_status + ".jpg"
        ref_img = cv.imread(ref_string)
        write_string = "Verified: " + name_status
        write(ref_img, write_string)
    else:
        ref_img = resultImg
        #stack_top = np.concatenate((img, ref_img))
        #stack_side = np.concatenate((img, ref_img))
    print(img.shape)
    ref_img = scale_down(ref_img, 0.2)
    print(ref_img.shape)
    img[10:ref_img.shape[0]+10,10:ref_img.shape[1]+10, 0:3] = ref_img

    k = ord('r')
    while (k != ord('s')):
        cv.imshow("Result", img)
        k = cv.waitKey(2)

def main():
    webcam = cv.VideoCapture(0)
    multiple_window_test(webcam)

if __name__=="__main__":
    main()