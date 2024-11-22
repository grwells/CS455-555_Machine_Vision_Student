#!/usr/bin/env python3
from deepface import DeepFace
import cv2 as cv
import mv_library

#Unnecessary now. 
path = "./sample_images/"
save_path = path + "test_face.jpg"

#Set up webcam
webcam = cv.VideoCapture(0)
#Stream the face verification
mv_library.stream_decorator(mv_library.verify_face_cam, param=webcam)
#verify_face_cam(webcam)
